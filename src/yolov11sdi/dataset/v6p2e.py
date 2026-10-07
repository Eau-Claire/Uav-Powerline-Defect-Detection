"""V6.2E parent: fetch, validate against config, stream members without extraction.

Low-disk policy (from the Errno 28 incident):
* never fully extract base/overlay archives;
* read images/labels via tarfile.extractfile() using the split manifest;
* materialize at most `bg_pool_max` TRAIN backgrounds;
* pHash VAL/TEST once into parent_val_test_phash.csv.gz (incremental);
* then close handles and (optionally) delete the large archives.
"""
from __future__ import annotations

import csv
import hashlib
import io
import logging
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..archive_utils import build_tar_suffix_index
from ..io_utils import atomic_write_bytes, atomic_write_csv, atomic_write_text, parse_yolo_text

log = logging.getLogger("yolov11sdi.v6p2e")


class ParentConflict(RuntimeError):
    """Frozen parent artifacts disagree with the configured expectations."""


# ------------------------------------------------------------- remotes

def parent_dir_remote(cfg) -> str:
    return f"{cfg.get('v6p2e.root')}/v6p2e/{cfg.get('v6p2e.tag')}"


def metadata_remotes(cfg) -> dict[str, str]:
    d = parent_dir_remote(cfg)
    files = cfg.get("v6p2e.files")
    return {k: f"{d}/{v}" for k, v in files.items()}


def archive_remotes(cfg, freeze: dict) -> dict[str, str]:
    fmt = {
        "root": cfg.get("v6p2e.root"),
        "base_tag": freeze["base_tag"],
        "overlay_tag": freeze["overlay_tag"],
        "parent_dataset_tag": freeze["parent_dataset_tag"],
    }
    return {
        "base_archive": cfg.get("v6p2e.base_archive").format(**fmt),
        "overlay_archive": cfg.get("v6p2e.overlay_archive").format(**fmt),
    }


# ----------------------------------------------------------- validation

def validate_parent(cfg, freeze: dict, manifest: pd.DataFrame) -> list[str]:
    """Return human-readable conflicts between frozen artifacts and config."""
    conflicts = []
    tag = cfg.get("v6p2e.tag")
    frozen_tag = freeze.get("final_tag") or freeze.get("tag")
    if frozen_tag and frozen_tag != tag:
        conflicts.append(f"freeze final_tag={frozen_tag!r} but config v6p2e.tag={tag!r}")
    for k, v in cfg.get("v6p2e.expected_lineage").items():
        if str(freeze.get(k)) != str(v):
            conflicts.append(f"freeze {k}={freeze.get(k)!r} but config expects {v!r}")
    names = freeze.get("taxonomy")
    if names is not None and list(names) != cfg.names:
        conflicts.append(f"freeze taxonomy={names} != config taxonomy")
    counts = manifest["split"].astype(str).value_counts().to_dict()
    exp = cfg.get("v6p2e.expected_counts")
    for split in ("train", "val", "test"):
        if int(counts.get(split, 0)) != int(exp[split]):
            conflicts.append(f"manifest {split}={counts.get(split, 0)} but expected {exp[split]}")
    if len(manifest) != int(exp["total"]):
        conflicts.append(f"manifest rows={len(manifest)} but expected total {exp['total']}")
    fsc = freeze.get("split_counts")
    if isinstance(fsc, dict):
        for split in ("train", "val", "test"):
            if split in fsc and int(fsc[split]) != int(counts.get(split, 0)):
                conflicts.append(f"freeze split_counts.{split}={fsc[split]} != manifest {counts.get(split, 0)}")
    required = {"split", "sha256", "image_path_source", "previous_v6p1_split", "resolution"}
    missing = required - set(manifest.columns)
    if missing:
        conflicts.append(f"manifest missing columns {sorted(missing)}")
    return conflicts


def load_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


# --------------------------------------------------------------- reader

@dataclass(frozen=True)
class SourceKeys:
    source: str
    rel: str
    img_key: str
    lbl_key: str


def row_source_keys(row) -> SourceKeys:
    """Map a manifest row to archive member keys (legacy provenance rule)."""
    old_path = str(row["image_path_source"]).replace("\\", "/")
    old_split = str(row["previous_v6p1_split"])
    marker = f"/images/{old_split}/"
    if marker not in old_path:
        raise RuntimeError(f"Bad manifest source path: {old_path}")
    suffix = old_path.split(marker, 1)[1]
    if suffix.startswith("base/"):
        source, rel = "base", suffix[len("base/"):]
    elif suffix.startswith("fotl/"):
        source, rel = "fotl", suffix[len("fotl/"):]
    else:
        raise RuntimeError(f"Unknown provenance suffix: {suffix}")
    label_rel = Path(rel).with_suffix(".txt").as_posix()
    return SourceKeys(source, rel, f"images/{old_split}/{rel}", f"labels/{old_split}/{label_rel}")


class ParentReader:
    """Random-access reader over base TAR + FOTL overlay TAR + reconciled label ZIP."""

    def __init__(self, base_tar: Path, overlay_tar: Path, recon_zip: Path):
        self.base_tar_path, self.overlay_tar_path, self.recon_zip_path = base_tar, overlay_tar, recon_zip
        self._base = self._overlay = None
        self._recon = None

    def __enter__(self) -> "ParentReader":
        self._base = tarfile.open(self.base_tar_path, "r:*")
        self._overlay = tarfile.open(self.overlay_tar_path, "r:*")
        self._recon = zipfile.ZipFile(self.recon_zip_path, "r")
        log.info("indexing parent archives (headers only)...")
        self.base_idx, d1 = build_tar_suffix_index(self._base)
        self.overlay_idx, d2 = build_tar_suffix_index(self._overlay)
        if d1 or d2:
            log.warning("duplicate archive suffix keys: base=%d overlay=%d", d1, d2)
        self.recon_idx = {Path(n).name: n for n in self._recon.namelist() if n.endswith(".txt")}
        log.info("indexed base=%d overlay=%d recon=%d", len(self.base_idx), len(self.overlay_idx), len(self.recon_idx))
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        for h in (self._base, self._overlay, self._recon):
            try:
                if h is not None:
                    h.close()
            except Exception:  # noqa: BLE001
                pass
        self._base = self._overlay = self._recon = None
        for attr in ("base_idx", "overlay_idx", "recon_idx"):
            if hasattr(self, attr):
                getattr(self, attr).clear()

    def _handles(self, source: str):
        if source == "base":
            return self._base, self.base_idx
        if source == "fotl":
            return self._overlay, self.overlay_idx
        raise KeyError(source)

    def _read_member(self, source: str, key: str) -> bytes:
        tf, idx = self._handles(source)
        member = idx.get(key)
        if member is None:
            raise KeyError(f"Missing archive member: {source}:{key}")
        f = tf.extractfile(member)
        if f is None:
            raise RuntimeError(f"Cannot extract member: {member.name}")
        return f.read()

    def member_offset(self, row) -> tuple[str, int]:
        k = row_source_keys(row)
        _, idx = self._handles(k.source)
        m = idx.get(k.img_key)
        return (k.source, m.offset_data if m is not None else -1)

    def read_image_bytes(self, row) -> bytes:
        k = row_source_keys(row)
        return self._read_member(k.source, k.img_key)

    def read_label_text(self, row) -> str:
        """Reconciled label for merged rows, else the original archive label."""
        if str(row.get("resolution", "")).startswith("merged"):
            zn = self.recon_idx.get(f"{row['sha256']}.txt")
            if zn is not None:
                return self._recon.read(zn).decode("utf-8", errors="ignore")
        k = row_source_keys(row)
        return self._read_member(k.source, k.lbl_key).decode("utf-8", errors="ignore")


# ------------------------------------------------------ background pool

def pool_order(manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Deterministic TRAIN order: sha256(sha + seed) (legacy rule)."""
    train = manifest[manifest["split"].astype(str) == "train"].copy()
    train["_sort"] = train["sha256"].astype(str).map(
        lambda s: hashlib.sha256((s + str(seed)).encode()).hexdigest()
    )
    return train.sort_values("_sort")


def build_background_pool(reader: ParentReader, manifest: pd.DataFrame, cfg, out_dir: Path,
                          rel, tick=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize up to bg_pool_max TRAIN images that contain an insulator anchor.

    Resumable: files already on disk (deterministic names) are reused.
    """
    max_n = int(cfg.get("parent_prepare.bg_pool_max"))
    anchor = int(cfg.get("parent_prepare.bg_anchor_class"))
    img_dir, lbl_dir = out_dir / "images", out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    rows, fails = [], []
    ordered = pool_order(manifest, int(cfg.get("seed")))
    for n_seen, (idx, row) in enumerate(ordered.iterrows(), 1):
        if len(rows) >= max_n:
            break
        try:
            keys = row_source_keys(row)
            suffix = Path(keys.rel).suffix.lower() or ".jpg"
            name = f"{str(row['sha256'])[:16]}_{idx:06d}{suffix}"
            ip, lp = img_dir / name, lbl_dir / Path(name).with_suffix(".txt")
            if ip.exists() and lp.exists():
                label_txt = lp.read_text(errors="ignore")
            else:
                label_txt = reader.read_label_text(row)
                if not any(cid == anchor for cid, *_ in parse_yolo_text(label_txt)):
                    continue
                atomic_write_bytes(ip, reader.read_image_bytes(row))
                atomic_write_text(lp, label_txt)
            rows.append({
                "manifest_index": int(idx),
                "sha256": str(row["sha256"]),
                "split": "train",
                "image": rel(ip),
                "label": rel(lp),
                "source_image_path": str(row["image_path_source"]),
            })
        except Exception as e:  # noqa: BLE001 - recorded, legacy behaviour
            fails.append({"manifest_index": int(idx), "sha256": str(row.get("sha256", "")), "error": repr(e)})
        if tick:
            tick(len(rows), max_n)
    return pd.DataFrame(rows), pd.DataFrame(fails)


# ---------------------------------------------------------------- pHash

PHASH_COLUMNS = ["manifest_index", "split", "image_path_source", "sha256", "phash_int"]


def compute_parent_phash(reader: ParentReader, manifest: pd.DataFrame, cfg, cache_path: Path,
                         partial_path: Path, tick=None) -> tuple[pd.DataFrame, list[dict]]:
    """pHash VAL/TEST with an incremental checkpoint (partial CSV, append + flush)."""
    from PIL import Image

    from ..hashing import phash_image

    splits = [str(s) for s in cfg.get("parent_prepare.phash_splits")]
    hash_size = int(cfg.get("parent_prepare.phash_hash_size"))
    subset = manifest[manifest["split"].astype(str).isin(splits)]

    done: set[int] = set()
    if partial_path.exists():
        try:
            prev = pd.read_csv(partial_path)
            done = set(int(x) for x in prev["manifest_index"])
            log.info("resume pHash checkpoint: %d rows already hashed", len(done))
        except Exception:  # noqa: BLE001 - corrupt partial => restart
            partial_path.unlink(missing_ok=True)

    todo = [(idx, row) for idx, row in subset.iterrows() if int(idx) not in done]
    # Read in archive order: much cheaper random access for gz members.
    todo.sort(key=lambda t: reader.member_offset(t[1]))
    fails: list[dict] = []
    new_file = not partial_path.exists()
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with open(partial_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PHASH_COLUMNS)
        if new_file:
            w.writeheader()
        for i, (idx, row) in enumerate(todo, 1):
            try:
                raw = reader.read_image_bytes(row)
                with Image.open(io.BytesIO(raw)) as im:
                    hv = phash_image(im, hash_size)
                w.writerow({
                    "manifest_index": int(idx), "split": str(row["split"]),
                    "image_path_source": str(row["image_path_source"]),
                    "sha256": str(row["sha256"]), "phash_int": hv,
                })
            except Exception as e:  # noqa: BLE001
                fails.append({"manifest_index": int(idx), "sha256": str(row["sha256"]), "error": repr(e)})
            if i % 200 == 0:
                fh.flush()
            if tick:
                tick(len(done) + i, len(subset))
    df = pd.read_csv(partial_path).drop_duplicates("manifest_index").sort_values("manifest_index")
    if len(df) == 0:
        raise RuntimeError("VAL/TEST pHash cache is empty; keep parent archives for debugging.")
    atomic_write_csv(cache_path, df)
    partial_path.unlink(missing_ok=True)
    return df, fails


def load_phash_cache(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(PHASH_COLUMNS) - set(df.columns)
    if missing:
        raise RuntimeError(f"pHash cache missing columns {sorted(missing)}: {path}")
    return df
