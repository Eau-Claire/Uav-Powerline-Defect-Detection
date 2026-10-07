"""Materialize the YOLO11n data-ablation dataset:

    V6.2E TRAIN + V6.2E VAL (frozen, untouched)
    + reviewed direct TRAIN rows (MPCD, PowerEquipment)
    + approved synthetic foreign objects (TRAIN only)

TEST is never materialized. V6.2E images are streamed straight from the
archives into the dataset (no intermediate full extraction). Re-running skips
files that already exist (deterministic names).
"""
from __future__ import annotations

import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from ..dataset.v6p2e import ParentReader, row_source_keys
from ..hashing import sha256_bytes, stable_json_hash
from ..io_utils import atomic_write_bytes, atomic_write_csv, atomic_write_text
from ..qa import integrity

log = logging.getLogger("yolov11sdi.dataset_builder")

ORIGIN_PARENT, ORIGIN_DIRECT, ORIGIN_SYN = "v6p2e", "v6p5_direct", "v6p5_synthetic"
DIRECT_PREFIX = {"mpcd_broken_strand": "mpcd", "power_equipment_stage": "pe"}


@dataclass
class BuiltDataset:
    root: Path
    data_yaml: Path
    manifest: pd.DataFrame
    tag: str
    counts: dict


def _write_pair(root: Path, split: str, name: str, img: bytes, lbl: bytes) -> None:
    ip = root / "images" / split / name
    lp = root / "labels" / split / Path(name).with_suffix(".txt")
    if not (ip.exists() and lp.exists()):
        atomic_write_bytes(ip, img)
        atomic_write_bytes(lp, lbl)


def add_parent_rows(reader: ParentReader, manifest: pd.DataFrame, splits, root: Path, tick=None) -> list[dict]:
    rows = integrity.training_rows(manifest, splits)
    out = []
    for n, (i, row) in enumerate(rows.iterrows(), 1):
        split = str(row["split"])
        suffix = Path(row_source_keys(row).rel).suffix.lower()
        name = f"{str(row['sha256'])[:16]}_{int(i):06d}{suffix}"
        ip = root / "images" / split / name
        lp = root / "labels" / split / Path(name).with_suffix(".txt")
        if not (ip.exists() and lp.exists()):
            atomic_write_bytes(ip, reader.read_image_bytes(row))
            atomic_write_text(lp, reader.read_label_text(row))
        out.append({"dataset_split": split, "file": name, "origin": ORIGIN_PARENT, "source": "V6.2E",
                    "sha256": str(row["sha256"]), "manifest_index": int(i), "review_id": ""})
        if tick and n % 500 == 0:
            tick(n, len(rows))
    return out


def add_direct_archive(archive: Path, arc_root: str, root: Path, keep_names: set[str] | None) -> list[dict]:
    """Only images/train/* of a direct archive; optional filter by reviewed KEEP basenames."""
    prefix = DIRECT_PREFIX[arc_root]
    out = []
    with tarfile.open(archive, "r:*") as t:
        members = {m.name: m for m in t.getmembers() if m.isfile()}
        for n in sorted(members):
            if f"{arc_root}/images/train/" not in n:
                continue
            base = Path(n).name
            if keep_names is not None and base not in keep_names:
                continue
            lname = Path(n.replace("/images/train/", "/labels/train/")).with_suffix(".txt").as_posix()
            lm = members.get(lname)
            if lm is None:
                raise RuntimeError(f"Missing overlay label: {lname}")
            raw = t.extractfile(members[n]).read()
            name = f"v65_{prefix}_{base}"
            _write_pair(root, "train", name, raw, t.extractfile(lm).read())
            out.append({"dataset_split": "train", "file": name, "origin": ORIGIN_DIRECT, "source": arc_root,
                        "sha256": sha256_bytes(raw), "manifest_index": -1, "review_id": ""})
    return out


def add_synthetic_archive(archive: Path, root: Path) -> list[dict]:
    out = []
    with tarfile.open(archive, "r:*") as t:
        members = {m.name: m for m in t.getmembers() if m.isfile()}
        for n in sorted(members):
            if "synthetic_foreign/images/train/" not in n:
                continue
            lname = Path(n.replace("/images/train/", "/labels/train/")).with_suffix(".txt").as_posix()
            lm = members.get(lname)
            if lm is None:
                raise RuntimeError(f"Missing synthetic label: {lname}")
            raw = t.extractfile(members[n]).read()
            name = f"v65_syn_{Path(n).name}"
            _write_pair(root, "train", name, raw, t.extractfile(lm).read())
            out.append({"dataset_split": "train", "file": name, "origin": ORIGIN_SYN, "source": "synthetic",
                        "sha256": sha256_bytes(raw), "manifest_index": -1, "review_id": ""})
    return out


def finalize(root: Path, records: list[dict], parent_manifest: pd.DataFrame, names: list[str],
             compose_tag: str) -> BuiltDataset:
    df = pd.DataFrame(records).sort_values(["dataset_split", "file"]).reset_index(drop=True)
    # Integrity gates — fail before any GPU time is spent.
    integrity.check_no_test_in_training(df, parent_manifest)
    integrity.check_val_frozen(df, parent_manifest)
    integrity.check_synthetic_train_only(df)
    problems = integrity.check_yolo_dir(root, len(names))
    if problems:
        raise integrity.IntegrityError(f"{len(problems)} dataset problem(s), e.g. {problems[:10]}")
    stray = {p.name for p in (root / "images").rglob("*") if p.is_file()} - set(df["file"])
    if stray:
        raise integrity.IntegrityError(f"{len(stray)} unexpected file(s) in dataset dir, e.g. {sorted(stray)[:5]}")

    digest = stable_json_hash({"compose_tag": compose_tag,
                               "records": df[["dataset_split", "file", "origin", "sha256"]].to_dict("records")})
    tag = f"v6p5_ablation_ds_{digest[:12]}"
    data_yaml = root / "data.yaml"
    atomic_write_text(data_yaml, yaml.safe_dump({
        "path": str(root), "train": "images/train", "val": "images/val",
        "names": {i: n for i, n in enumerate(names)},
    }, sort_keys=False))
    atomic_write_csv(root / "dataset_manifest.csv", df)
    counts = df.groupby(["dataset_split", "origin"]).size().rename("n").reset_index()
    counts = {f"{r.dataset_split}/{r.origin}": int(r.n) for r in counts.itertuples()}
    counts["train_total"] = int((df["dataset_split"] == "train").sum())
    counts["val_total"] = int((df["dataset_split"] == "val").sum())
    return BuiltDataset(root=root, data_yaml=data_yaml, manifest=df, tag=tag, counts=counts)
