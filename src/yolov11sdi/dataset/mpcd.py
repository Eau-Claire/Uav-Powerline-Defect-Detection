"""MPCD / PowerLine-MTYOLO -> broken_strand (class 5).

Benelmostafa & Medromi, Drones 2025 (doi:10.3390/drones9070505).
Broken / fracture / discontinuity / strand-break classes map to broken_strand.
Source TRAIN -> overlay train; source val/test -> external_eval (never V6.2E VAL/TEST).
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import yaml

from ..hashing import sha256_file
from ..io_utils import IMG_EXTS, read_yolo_label

log = logging.getLogger("yolov11sdi.mpcd")
SOURCE = "MPCD"


def download_and_extract(cfg, ext_root: Path) -> tuple[Path, str | None]:
    """Returns (extracted dir, zip sha256 or None when reusing an extraction)."""
    mdir, mzip = ext_root / "MPCD", ext_root / "MPCD.zip"
    if mdir.exists() and any(mdir.rglob("data.yaml")):
        log.info("MPCD: reuse extracted %s", mdir)
        return mdir, None
    if not mzip.exists():
        import gdown

        url = f"https://drive.google.com/uc?id={cfg.get('mpcd.gdrive_id')}"
        log.info("MPCD: downloading %s", url)
        part = mzip.with_suffix(".zip.part")
        if gdown.download(url, str(part), quiet=False) is None or not part.exists():
            raise RuntimeError(f"MPCD Google Drive download failed. Place the archive manually at {mzip}")
        part.rename(mzip)
    zsha = sha256_file(mzip)
    tmp = ext_root / ".MPCD.extracting"
    with zipfile.ZipFile(mzip) as z:
        z.extractall(tmp)
    tmp.rename(mdir)
    mzip.unlink()  # legacy low-disk policy: keep only the extraction
    return mdir, zsha


def _norm_names(names):
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names or [])


def broken_class_ids(src_names: list[str], keywords: list[str]) -> list[int]:
    out = []
    for cid, name in enumerate(src_names):
        n = str(name).lower().replace("_", " ").replace("-", " ")
        if any(k in n for k in keywords):
            out.append(cid)
    return out


def _image_to_label(ip: Path) -> Path | None:
    s = ip.as_posix()
    return Path(s.replace("/images/", "/labels/")).with_suffix(".txt") if "/images/" in s else None


def parse_mpcd(cfg, mdir: Path) -> list[dict]:
    keywords = list(cfg.get("mpcd.name_keywords"))
    target = int(cfg.get("mpcd.target_class"))
    records: dict[tuple[str, str], dict] = {}
    for yp in sorted(mdir.rglob("data.yaml")):
        try:
            dcfg = yaml.safe_load(yp.read_text())
            base = Path(dcfg.get("path", yp.parent))
            if not base.is_absolute():
                base = (yp.parent / base).resolve()
            src_names = _norm_names(dcfg.get("names", []))
        except Exception:  # noqa: BLE001
            continue
        ids = broken_class_ids(src_names, keywords)
        if not ids:
            continue
        log.info("MPCD yaml %s names=%s mapped broken ids=%s", yp.relative_to(mdir), src_names, ids)
        for split_key in ("train", "val", "test"):
            raw = dcfg.get(split_key)
            if raw is None:
                continue
            p = Path(raw) if Path(raw).is_absolute() else base / raw
            if p.is_file():
                imgs = [Path(x.strip()) if Path(x.strip()).is_absolute() else base / x.strip()
                        for x in p.read_text().splitlines() if x.strip()]
            elif p.exists():
                imgs = sorted(x for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS)
            else:
                continue
            for ip in imgs:
                lp = _image_to_label(ip)
                if lp is None or not lp.exists():
                    continue
                mapped = [(target, xc, yc, w, h) for cid, xc, yc, w, h in read_yolo_label(lp) if cid in ids]
                if not mapped:
                    continue
                records[(split_key, ip.as_posix())] = {
                    "source": SOURCE,
                    "source_split": split_key,
                    "target_split": "train" if split_key == "train" else "external_eval",
                    "source_image": ip.relative_to(mdir).as_posix() if ip.is_relative_to(mdir) else ip.name,
                    "source_image_abs": str(ip),
                    "mapped_rows": mapped,
                    "license": cfg.get("mpcd.license_note"),
                }
    out = [records[k] for k in sorted(records)]
    if not out:
        raise RuntimeError("MPCD parsed but produced 0 mapped broken-strand images.")
    return out
