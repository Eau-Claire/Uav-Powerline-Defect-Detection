"""Power Equipment Image Dataset (Xiong et al., IET GTD 2021).

    dx_dg -> broken_strand (5)
    yw    -> foreign_object (4)
    nw    -> bird_nest (3)
    dx_sg -> REVIEW ONLY (strand loosening), never merged

Split: deterministic_fraction(key) < train_frac -> train, else external_eval.
The key reproduces the legacy Kaggle absolute path (see config
power_equipment.split_key_prefix) so the split is machine independent.
"""
from __future__ import annotations

import logging
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from ..hashing import deterministic_fraction, sha256_file
from ..io_utils import IMG_EXTS, xyxy_to_yolo

log = logging.getLogger("yolov11sdi.power_equipment")
SOURCE = "PowerEquipment"


def map_object(name: str, power_map: dict, review_only: dict) -> tuple[int | None, str | None]:
    """(mapped class id or None, review role or None). dx_sg is never mapped."""
    name = (name or "").strip()
    if name in review_only:
        return None, review_only[name]
    if name in power_map:
        return int(power_map[name]), None
    return None, None


def split_for(rel_image: str, cfg) -> str:
    prefix = str(cfg.get("power_equipment.split_key_prefix")).rstrip("/")
    key = f"{prefix}/{rel_image}"
    return "train" if deterministic_fraction(key) < float(cfg.get("power_equipment.train_frac")) else "external_eval"


def download_and_extract(cfg, ext_root: Path, hf_cache: Path) -> tuple[Path, str | None]:
    pe_dir = ext_root / "power_equipment_object_detection"
    if pe_dir.exists() and any(pe_dir.rglob("*.xml")):
        log.info("PowerEquipment: reuse extracted %s", pe_dir)
        return pe_dir, None
    from huggingface_hub import hf_hub_download

    zpath = Path(hf_hub_download(
        repo_id=cfg.get("power_equipment.hf_repo"), filename=cfg.get("power_equipment.filename"),
        repo_type="dataset", cache_dir=str(hf_cache),
    ))
    zsha = sha256_file(zpath)
    tmp = ext_root / ".power_equipment.extracting"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(tmp)
    tmp.rename(pe_dir)
    shutil.rmtree(hf_cache, ignore_errors=True)
    return pe_dir, zsha


def parse_power_equipment(cfg, pe_dir: Path) -> tuple[list[dict], list[dict]]:
    power_map = {k: int(v) for k, v in cfg.get("power_equipment.map").items()}
    review_only = dict(cfg.get("power_equipment.review_only"))
    by_stem: dict[str, list[Path]] = {}
    for ip in sorted(pe_dir.rglob("*")):
        if ip.suffix.lower() in IMG_EXTS:
            by_stem.setdefault(ip.stem, []).append(ip)

    records, review = [], []
    xmls = sorted(pe_dir.rglob("*.xml"))
    log.info("PowerEquipment XML files: %d", len(xmls))
    for xp in xmls:
        try:
            root = ET.parse(xp).getroot()
        except ET.ParseError:
            continue
        filename = root.findtext("filename")
        cand = None
        if filename and (xp.parent / filename).exists():
            cand = xp.parent / filename
        else:
            hits = by_stem.get(Path(filename).stem if filename else xp.stem, [])
            cand = hits[0] if hits else None
        if cand is None or not cand.exists():
            continue
        size = root.find("size")
        if size is None:
            from PIL import Image

            with Image.open(cand) as im:
                W0, H0 = im.size
        else:
            W0, H0 = int(float(size.findtext("width"))), int(float(size.findtext("height")))
        rel_image = cand.relative_to(pe_dir).as_posix()

        mapped = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            box = [
                max(0.0, min(1.0, float(bb.findtext("xmin")) / W0)),
                max(0.0, min(1.0, float(bb.findtext("ymin")) / H0)),
                max(0.0, min(1.0, float(bb.findtext("xmax")) / W0)),
                max(0.0, min(1.0, float(bb.findtext("ymax")) / H0)),
            ]
            cid, role = map_object(obj.findtext("name"), power_map, review_only)
            if cid is not None:
                mapped.append((cid, *xyxy_to_yolo(box)))
            if role is not None:
                review.append({
                    "source": SOURCE, "source_image": rel_image, "source_xml": xp.relative_to(pe_dir).as_posix(),
                    "source_class": (obj.findtext("name") or "").strip(), "review_role": role, "bbox_xyxy": box,
                })
        if mapped:
            split = split_for(rel_image, cfg)
            records.append({
                "source": SOURCE, "source_split": split, "target_split": split,
                "source_image": rel_image, "source_image_abs": str(cand),
                "mapped_rows": mapped, "license": cfg.get("power_equipment.license_note"),
            })
    if not records:
        raise RuntimeError("PowerEquipment parsed but produced 0 mapped candidate images.")
    return records, review
