"""Open Images V7 appearance source for synthetic foreign objects.

Disk-safe implementation (preserved from the DISKSAFE notebook):
* download `detections` only — NEVER the segmentation mask ZIPs;
* build a local foreground alpha with GrabCut seeded by the detection bbox;
* one class at a time; source images deleted right after cropping;
* checkpoint per class (crops + per-class manifest), so a restart resumes at
  the next unfinished class.

Portability fix: crop ids are derived from the Open Images image id rather
than the absolute download path, so local and Kaggle runs agree.
"""
from __future__ import annotations

import gc
import hashlib
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ..hashing import sha256_file
from ..io_utils import atomic_path, atomic_write_csv, read_csv_safe

log = logging.getLogger("yolov11sdi.openimages")

CROP_COLUMNS = [
    "source", "source_split", "oi_image_id", "source_sha256", "foreign_subtype",
    "detection_index", "crop_path", "mask_available", "mask_source", "foreground_ratio",
]


def subtype_of(oi_class: str) -> str:
    return oi_class.lower().replace(" ", "_")


def grabcut_rgba_from_detection(im, bbox, gc_cfg: dict):
    """RGBA crop whose alpha is a GrabCut foreground mask seeded by the bbox.

    Returns (rgba, fg_ratio) or None when the mask is implausible.
    `bbox` is Open Images relative [x, y, w, h].
    """
    import cv2
    from PIL import Image

    pad_frac = float(gc_cfg["pad_frac"])
    min_box = int(gc_cfg["min_box_px"])
    W0, H0 = im.size
    x, y, w, h = map(float, bbox)
    bx1, by1 = max(0, int(round(x * W0))), max(0, int(round(y * H0)))
    bx2, by2 = min(W0, int(round((x + w) * W0))), min(H0, int(round((y + h) * H0)))
    bw, bh = bx2 - bx1, by2 - by1
    if bw < min_box or bh < min_box:
        return None

    px, py = max(4, int(round(bw * pad_frac))), max(4, int(round(bh * pad_frac)))
    cx1, cy1 = max(0, bx1 - px), max(0, by1 - py)
    cx2, cy2 = min(W0, bx2 + px), min(H0, by2 + py)
    crop = im.crop((cx1, cy1, cx2, cy2)).convert("RGB")
    arr = np.asarray(crop)
    ch, cw = arr.shape[:2]
    if cw < 10 or ch < 10:
        return None

    rx1, ry1 = max(1, bx1 - cx1), max(1, by1 - cy1)
    rx2, ry2 = min(cw - 1, bx2 - cx1), min(ch - 1, by2 - cy1)
    rw, rh = max(2, rx2 - rx1), max(2, ry2 - ry1)
    if rx1 + rw >= cw:
        rw = cw - rx1 - 1
    if ry1 + rh >= ch:
        rh = ch - ry1 - 1
    if rw < 2 or rh < 2:
        return None

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    mask = np.zeros((ch, cw), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, (int(rx1), int(ry1), int(rw), int(rh)), bgd, fgd,
                    int(gc_cfg["iterations"]), cv2.GC_INIT_WITH_RECT)
    except Exception:  # noqa: BLE001
        return None

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # Restrict alpha to a slightly expanded detection box.
    allowed = np.zeros_like(fg)
    ef = float(gc_cfg["allowed_expand_frac"])
    ex, ey = max(2, int(ef * rw)), max(2, int(ef * rh))
    allowed[max(0, ry1 - ey):min(ch, ry1 + rh + ey), max(0, rx1 - ex):min(cw, rx1 + rw + ex)] = 255
    fg = cv2.bitwise_and(fg, allowed)
    k = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=1)

    ratio = float((fg > 0).mean())
    if ratio < float(gc_cfg["min_fg_ratio"]) or ratio > float(gc_cfg["max_fg_ratio"]):
        return None
    rgba = crop.convert("RGBA")
    rgba.putalpha(Image.fromarray(fg, mode="L"))
    return rgba, ratio


def class_manifest_path(crops_root: Path, subtype: str) -> Path:
    return crops_root / f"manifest_{subtype}.csv"


def download_class(cfg, oi_class: str, class_root: Path) -> None:
    import fiftyone.utils.openimages as fouo

    oi = cfg.get("openimages")
    # CRITICAL: detections only. 'segmentations' pulls huge mask ZIP shards.
    fouo.download_open_images_split(
        dataset_dir=str(class_root), split=oi["split"], version=oi["version"],
        label_types=["detections"], classes=[oi_class], shuffle=True,
        seed=int(oi["seed"]), max_samples=int(oi["max_samples_per_class"]),
        num_workers=int(oi["num_workers"]),
    )


def crop_class(cfg, oi_class: str, class_root: Path, crops_root: Path, rel) -> tuple[pd.DataFrame, int]:
    import fiftyone.utils.openimages as fouo
    from PIL import Image

    oi = cfg.get("openimages")
    subtype = subtype_of(oi_class)
    out_dir = crops_root / subtype
    out_dir.mkdir(parents=True, exist_ok=True)
    imp = fouo.OpenImagesV7DatasetImporter(
        dataset_dir=str(class_root), label_types=["detections"], classes=[oi_class],
        only_matching=True, shuffle=False,
    )
    imp.setup()
    rows, rejected = [], 0
    try:
        for image_path, _meta, labels in imp:
            image_path = Path(image_path)
            if not image_path.exists():
                continue
            if not isinstance(labels, dict):
                labels = {"detections": labels}
            dets = labels.get("detections")
            if dets is None or not hasattr(dets, "detections"):
                continue
            cands = [d for d in dets.detections if d.label == oi_class]
            if not cands:
                continue
            try:
                im = Image.open(image_path).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            src_sha = sha256_file(image_path)
            image_id = image_path.stem
            for j, det in enumerate(cands):
                res = grabcut_rgba_from_detection(im, det.bounding_box, oi["grabcut"])
                if res is None:
                    rejected += 1
                    continue
                rgba, ratio = res
                uid = hashlib.sha256(f"{image_id}|{oi_class}|{j}".encode()).hexdigest()[:16]
                op = out_dir / f"{uid}.png"
                with atomic_path(op) as tmp:
                    rgba.save(tmp, format="PNG", optimize=True)
                rows.append({
                    "source": "open_images_v7", "source_split": oi["split"], "oi_image_id": image_id,
                    "source_sha256": src_sha, "foreign_subtype": subtype, "detection_index": j,
                    "crop_path": rel(op), "mask_available": True,
                    "mask_source": "grabcut_from_openimages_detection", "foreground_ratio": ratio,
                })
    finally:
        imp.close()
    df = pd.DataFrame(rows, columns=CROP_COLUMNS).sort_values(["oi_image_id", "detection_index"])
    return df, rejected


def build_openimages_crops(cfg, work_root: Path, crops_root: Path, rel, on_class_done=None) -> pd.DataFrame:
    """Per-class checkpointed crop generation. Returns the combined manifest."""
    classes = list(cfg.get("openimages.classes"))
    tmp_root = work_root / "open_images_v7_by_class"
    for oi_class in classes:
        subtype = subtype_of(oi_class)
        mpath = class_manifest_path(crops_root, subtype)
        if mpath.exists() and len(read_csv_safe(mpath)) > 0:
            log.info("OpenImages %s: checkpoint found, skip", oi_class)
            if on_class_done:
                on_class_done(oi_class)
            continue
        class_root = tmp_root / subtype
        shutil.rmtree(class_root, ignore_errors=True)
        shutil.rmtree(crops_root / subtype, ignore_errors=True)
        class_root.mkdir(parents=True, exist_ok=True)
        log.info("OpenImages %s: detections-only download + GrabCut", oi_class)
        download_class(cfg, oi_class, class_root)
        df, rejected = crop_class(cfg, oi_class, class_root, crops_root, rel)
        log.info("OpenImages %s: usable masked crops=%d rejected masks=%d", oi_class, len(df), rejected)
        shutil.rmtree(class_root, ignore_errors=True)   # free source images now
        gc.collect()
        if len(df) == 0:
            raise RuntimeError(f"No usable masked crops for {oi_class}")
        atomic_write_csv(mpath, df)
        if on_class_done:
            on_class_done(oi_class)
    shutil.rmtree(tmp_root, ignore_errors=True)
    return load_crop_manifest(cfg, crops_root)


def load_crop_manifest(cfg, crops_root: Path) -> pd.DataFrame:
    frames = []
    for oi_class in cfg.get("openimages.classes"):
        p = class_manifest_path(crops_root, subtype_of(oi_class))
        frames.append(read_csv_safe(p))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CROP_COLUMNS)
    if cfg.get("synthetic.require_mask"):
        expected = [subtype_of(c) for c in cfg.get("openimages.classes")]
        have = set(df.loc[df["mask_available"].astype(str).str.lower() == "true", "foreign_subtype"]) if len(df) else set()
        missing = [s for s in expected if s not in have]
        if missing:
            raise RuntimeError("SYNTH_REQUIRE_MASK=True but no masked crops for: " + ", ".join(missing))
    return df
