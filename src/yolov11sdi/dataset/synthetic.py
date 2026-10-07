"""Relation-aware synthetic foreign objects (TRAIN ONLY).

Masked Open Images crops are composited NEAR an insulator anchor on INTERNAL
V6.2E TRAIN backgrounds and labelled foreign_object (class 4). Original
background GT is preserved. Using internal TRAIN backgrounds is intentional and
is not external leakage.

Checkpointed per subtype: each finished subtype writes its manifest CSV; a
restart skips finished subtypes. Each subtype uses its own RNG seeded from
(seed, subtype) so resuming does not perturb the other subtypes.
"""
from __future__ import annotations

import hashlib
import logging
import random
import shutil
from pathlib import Path

import pandas as pd

from ..hashing import seeded_int, sha256_file
from ..io_utils import atomic_path, atomic_write_csv, read_csv_safe, read_yolo_label, write_yolo_label, xyxy_to_yolo

log = logging.getLogger("yolov11sdi.synthetic")

SYN_COLUMNS = [
    "overlay_image", "overlay_label", "source", "foreign_subtype", "oi_image_id", "oi_source_sha256",
    "oi_crop", "background_image", "background_sha256", "anchor_class", "foreign_box_xyxy",
    "synthetic", "train_only",
]


def find_anchors(bg_pool: pd.DataFrame, anchor_class: int, resolve) -> pd.DataFrame:
    rows = []
    for _, r in bg_pool.iterrows():
        for j, (cid, xc, yc, w, h) in enumerate(read_yolo_label(resolve(r["label"]))):
            if cid == anchor_class:
                rows.append({"image": r["image"], "label": r["label"], "background_sha256": r["sha256"],
                             "anchor_index": j, "anchor_xc": xc, "anchor_yc": yc, "anchor_w": w, "anchor_h": h})
    return pd.DataFrame(rows)


def resize_rgba(obj, longside: int):
    from PIL import Image

    ow, oh = obj.size
    sc = longside / max(ow, oh)
    return obj.resize((max(2, int(ow * sc)), max(2, int(oh * sc))), Image.Resampling.LANCZOS)


def compose_one(bg, obj, anchor, subtype: str, rng: random.Random, scfg: dict):
    """Legacy relation-aware placement near the anchor. Returns (image, xyxy) or None."""
    W0, H0 = bg.size
    ax, ay, aw, ah = anchor
    acx, acy = ax * W0, ay * H0
    along = max(aw * W0, ah * H0)
    lo, hi = scfg["rel_scale"].get(subtype, scfg["rel_scale"]["default"])
    rel = rng.uniform(lo, hi)
    obj = resize_rgba(obj, max(int(scfg["min_long_side_px"]),
                               min(int(along * rel), int(min(W0, H0) * float(scfg["max_frac_of_min_side"])))))
    if subtype == "person":
        dx = rng.uniform(-.8, .8) * max(aw * W0, 40)
        dy = rng.uniform(.35, 1.2) * max(ah * H0, 40)
    else:
        dx = rng.uniform(-.7, .7) * max(aw * W0, 35)
        dy = rng.uniform(-.6, .6) * max(ah * H0, 35)
    x1 = max(0, min(W0 - obj.width, int(acx + dx - obj.width / 2)))
    y1 = max(0, min(H0 - obj.height, int(acy + dy - obj.height / 2)))
    x2, y2 = x1 + obj.width, y1 + obj.height
    if x2 <= x1 or y2 <= y1:
        return None
    out = bg.convert("RGBA")
    out.alpha_composite(obj, (x1, y1))
    return out.convert("RGB"), [x1 / W0, y1 / H0, x2 / W0, y2 / H0]


def subtype_manifest_path(manifest_dir: Path, subtype: str) -> Path:
    # Kept OUTSIDE the overlay root so it never enters the overlay digest.
    return manifest_dir / f"synthetic_{subtype}.csv"


def generate_subtype(subtype: str, pool: list[dict], anchors: pd.DataFrame, cfg, syn_root: Path,
                     resolve, rel) -> pd.DataFrame:
    from PIL import Image

    scfg = cfg.get("synthetic")
    seed = int(cfg.get("seed"))
    target_n = int(scfg["per_subtype"])
    foreign = int(scfg["foreign_class"])
    rng = random.Random(seeded_int(seed, "synthetic", subtype))
    img_dir, lbl_dir = syn_root / "images/train", syn_root / "labels/train"
    for p in list(img_dir.glob(f"syn_{subtype}_*")) + list(lbl_dir.glob(f"syn_{subtype}_*")):
        p.unlink()   # partial subtype from an interrupted run
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    made, attempts, rows = 0, 0, []
    max_attempts = target_n * int(scfg["max_attempts_factor"])
    while made < target_n and attempts < max_attempts:
        attempts += 1
        objrec = rng.choice(pool)
        a = anchors.iloc[rng.randrange(len(anchors))]
        try:
            bg = Image.open(resolve(a["image"])).convert("RGB")
            obj = Image.open(resolve(objrec["crop_path"])).convert("RGBA")
        except Exception:  # noqa: BLE001
            continue
        res = compose_one(bg, obj, (float(a["anchor_xc"]), float(a["anchor_yc"]),
                                    float(a["anchor_w"]), float(a["anchor_h"])), subtype, rng, scfg)
        if res is None:
            continue
        sim, box = res
        labels = read_yolo_label(resolve(a["label"]))
        labels.append((foreign, *xyxy_to_yolo(box)))
        uid = hashlib.sha256(
            f"{a['background_sha256']}|{a['anchor_index']}|{objrec['oi_image_id']}|"
            f"{objrec['detection_index']}|{subtype}|{made}|{seed}".encode()
        ).hexdigest()[:18]
        di = img_dir / f"syn_{subtype}_{uid}.jpg"
        dl = lbl_dir / f"syn_{subtype}_{uid}.txt"
        with atomic_path(di) as tmp:
            sim.save(tmp, format="JPEG", quality=int(scfg["jpeg_quality"]))
        write_yolo_label(dl, labels)
        rows.append({
            "overlay_image": rel(di), "overlay_label": rel(dl),
            "source": "OpenImagesV7_synthetic_relation", "foreign_subtype": subtype,
            "oi_image_id": objrec["oi_image_id"], "oi_source_sha256": objrec["source_sha256"],
            "oi_crop": objrec["crop_path"], "background_image": a["image"],
            "background_sha256": sha256_file(resolve(a["image"])), "anchor_class": cfg.names[int(scfg["anchor_class"])],
            "foreign_box_xyxy": [round(v, 8) for v in box], "synthetic": True, "train_only": True,
        })
        made += 1
    log.info("synthetic %s: made=%d attempts=%d", subtype, made, attempts)
    return pd.DataFrame(rows, columns=SYN_COLUMNS)


def build_synthetic(cfg, crops: pd.DataFrame, anchors: pd.DataFrame, syn_root: Path, manifest_dir: Path,
                    resolve, rel, on_subtype_done=None) -> pd.DataFrame:
    if len(anchors) == 0:
        raise RuntimeError("No insulator anchors found in TRAIN background pool.")
    usable = crops
    if cfg.get("synthetic.require_mask"):
        usable = crops[crops["mask_available"].astype(str).str.lower() == "true"]
    if len(usable) == 0:
        raise RuntimeError("No usable Open Images object crops for synthetic generation.")
    usable = usable.sort_values(["foreign_subtype", "oi_image_id", "detection_index"])
    frames = []
    for subtype in sorted(usable["foreign_subtype"].unique()):
        mpath = subtype_manifest_path(manifest_dir, subtype)
        if mpath.exists() and len(read_csv_safe(mpath)) > 0:
            log.info("synthetic %s: checkpoint found, skip", subtype)
        else:
            pool = usable[usable["foreign_subtype"] == subtype].to_dict("records")
            df = generate_subtype(subtype, pool, anchors, cfg, syn_root, resolve, rel)
            atomic_write_csv(mpath, df)
        frames.append(read_csv_safe(mpath))
        if on_subtype_done:
            on_subtype_done(subtype)
    return pd.concat(frames, ignore_index=True)


def reset_synthetic(syn_root: Path, manifest_dir: Path) -> None:
    shutil.rmtree(syn_root, ignore_errors=True)
    shutil.rmtree(manifest_dir, ignore_errors=True)
