#!/usr/bin/env python
"""Build dataset v6p5_v6 from v6p5_v5 (7 classes, ids unchanged) with the foreign_object review of 2026-10-15
(data/label_audit/fo_review_v6/, see its README section in data/label_audit/README.md).

Rule J  fo_review  train only, decisions.csv via fo_review_v6/apply.py:
          FO boxes that are workers -> person, bird nests -> bird_nest, drop non-FO boxes (plants growing on the ground,
          empty / equipment-only, drawings, unclear), add missed FO / person / bird_nest boxes, drop all 161 v65
          synthetic FO images (pasted birds labelled FO, tiny blurred pastes)
Rule K  old_veg    the v2 audit dropped FO boxes as "vegetation"; FO now includes vegetation on a pole / line / equipment
          (user decision): train images whose dropped box was such vegetation are removed (no box size to restore);
          val_clean_v6 = val_clean + those boxes restored from the untouched `val` labels.
`val` and `val_clean` are copied unchanged (comparable with v5); `val_clean_v6` is the new main metric set.
Images are hard-linked (copied if linking fails); v5 is never modified.
Usage: python scripts/build_v6.py --v5 datasets/v6p5_v5 --out datasets/v6p5_v6
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

REVIEW = Path(__file__).resolve().parents[1] / "data/label_audit/fo_review_v6"
sys.path.insert(0, str(REVIEW))
from apply import apply, load  # noqa: E402

NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand", "person"]


def link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v5", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    assert yaml.safe_load((a.v5 / "data.yaml").read_text())["names"] in (NAMES, dict(enumerate(NAMES)))
    dec = load()
    old = pd.read_csv(REVIEW / "old_veg_decisions.csv")
    drop_k = set(old[old.decision == "drop_image_train"].image)
    restore = defaultdict(list)
    for r in old[old.decision == "restore_fo_val_clean_v6"].itertuples():
        restore[r.image].append([float(v) for v in r.box.split()])

    changes, stats = [], defaultdict(Counter)

    def log(split, image, rule, detail=""):
        changes.append({"split": split, "image": image, "rule": rule, "detail": detail})
        stats[split][rule] += 1

    a.out.mkdir(parents=True, exist_ok=False)
    for split, src in (("train", "train"), ("val", "val"), ("val_clean", "val_clean"), ("val_clean_v6", "val_clean")):
        (a.out / "images" / split).mkdir(parents=True)
        (a.out / "labels" / split).mkdir(parents=True)
        for img in sorted((a.v5 / "images" / src).glob("*.jpg")):
            lines = (a.v5 / "labels" / src / (img.stem + ".txt")).read_text().splitlines()
            if split == "train":
                if img.name in drop_k:
                    log(split, img.name, "K_old_veg", "drop image (vegetation-on-structure FO box lost in v2 audit)")
                    continue
                lines, ch = apply(lines, dec.get(img.name))
                if lines is None:
                    log(split, img.name, "J_fo_review", "drop synthetic image")
                    continue
                if ch:
                    log(split, img.name, "J_fo_review", "; ".join(ch))
            if split == "val_clean_v6" and img.name in restore:
                for b in restore[img.name]:
                    lines.append(f"4 {(b[0]+b[2])/2:.6f} {(b[1]+b[3])/2:.6f} {b[2]-b[0]:.6f} {b[3]-b[1]:.6f}")
                log(split, img.name, "K_old_veg", f"restore {len(restore[img.name])} FO boxes")
            lines = [l for l in lines if l.strip()]
            (a.out / "labels" / split / (img.stem + ".txt")).write_bytes(("\n".join(lines) + ("\n" if lines else "")).encode())
            link(img, a.out / "images" / split / img.name)
            stats[split]["images"] += 1
            for l in lines:
                stats[split][f"box:{NAMES[int(l.split()[0])]}"] += 1
    assert stats["val_clean_v6"]["K_old_veg"] == len(restore), "a restored val_clean_v6 image is missing"

    names = dict(enumerate(NAMES))
    for f, v in (("data.yaml", "val"), ("data_val_clean.yaml", "val_clean"), ("data_val_clean_v6.yaml", "val_clean_v6")):
        (a.out / f).write_text(yaml.safe_dump({"path": str(a.out.resolve()), "train": "images/train", "val": f"images/{v}",
                                               "names": names}, sort_keys=False))
    with (a.out / "changes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "image", "rule", "detail"])
        w.writeheader()
        w.writerows(changes)
    (a.out / "annotations").mkdir()  # COCO view (RT-DETR / DETR-family); category ids = YOLO ids, bbox in pixels
    for split in ("train", "val", "val_clean", "val_clean_v6"):
        images, anns = [], []
        for i, img in enumerate(sorted((a.out / "images" / split).glob("*.jpg"))):
            W, H = Image.open(img).size
            images.append({"id": i, "file_name": f"images/{split}/{img.name}", "width": W, "height": H})
            for line in (a.out / "labels" / split / (img.stem + ".txt")).read_text().splitlines():
                c, x, y, w, h = line.split()
                x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
                anns.append({"id": len(anns), "image_id": i, "category_id": int(c), "iscrowd": 0, "area": w * h,
                             "bbox": [round(x - w / 2, 2), round(y - h / 2, 2), round(w, 2), round(h, 2)]})
        (a.out / "annotations" / f"instances_{split}.json").write_text(json.dumps(
            {"images": images, "annotations": anns, "categories": [{"id": k, "name": v} for k, v in names.items()]}))
    label_hash = hashlib.sha256()
    for f in sorted((a.out / "labels").rglob("*.txt")):
        label_hash.update(f.relative_to(a.out).as_posix().encode() + f.read_bytes().replace(b"\r\n", b"\n"))  # same on any OS
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "v6p5_v5", "source_labels_sha256": json.loads((a.v5 / "manifest.json").read_text())["labels_sha256"],
        "review": {f: sha(REVIEW / f) for f in ("decisions.csv", "items.csv", "old_veg_decisions.csv")},
        "labels_sha256": label_hash.hexdigest(),
        "val_policy": "val, val_clean = v5 unchanged (comparable); val_clean_v6 = val_clean + K_old_veg (main metric)",
        "fo_definition": "real foreign object incl. hornet nests and vegetation on a pole/line/equipment; NOT a person, "
                         "bird nest, plants growing on the ground, electrical equipment, empty region, drawing",
        "stats": {k: dict(sorted(v.items())) for k, v in stats.items()},
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["stats"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
