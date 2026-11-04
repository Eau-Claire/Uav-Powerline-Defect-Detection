#!/usr/bin/env python
"""Build an operational UAV test subset by excluding frames with sub-threshold GT objects."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/test_main_v1"
DEFAULT_OUT = ROOT / "data/test_operational_min8_v1"
NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-side", type=float, default=8.0)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    image_out, label_out = output / "images/test", output / "labels/test"
    image_out.mkdir(parents=True); label_out.mkdir(parents=True)
    manifest, distribution, excluded = [], Counter(), []
    images = sorted(p for p in (source / "images/test").iterdir() if p.suffix.lower() in EXTS)
    for image in images:
        label = source / "labels/test" / f"{image.stem}.txt"
        with Image.open(image) as opened:
            width, height = opened.size
        scale = args.imgsz / max(width, height)
        boxes = []
        for line in label.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                cls = int(float(parts[0]))
                side = min(float(parts[3]) * width * scale, float(parts[4]) * height * scale)
                boxes.append((cls, side))
        tiny = [(cls, side) for cls, side in boxes if side < args.min_side]
        keep = not tiny
        row = {
            "image": image.name, "kept": int(keep), "width": width, "height": height,
            "boxes": len(boxes), "minimum_side_at_imgsz": round(min((side for _, side in boxes), default=9999), 4),
            "tiny_boxes": len(tiny), "tiny_classes": "|".join(NAMES[cls] for cls, _ in tiny),
        }
        manifest.append(row)
        if not keep:
            excluded.append(row)
            continue
        target_image = image_out / image.name
        try:
            os.link(image, target_image)
        except OSError:
            shutil.copy2(image, target_image)
        shutil.copy2(label, label_out / label.name)
        if not boxes:
            distribution["background_images"] += 1
        for cls, _ in boxes:
            distribution[f"{NAMES[cls]}_boxes"] += 1
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)
    with (output / "excluded.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(excluded)
    dataset = {"path": output.as_posix(), "train": "images/test", "val": "images/test", "test": "images/test",
               "names": {index: name for index, name in enumerate(NAMES)}}
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    freeze = {
        "name": output.name, "role": "secondary operational UAV subset; never replaces test_main_v1",
        "source": str(source), "rule": "exclude the entire image if any GT box has min(width,height) below threshold after letterbox scaling",
        "imgsz": args.imgsz, "minimum_side_pixels": args.min_side,
        "source_images": len(images), "kept_images": len(images) - len(excluded), "excluded_images": len(excluded),
        "manifest_sha256": sha256(output / "manifest.csv"), "distribution": dict(distribution),
    }
    (output / "freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
