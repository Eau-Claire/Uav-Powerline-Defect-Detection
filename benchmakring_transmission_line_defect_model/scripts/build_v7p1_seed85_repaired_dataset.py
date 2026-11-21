from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def xywh_to_xyxy(box):
    _, x, y, w, h = box
    return x-w/2, y-h/2, x+w/2, y+h/2


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    bb = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / max(aa + bb - inter, 1e-12)


def main():
    a = parse_args(); src, audit, out = a.source.resolve(), a.audit.resolve(), a.output.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    proposals = defaultdict(list)
    with (audit/"potential_missing_boxes.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            proposals[r["image"]].append(r)

    out.mkdir(parents=True)
    counts = Counter()
    change_rows = []
    for split in ("train", "val", "test"):
        src_images, src_labels = src/"images"/split, src/"labels"/split
        dst_images, dst_labels = out/"images"/split, out/"labels"/split
        dst_images.mkdir(parents=True); dst_labels.mkdir(parents=True)
        images = sorted(p for p in src_images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        for image in images:
            link_or_copy(image, dst_images/image.name)
            old_label = src_labels/f"{image.stem}.txt"
            new_label = dst_labels/old_label.name
            if split != "train":
                shutil.copy2(old_label, new_label)
                continue

            boxes = []
            for line in old_label.read_text(encoding="utf-8-sig").splitlines():
                parts = line.split()
                if len(parts) == 5:
                    boxes.append(tuple([int(float(parts[0]))] + [float(x) for x in parts[1:]]))
            with Image.open(image) as im:
                width, height = im.size
            changes_before = len(change_rows)

            # Merge bug repair: when class 0 and class 1 have identical extents,
            # retain broken and remove the contradictory normal label.
            broken_xyxy = [xywh_to_xyxy(b) for b in boxes if b[0] == 1]
            repaired = []
            for box in boxes:
                if box[0] == 0 and any(iou(xywh_to_xyxy(box), bb) >= .999999 for bb in broken_xyxy):
                    counts["contradictory_normal_removed"] += 1
                    change_rows.append({"image": image.name, "action": "remove", "class": "insulator",
                                        "reason": "IoU=1 with insulator_broken merge-conflict", "confidence": ""})
                    continue
                repaired.append(box)
            boxes = repaired

            # Rule F on the complete train split.
            repaired = []
            for box in boxes:
                if box[3]*width < 4 or box[4]*height < 4:
                    counts["tiny_boxes_removed"] += 1
                    counts[f"tiny_removed_{NAMES[box[0]]}"] += 1
                    change_rows.append({"image": image.name, "action": "remove", "class": NAMES[box[0]],
                                        "reason": "Rule F: width or height <4 px", "confidence": ""})
                    continue
                repaired.append(box)
            boxes = repaired

            # Add high-confidence Rule-B candidates produced by seed85.
            for p in sorted(proposals.get(image.name, []), key=lambda z: float(z["confidence"]), reverse=True):
                cls = NAMES.index(p["pred_class"])
                x1 = max(0.0, min(width, float(p["x1"]))) / width
                y1 = max(0.0, min(height, float(p["y1"]))) / height
                x2 = max(0.0, min(width, float(p["x2"]))) / width
                y2 = max(0.0, min(height, float(p["y2"]))) / height
                candidate = (cls, (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1)
                if candidate[3]*width < 4 or candidate[4]*height < 4:
                    counts["judge_candidate_tiny_skipped"] += 1
                    continue
                if any(b[0] == cls and iou(xywh_to_xyxy(candidate), xywh_to_xyxy(b)) >= .85 for b in boxes):
                    counts["judge_candidate_deduped"] += 1
                    continue
                boxes.append(candidate)
                counts["judge_boxes_added"] += 1
                counts[f"judge_added_{NAMES[cls]}"] += 1
                change_rows.append({"image": image.name, "action": "add", "class": NAMES[cls],
                                    "reason": "seed85 conf>=0.60 and max IoU<0.30", "confidence": p["confidence"]})

            if len(change_rows) == changes_before:
                shutil.copy2(old_label, new_label)
            else:
                lines = [f"{b[0]} {b[1]:.8f} {b[2]:.8f} {b[3]:.8f} {b[4]:.8f}" for b in boxes]
                new_label.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

    # Preserve provenance reports and create local dataset YAMLs.
    if (src/"merge_report").is_dir():
        shutil.copytree(src/"merge_report", out/"merge_report")
    for name in ("DATASET_STATUS.json",):
        if (src/name).is_file(): shutil.copy2(src/name, out/name)
    yaml_text = (
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" +
        "".join(f"  {i}: {name}\n" for i, name in enumerate(NAMES))
    )
    (out/"data.yaml").write_text(yaml_text, encoding="utf-8")
    (out/"data_local.yaml").write_text(yaml_text, encoding="utf-8")
    report = out/"repair_report"; report.mkdir()
    with (report/"changes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image","action","class","reason","confidence"])
        w.writeheader(); w.writerows(change_rows)
    summary = {
        "source_dataset": str(src), "judge": "weights/v7p1_seed85_curated_yolo26m_best.pt",
        "policy": "conservative additive relabel; conf>=0.60; max IoU<0.30; Rule F; repair exact class-0/class-1 conflicts",
        "counts": dict(sorted(counts.items())), "changes": len(change_rows),
    }
    (report/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Created: {out}")


if __name__ == "__main__":
    main()
