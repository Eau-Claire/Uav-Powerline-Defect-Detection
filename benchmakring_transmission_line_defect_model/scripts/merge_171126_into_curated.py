#!/usr/bin/env python3
"""Build a non-destructive curated v2 dataset from data/171126.

Roboflow YOLO26 exports may contain either 5-column boxes or polygon/SAM
annotations. Polygons are converted to tight axis-aligned boxes. Only
unambiguous source classes are imported; ambiguous vegetation/fire labels are
reported for manual review instead of silently changing the taxonomy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CURATED_NAMES = [
    "insulator", "insulator_broken", "insulator_flashover",
    "bird_nest", "foreign_object", "broken_strand",
]

# None means deliberately excluded because the semantic mapping is ambiguous.
SOURCES = {
    "power line inspection.v5-train-val-test.yolo26": {
        "prefix": "pli5",
        "names": ["balloon", "bird nest", "branch", "fire", "kite", "trees"],
        "mapping": {0: 4, 1: 3, 2: 4, 3: None, 4: 4, 5: 4},
    },
    "power line.v1i.yolo26": {
        "prefix": "pl1",
        "names": ["Broken Cable", "ddan_tran_tt", "vdat_tt", "vday_tt"],
        # "Broken Cable" annotations generally cover a long cable section, not
        # the local frayed/broken-strand defect used by the curated taxonomy.
        "mapping": {0: None, 1: None, 2: None, 3: None},
    },
    "Powerline Project 3.v2i.yolo26": {
        "prefix": "pp3",
        "names": ["Broken Cable", "Broken Insulator", "Cable", "Insulators", "Tower", "Vegetation"],
        "mapping": {0: None, 1: 1, 2: None, 3: 0, 4: None, 5: 4},
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def parse_annotation(line: str):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        vals = [float(x) for x in parts[1:]]
    except ValueError:
        return None
    if len(vals) == 4:  # standard YOLO box
        xc, yc, w, h = vals
        x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
        kind = "box"
    elif len(vals) >= 6 and len(vals) % 2 == 0:  # polygon/SAM
        xs, ys = vals[0::2], vals[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        kind = "polygon"
    else:
        return None
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return cls, [x1, y1, x2, y2], kind


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-12)


def dedupe_boxes(boxes, threshold=0.85):
    # Larger boxes first; identical-class boxes describing the same object collapse.
    ranked = sorted(boxes, key=lambda z: (z[1][2] - z[1][0]) * (z[1][3] - z[1][1]), reverse=True)
    kept, removed = [], 0
    for item in ranked:
        if any(item[0] == old[0] and iou(item[1], old[1]) >= threshold for old in kept):
            removed += 1
        else:
            kept.append(item)
    return kept, removed


def normalize_broken_insulators(boxes):
    """Expand damage-only broken boxes to the containing full insulator box.

    The imported PP3 labels often include a tiny Broken Insulator polygon and
    a full Insulators polygon. Curated convention labels the whole defective
    insulator, so the former inherits the latter's extent. A tiny broken label
    without a matching full-insulator annotation is omitted for review.
    """
    insulators = [b for b in boxes if b[0] == 0]
    output, expanded, omitted = [], 0, 0
    for cls, box, kind in boxes:
        if cls != 1:
            output.append((cls, box, kind))
            continue
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        candidates = []
        for candidate in insulators:
            cb = candidate[1]
            contains = cb[0] <= cx <= cb[2] and cb[1] <= cy <= cb[3]
            if contains or iou(box, cb) > 0:
                candidates.append(candidate)
        if candidates:
            chosen = min(candidates, key=lambda z: (z[1][2]-z[1][0]) * (z[1][3]-z[1][1]))
            output.append((1, chosen[1], "expanded_from_damage"))
            expanded += 1
        else:
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area >= 0.005:  # already plausibly covers the complete assembly
                output.append((cls, box, kind))
            else:
                omitted += 1
    return output, expanded, omitted


def image_for_label(label: Path) -> Path | None:
    image_dir = label.parent.parent / "images"
    for ext in IMAGE_EXTS:
        p = image_dir / f"{label.stem}{ext}"
        if p.exists():
            return p
    return None


def draw_preview(image_path: Path, boxes, output: Path) -> None:
    im = cv2.imread(str(image_path))
    if im is None:
        return
    h, w = im.shape[:2]
    colors = [(255, 180, 0), (0, 80, 255), (255, 0, 180), (0, 220, 0), (0, 160, 255), (255, 80, 0)]
    for cls, (x1, y1, x2, y2), _ in boxes:
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(im, p1, p2, colors[cls], 2)
        cv2.putText(im, CURATED_NAMES[cls], (p1[0], max(18, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, .55, colors[cls], 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), im)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", type=Path, required=True)
    ap.add_argument("--incoming", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--preview-limit", type=int, default=80)
    args = ap.parse_args()
    curated, incoming, out = args.curated.resolve(), args.incoming.resolve(), args.output.resolve()
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {out}")

    report_dir = out / "merge_report"
    report_dir.mkdir(parents=True)
    counts = Counter()
    rows = []

    # Clone curated splits. Images use hard links when possible; labels are copied
    # because annotations must never share writable storage with the original.
    known_hashes = {}
    protected_hashes = set()
    for split in ("train", "val", "test"):
        src_i, src_l = curated / "images" / split, curated / "labels" / split
        dst_i, dst_l = out / "images" / split, out / "labels" / split
        dst_i.mkdir(parents=True, exist_ok=True); dst_l.mkdir(parents=True, exist_ok=True)
        for image in src_i.iterdir():
            if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
                digest = sha256(image)
                known_hashes[digest] = f"curated/{split}/{image.name}"
                if split in {"val", "test"}:
                    protected_hashes.add(digest)
                link_or_copy(image, dst_i / image.name)
        for label in src_l.glob("*.txt"):
            shutil.copy2(label, dst_l / label.name)
        counts[f"curated_{split}_images"] = sum(1 for p in src_i.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    preview_count = Counter()
    for source_name, spec in SOURCES.items():
        root = incoming / source_name
        if not root.exists():
            raise FileNotFoundError(root)
        for split in ("train", "valid", "test"):
            label_dir = root / split / "labels"
            if not label_dir.exists():
                continue
            for label in sorted(label_dir.glob("*.txt")):
                image = image_for_label(label)
                if image is None:
                    counts["missing_image"] += 1
                    continue
                digest = sha256(image)
                if digest in known_hashes:
                    reason = "duplicate_protected" if digest in protected_hashes else "duplicate_existing"
                    counts[reason] += 1
                    rows.append([source_name, split, image.name, "skip", reason, known_hashes[digest], 0, 0, 0])
                    continue

                boxes, ambiguous, malformed, polygon_count = [], 0, 0, 0
                for line in label.read_text(encoding="utf-8", errors="replace").splitlines():
                    parsed = parse_annotation(line)
                    if parsed is None:
                        malformed += 1
                        continue
                    old_cls, xyxy, kind = parsed
                    new_cls = spec["mapping"].get(old_cls)
                    if new_cls is None:
                        ambiguous += 1
                        continue
                    if kind == "polygon":
                        polygon_count += 1
                    boxes.append((new_cls, xyxy, kind))
                boxes, expanded_broken, tiny_broken_omitted = normalize_broken_insulators(boxes)
                boxes, duplicate_boxes = dedupe_boxes(boxes)
                if not boxes:
                    counts["no_usable_boxes"] += 1
                    rows.append([source_name, split, image.name, "skip", "no_usable_boxes", "", 0, ambiguous, malformed])
                    continue

                # Prefix plus digest makes collisions impossible and preserves provenance.
                stem = f"{spec['prefix']}_{split}_{label.stem}_{digest[:10]}"
                dst_image = out / "images" / "train" / f"{stem}{image.suffix.lower()}"
                dst_label = out / "labels" / "train" / f"{stem}.txt"
                link_or_copy(image, dst_image)
                lines = []
                for cls, (x1, y1, x2, y2), _ in boxes:
                    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                    lines.append(f"{cls} {xc:.8f} {yc:.8f} {x2-x1:.8f} {y2-y1:.8f}")
                    counts[f"box_{CURATED_NAMES[cls]}"] += 1
                dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
                known_hashes[digest] = f"incoming/{source_name}/{split}/{image.name}"
                counts["imported_images"] += 1
                counts["polygon_boxes_converted"] += polygon_count
                counts["duplicate_boxes_removed"] += duplicate_boxes
                counts["broken_insulator_boxes_expanded"] += expanded_broken
                counts["tiny_unmatched_broken_insulator_omitted"] += tiny_broken_omitted
                counts["ambiguous_boxes_omitted"] += ambiguous
                counts["malformed_boxes_omitted"] += malformed
                rows.append([source_name, split, image.name, "import", "", dst_image.name, len(boxes), ambiguous, malformed])
                for cls in sorted({b[0] for b in boxes}):
                    if preview_count[cls] < args.preview_limit:
                        draw_preview(image, boxes, report_dir / "previews" / CURATED_NAMES[cls] / f"{stem}.jpg")
                        preview_count[cls] += 1

    data_yaml = (
        f"path: {out.as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(CURATED_NAMES))
    )
    (out / "data.yaml").write_text(data_yaml, encoding="utf-8")
    with (report_dir / "files.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["source", "source_split", "image", "action", "reason", "destination_or_duplicate", "kept_boxes", "ambiguous_omitted", "malformed_omitted"]); w.writerows(rows)
    summary = dict(sorted(counts.items()))
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "DATASET_STATUS.json").write_text(json.dumps({
        "base_dataset": str(curated), "incoming": str(incoming),
        "policy": "all accepted incoming images added to train only; curated val/test unchanged",
        "summary": summary,
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Created: {out}")


if __name__ == "__main__":
    main()
