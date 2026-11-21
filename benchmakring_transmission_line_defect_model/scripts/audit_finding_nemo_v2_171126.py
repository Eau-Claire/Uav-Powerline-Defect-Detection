from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(image: Image.Image) -> int:
    pixels = list(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR).getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def inspect_image(item):
    split, path = item
    try:
        with Image.open(path) as im:
            width, height = im.size
            fingerprint = dhash(im)
        return dict(split=split, image=path.name, stem=path.stem, sha256=sha256(path), dhash=f"{fingerprint:016x}",
                    width=width, height=height, bytes=path.stat().st_size, error="")
    except Exception as exc:
        return dict(split=split, image=path.name, stem=path.stem, sha256="", dhash="", width=0, height=0,
                    bytes=path.stat().st_size, error=f"{type(exc).__name__}: {exc}")


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / max((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter, 1e-12)


def inspect_label(item, dims):
    split, path = item
    image_name = next((path.stem + suffix for suffix in IMAGE_SUFFIXES if (path.parents[2] / "images" / split / (path.stem + suffix)).is_file()), "")
    width, height = dims.get((split, path.stem), (0, 0))
    rows, errors = [], []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not raw.strip(): continue
        parts = raw.split()
        try:
            if len(parts) != 5: raise ValueError("expected 5 fields")
            cls = int(parts[0]); x, y, w, h = map(float, parts[1:])
            if cls not in range(6): raise ValueError("class outside 0..5")
            if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0: raise ValueError("non-finite/non-positive")
            if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6: raise ValueError("box outside image")
            rows.append((cls, x, y, w, h))
        except Exception as exc:
            errors.append(dict(split=split, label=path.name, line=line_no, text=raw, error=str(exc)))
    duplicates = 0
    for i, a in enumerate(rows):
        aa = (a[1]-a[3]/2, a[2]-a[4]/2, a[1]+a[3]/2, a[2]+a[4]/2)
        for b in rows[i+1:]:
            if a[0] != b[0]: continue
            bb = (b[1]-b[3]/2, b[2]-b[4]/2, b[1]+b[3]/2, b[2]+b[4]/2)
            duplicates += iou(aa, bb) >= 0.95
    counts, tiny, huge = Counter(r[0] for r in rows), Counter(), Counter()
    for cls, _, _, w, h in rows:
        tiny[cls] += bool(width and height and (w*width < 4 or h*height < 4))
        huge[cls] += w*h > 0.75
    return dict(split=split, label=path.name, image=image_name, boxes=len(rows), empty=not rows,
                duplicate_boxes=duplicates, class_counts=dict(counts), tiny_counts=dict(tiny), huge_counts=dict(huge)), errors


def write_csv(path, rows):
    rows = list(rows)
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    a = parse_args(); root, base, out = a.dataset.resolve(), a.base.resolve(), a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    provenance = list(csv.DictReader((root / "merge_report/files.csv").open(encoding="utf-8-sig")))
    imported_names = {r["destination_or_duplicate"] for r in provenance if r["action"] == "import"}
    image_items, label_items = [], []
    for split in ("train", "val", "test"):
        image_items += [(split, p) for p in sorted((root/"images"/split).iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
        label_items += [(split, p) for p in sorted((root/"labels"/split).glob("*.txt"))]
    print(f"Inspecting {len(image_items)} images...", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool: images = list(pool.map(inspect_image, image_items))
    dims = {(r["split"], r["stem"]): (r["width"], r["height"]) for r in images}
    print(f"Inspecting {len(label_items)} labels...", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool: label_results = list(pool.map(lambda item: inspect_label(item, dims), label_items))
    labels = [x[0] for x in label_results]; label_errors = [e for x in label_results for e in x[1]]

    by_sha, by_dhash = defaultdict(list), defaultdict(list)
    for r in images:
        if r["sha256"]: by_sha[r["sha256"]].append(r)
        if r["dhash"]: by_dhash[r["dhash"]].append(r)
    exact_groups = [v for v in by_sha.values() if len(v) > 1]
    cross_exact = [g for g in exact_groups if len({r["split"] for r in g}) > 1]
    dhash_groups = [v for v in by_dhash.values() if len(v) > 1]
    cross_dhash = [g for g in dhash_groups if len({r["split"] for r in g}) > 1]

    summary = {"dataset": str(root), "recommended_name": "finding_nemo_curated_v2_plus_171126_train_only", "splits": {},
               "invalid_label_rows": len(label_errors), "unreadable_images": sum(bool(r["error"]) for r in images),
               "exact_duplicate_groups": len(exact_groups), "cross_split_exact_duplicate_groups": len(cross_exact),
               "same_dhash_groups": len(dhash_groups), "cross_split_same_dhash_groups": len(cross_dhash),
               "imported_expected": len(imported_names), "imported_present": sum(r["image"] in imported_names for r in images if r["split"] == "train")}
    for split in ("train", "val", "test"):
        ir = [r for r in images if r["split"] == split]; lr = [r for r in labels if r["split"] == split]
        counts, tiny, huge = Counter(), Counter(), Counter()
        for r in lr:
            counts.update({int(k): v for k, v in r["class_counts"].items()}); tiny.update({int(k): v for k, v in r["tiny_counts"].items()}); huge.update({int(k): v for k, v in r["huge_counts"].items()})
        summary["splits"][split] = {"images": len(ir), "labels": len(lr), "empty_labels": sum(r["empty"] for r in lr),
            "duplicate_boxes_iou95": sum(r["duplicate_boxes"] for r in lr), "boxes": {NAMES[k]: counts[k] for k in range(6)},
            "tiny_boxes_lt4px": {NAMES[k]: tiny[k] for k in range(6)}, "huge_boxes_area_gt75pct": {NAMES[k]: huge[k] for k in range(6)}}
    base_checks = {}
    for split in ("train", "val", "test"):
        old_images = {p.name for p in (base/"images"/split).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
        new_images = {r["image"] for r in images if r["split"] == split}
        old_labels = {p.name: sha256(p) for p in (base/"labels"/split).glob("*.txt")}
        new_label_dir = root/"labels"/split
        changed = sum(not (new_label_dir/name).is_file() or sha256(new_label_dir/name) != digest for name, digest in old_labels.items())
        base_checks[split] = {"base_images_missing": len(old_images-new_images), "new_images_vs_base": len(new_images-old_images), "base_labels_changed_or_missing": changed}
    summary["base_preservation"] = base_checks
    (out/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out/"unreadable_images.csv", [r for r in images if r["error"]]); write_csv(out/"invalid_labels.csv", label_errors)
    write_csv(out/"suspicious_labels.csv", [r for r in labels if r["empty"] or r["duplicate_boxes"] or any(r["huge_counts"].values())])
    write_csv(out/"exact_duplicates.csv", [{"sha256": sha, "splits": "|".join(r["split"] for r in group), "images": "|".join(r["image"] for r in group)} for sha, group in by_sha.items() if len(group)>1])
    write_csv(out/"same_dhash.csv", [{"dhash": h, "splits": "|".join(r["split"] for r in group), "images": "|".join(r["image"] for r in group)} for h, group in by_dhash.items() if len(group)>1])
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
