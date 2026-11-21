from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from PIL import Image
from ultralytics import YOLO


NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=8)
    return p.parse_args()


def quantiles(values):
    if not values:
        return {}
    x = sorted(values)
    def q(p):
        return x[round((len(x) - 1) * p)]
    return {"min": x[0], "p01": q(.01), "p05": q(.05), "p25": q(.25), "p50": q(.5), "p75": q(.75), "p95": q(.95), "p99": q(.99), "max": x[-1]}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def read_gt(label, width, height):
    boxes = []
    for line_no, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0])); xc, yc, bw, bh = map(float, parts[1:])
        boxes.append({
            "class_id": cls,
            "xyxy": ((xc-bw/2)*width, (yc-bh/2)*height, (xc+bw/2)*width, (yc+bh/2)*height),
            "w_px": bw*width, "h_px": bh*height, "area_norm": bw*bh,
            "aspect": bw/bh if bh else float("inf"), "line": line_no,
            "touch_border": xc-bw/2 <= 0.002 or yc-bh/2 <= 0.002 or xc+bw/2 >= 0.998 or yc+bh/2 >= 0.998,
        })
    return boxes


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows and not fields:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    a = args(); root, out = a.dataset.resolve(), a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    provenance = list(csv.DictReader((root/"merge_report/files.csv").open(encoding="utf-8-sig")))
    imported_rows = {r["destination_or_duplicate"]: r for r in provenance if r["action"] == "import"}
    image_dir, label_dir = root/"images/train", root/"labels/train"
    image_paths = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    current_names = {p.name for p in image_paths}
    imported_names = current_names & set(imported_rows)

    records, boxes_by_group, gt_by_image = [], defaultdict(list), {}
    for index, image in enumerate(image_paths, 1):
        group = "imported" if image.name in imported_names else "base"
        source = imported_rows[image.name]["source"] if group == "imported" else "curated_base"
        with Image.open(image) as im: width, height = im.size
        label = label_dir/f"{image.stem}.txt"
        gt = read_gt(label, width, height)
        gt_by_image[image.name] = gt
        records.append({"image": image.name, "group": group, "source": source, "width": width, "height": height, "boxes": len(gt), "empty": not gt})
        for box in gt:
            box.update({"image": image.name, "group": group, "source": source})
            boxes_by_group[group].append(box)

    def group_summary(group):
        rec = [r for r in records if r["group"] == group]; boxes = boxes_by_group[group]
        classes = Counter(b["class_id"] for b in boxes)
        return {
            "images": len(rec), "empty_images": sum(r["empty"] for r in rec), "boxes": len(boxes),
            "boxes_per_image": quantiles([r["boxes"] for r in rec]),
            "class_boxes": {NAMES[k]: classes[k] for k in range(6)},
            "width_px": quantiles([b["w_px"] for b in boxes]), "height_px": quantiles([b["h_px"] for b in boxes]),
            "area_norm": quantiles([b["area_norm"] for b in boxes]), "aspect_ratio": quantiles([b["aspect"] for b in boxes]),
            "tiny_lt4px": sum(b["w_px"] < 4 or b["h_px"] < 4 for b in boxes),
            "touch_border": sum(b["touch_border"] for b in boxes),
            "huge_area_gt75pct": sum(b["area_norm"] > .75 for b in boxes),
        }

    source_summary = {}
    for source in sorted({r["source"] for r in records if r["group"] == "imported"}):
        rec = [r for r in records if r["source"] == source]
        boxes = [b for b in boxes_by_group["imported"] if b["source"] == source]
        classes = Counter(b["class_id"] for b in boxes)
        prov = [imported_rows[r["image"]] for r in rec]
        source_summary[source] = {
            "images": len(rec), "empty_images": sum(r["empty"] for r in rec), "boxes": len(boxes),
            "class_boxes": {NAMES[k]: classes[k] for k in range(6)},
            "ambiguous_omitted": sum(int(r["ambiguous_omitted"] or 0) for r in prov),
            "malformed_omitted": sum(int(r["malformed_omitted"] or 0) for r in prov),
            "tiny_lt4px": sum(b["w_px"] < 4 or b["h_px"] < 4 for b in boxes),
            "touch_border": sum(b["touch_border"] for b in boxes),
            "box_area_p50": median([b["area_norm"] for b in boxes]) if boxes else None,
        }

    print(f"Running seed85 judge on {len(imported_names)} imported images...", flush=True)
    imported_paths = [image_dir/name for name in sorted(imported_names)]
    model = YOLO(str(a.weights.resolve()))
    # Ultralytics may stack an entire Python list before preprocessing even
    # when batch is set. Explicit chunks prevent a 3k-image allocation.
    def prediction_pairs():
        for start in range(0, len(imported_paths), a.batch):
            chunk = imported_paths[start:start + a.batch]
            results = model.predict(source=[str(p) for p in chunk], imgsz=640, conf=.25, iou=.7,
                                    batch=a.batch, device=0, workers=0, verbose=False, stream=False)
            yield from zip(chunk, results)

    missing, conflicts, judge_stats = [], [], Counter()
    for idx, (image, result) in enumerate(prediction_pairs(), 1):
        gt = gt_by_image[image.name]
        pred = []
        if result.boxes is not None:
            for xyxy, cls, conf in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                pred.append((xyxy, int(cls), float(conf)))
        covered = set()
        for xyxy, cls, conf in pred:
            overlaps = [iou(xyxy, g["xyxy"]) for g in gt]
            max_any = max(overlaps, default=0.0)
            same = [(v, j) for j, (v, g) in enumerate(zip(overlaps, gt)) if g["class_id"] == cls]
            max_same, same_idx = max(same, default=(0.0, -1))
            if max_same >= .5: covered.add(same_idx)
            if conf >= .60 and max_any < .30:
                missing.append({"image": image.name, "source": imported_rows[image.name]["source"], "pred_class": NAMES[cls],
                                "confidence": round(conf, 6), "max_iou_any_gt": round(max_any, 6),
                                "x1": round(xyxy[0], 2), "y1": round(xyxy[1], 2), "x2": round(xyxy[2], 2), "y2": round(xyxy[3], 2), "gt_boxes": len(gt)})
            if conf >= .60 and max_any >= .50:
                j = overlaps.index(max_any)
                if gt[j]["class_id"] != cls:
                    conflicts.append({"image": image.name, "source": imported_rows[image.name]["source"],
                                      "gt_class": NAMES[gt[j]["class_id"]], "pred_class": NAMES[cls],
                                      "confidence": round(conf, 6), "iou": round(max_any, 6), "gt_line": gt[j]["line"]})
        judge_stats["gt_boxes"] += len(gt); judge_stats["gt_covered_same_class_iou50_conf25"] += len(covered)
        judge_stats["predictions_conf25"] += len(pred)
        if idx % 250 == 0: print(f"  judged {idx}/{len(imported_paths)}", flush=True)

    summary = {
        "dataset": str(root), "judge": str(a.weights.resolve()),
        "groups": {g: group_summary(g) for g in ("base", "imported")},
        "sources": source_summary,
        "merge_report": {
            "imports_expected": len(imported_rows), "imports_present": len(imported_names),
            "imports_removed_before_audit": len(set(imported_rows)-current_names),
        },
        "judge_audit": dict(judge_stats) | {
            "potential_missing_predictions_conf60_iou_lt30": len(missing),
            "high_conf_class_conflicts_iou50": len(conflicts),
        },
    }
    (out/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out/"potential_missing_boxes.csv", missing,
              ["image","source","pred_class","confidence","max_iou_any_gt","x1","y1","x2","y2","gt_boxes"])
    write_csv(out/"class_conflicts.csv", conflicts,
              ["image","source","gt_class","pred_class","confidence","iou","gt_line"])
    write_csv(out/"image_inventory.csv", records)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
