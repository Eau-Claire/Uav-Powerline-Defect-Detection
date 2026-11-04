#!/usr/bin/env python3
"""Diagnose Finding Nemo test errors without changing the frozen test set.

The script caches YOLO predictions once and scores the same predictions against
the original ``test_main_v1`` labels and the human-reviewed labels.  Resolution
selection is deliberately a separate ``sweep-val`` command that only reads VAL.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BENCHMARK = Path(__file__).resolve().parents[1]
REPO = BENCHMARK.parent
REVIEWED = BENCHMARK / "data/finding_nemo_human_reviewed_6cls"
RAW = BENCHMARK / "data/test_main_v1"
MODEL = BENCHMARK / "results/finding_nemo/yolo26m/weights/best.pt"
OUT = BENCHMARK / "results/finding_nemo/error_analysis_yolo26m"
NAMES = [
    "insulator", "insulator_broken", "insulator_flashover",
    "bird_nest", "foreign_object", "broken_strand",
]
COLORS = [(40, 180, 255), (255, 70, 70), (255, 170, 30),
          (120, 220, 80), (190, 90, 255), (255, 220, 40)]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_index(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS}


def load_yolo_labels(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    if not path.exists():
        return boxes
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 5:
            raise ValueError(f"Invalid label at {path}:{line_no}")
        cls, cx, cy, bw, bh = map(float, fields[:5])
        x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
        x2, y2 = (cx + bw / 2) * width, (cy + bh / 2) * height
        boxes.append({"class_id": int(cls), "box": [x1, y1, x2, y2]})
    return boxes


def iou_matrix(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    aa = np.asarray([x["box"] for x in a], dtype=np.float32)
    bb = np.asarray([x["box"] for x in b], dtype=np.float32)
    lt = np.maximum(aa[:, None, :2], bb[None, :, :2])
    rb = np.minimum(aa[:, None, 2:], bb[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(aa[:, 2] - aa[:, 0], 0, None) * np.clip(aa[:, 3] - aa[:, 1], 0, None)
    area_b = np.clip(bb[:, 2] - bb[:, 0], 0, None) * np.clip(bb[:, 3] - bb[:, 1], 0, None)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def greedy_pairs(overlaps: np.ndarray, allowed: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    candidates = np.argwhere((overlaps >= threshold) & allowed)
    ranked = sorted(((int(i), int(j), float(overlaps[i, j])) for i, j in candidates),
                    key=lambda item: item[2], reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs = []
    for i, j, score in ranked:
        if i not in used_a and j not in used_b:
            pairs.append((i, j, score))
            used_a.add(i)
            used_b.add(j)
    return pairs


def size_bin(box: list[float], width: int, height: int) -> str:
    # COCO area thresholds, normalized to the original image's pixel area.
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    scaled = area * (640.0 * 640.0) / max(width * height, 1)
    return "small" if scaled < 32**2 else "medium" if scaled < 96**2 else "large"


def predict(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    images = image_index(REVIEWED / "images/test")
    if args.limit:
        images = dict(list(sorted(images.items()))[:args.limit])
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / f"predictions_imgsz{args.imgsz}.jsonl"
    metadata = args.out / f"predictions_imgsz{args.imgsz}.meta.json"
    model = YOLO(str(args.model))
    started = time.perf_counter()
    count = 0
    with cache.open("w", encoding="utf-8") as output:
        for result in model.predict(
            source=[str(images[k]) for k in sorted(images)], imgsz=args.imgsz,
            conf=args.conf, iou=args.nms_iou, device=args.device, stream=True,
            verbose=False, augment=args.augment,
        ):
            height, width = result.orig_shape
            detections = []
            if result.boxes is not None:
                for box, cls, conf in zip(result.boxes.xyxy.cpu().tolist(),
                                          result.boxes.cls.cpu().tolist(),
                                          result.boxes.conf.cpu().tolist()):
                    detections.append({"class_id": int(cls), "confidence": float(conf),
                                       "box": [float(v) for v in box]})
            record = {"stem": Path(result.path).stem, "width": width, "height": height,
                      "detections": detections}
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
    metadata.write_text(json.dumps({
        "checkpoint": str(args.model.resolve()), "checkpoint_sha256": sha256(args.model),
        "reviewed_dataset": str(REVIEWED.resolve()), "images": count,
        "imgsz": args.imgsz, "confidence": args.conf, "nms_iou": args.nms_iou,
        "augment": args.augment, "device": args.device,
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2), encoding="utf-8")
    print(f"Saved {count} predictions to {cache}")


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return {record["stem"]: record for line in handle if (record := json.loads(line))}


def classify_image(stem: str, predictions: list[dict[str, Any]], gt: list[dict[str, Any]],
                   width: int, height: int, gt_version: str) -> tuple[list[dict[str, Any]], Counter]:
    overlaps = iou_matrix(gt, predictions)
    same = np.zeros_like(overlaps, dtype=bool)
    for i, truth in enumerate(gt):
        for j, pred in enumerate(predictions):
            same[i, j] = truth["class_id"] == pred["class_id"]
    tp = greedy_pairs(overlaps, same, 0.5)
    used_gt = {i for i, _, _ in tp}
    used_pred = {j for _, j, _ in tp}
    rows: list[dict[str, Any]] = []
    counts: Counter = Counter({"tp": len(tp)})

    remaining_gt = [i for i in range(len(gt)) if i not in used_gt]
    remaining_pred = [j for j in range(len(predictions)) if j not in used_pred]
    for i in list(remaining_gt):
        candidates = [(j, float(overlaps[i, j])) for j in remaining_pred
                      if gt[i]["class_id"] != predictions[j]["class_id"] and overlaps[i, j] >= 0.5]
        if candidates:
            j, score = max(candidates, key=lambda x: x[1])
            rows.append(error_row(stem, gt_version, "class_confusion", gt[i], predictions[j],
                                  score, width, height))
            counts["class_confusion"] += 1
            used_gt.add(i); used_pred.add(j); remaining_pred.remove(j)

    remaining_gt = [i for i in range(len(gt)) if i not in used_gt]
    remaining_pred = [j for j in range(len(predictions)) if j not in used_pred]
    for i in list(remaining_gt):
        candidates = [(j, float(overlaps[i, j])) for j in remaining_pred
                      if gt[i]["class_id"] == predictions[j]["class_id"] and overlaps[i, j] >= 0.1]
        if candidates:
            j, score = max(candidates, key=lambda x: x[1])
            rows.append(error_row(stem, gt_version, "localization", gt[i], predictions[j],
                                  score, width, height))
            counts["localization"] += 1
            used_gt.add(i); used_pred.add(j); remaining_pred.remove(j)

    for i, truth in enumerate(gt):
        if i not in used_gt:
            rows.append(error_row(stem, gt_version, "fn", truth, None, 0.0, width, height))
            counts["fn"] += 1
    for j, pred in enumerate(predictions):
        if j not in used_pred:
            rows.append(error_row(stem, gt_version, "fp", None, pred, 0.0, width, height))
            counts["fp"] += 1
    return rows, counts


def error_row(stem: str, version: str, kind: str, gt: dict[str, Any] | None,
              pred: dict[str, Any] | None, overlap: float, width: int, height: int) -> dict[str, Any]:
    gt_cls = gt["class_id"] if gt else None
    pred_cls = pred["class_id"] if pred else None
    ref_box = gt["box"] if gt else pred["box"]
    return {
        "stem": stem, "gt_version": version, "error_type": kind,
        "gt_class_id": gt_cls, "gt_class": NAMES[gt_cls] if gt_cls is not None else "",
        "pred_class_id": pred_cls, "pred_class": NAMES[pred_cls] if pred_cls is not None else "",
        "confidence": pred.get("confidence", "") if pred else "", "iou": overlap,
        "size": size_bin(ref_box, width, height),
        "gt_box": json.dumps(gt["box"]) if gt else "",
        "pred_box": json.dumps(pred["box"]) if pred else "",
    }


def compare_gt(stem: str, raw: list[dict[str, Any]], reviewed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overlaps = iou_matrix(raw, reviewed)
    same = np.array([[a["class_id"] == b["class_id"] for b in reviewed] for a in raw], dtype=bool)
    if same.shape != overlaps.shape:
        same = np.zeros_like(overlaps, dtype=bool)
    exact = greedy_pairs(overlaps, same, 0.999)
    used_raw = {i for i, _, _ in exact}; used_review = {j for _, j, _ in exact}
    changed = greedy_pairs(overlaps, same, 0.1)
    rows = []
    for i, j, score in changed:
        if i in used_raw or j in used_review:
            continue
        rows.append({"stem": stem, "change": "box_changed", "class": NAMES[raw[i]["class_id"]],
                     "iou": score, "raw_box": json.dumps(raw[i]["box"]),
                     "reviewed_box": json.dumps(reviewed[j]["box"])})
        used_raw.add(i); used_review.add(j)
    for i, box in enumerate(raw):
        if i not in used_raw:
            rows.append({"stem": stem, "change": "removed_or_reclassified", "class": NAMES[box["class_id"]],
                         "iou": "", "raw_box": json.dumps(box["box"]), "reviewed_box": ""})
    for j, box in enumerate(reviewed):
        if j not in used_review:
            rows.append({"stem": stem, "change": "added_or_reclassified", "class": NAMES[box["class_id"]],
                         "iou": "", "raw_box": "", "reviewed_box": json.dumps(box["box"])})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows and not fields:
        return
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def draw_gallery(out: Path, rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]],
                 reviewed_index: dict[str, Path], limit: int) -> None:
    gallery = out / "gallery"
    for kind in ("fn", "fp", "localization", "class_confusion"):
        candidates = [r for r in rows if r["gt_version"] == "reviewed" and r["error_type"] == kind]
        candidates.sort(key=lambda r: float(r["confidence"] or 0), reverse=True)
        seen: set[str] = set()
        selected = []
        for row in candidates:
            if row["stem"] not in seen:
                selected.append(row); seen.add(row["stem"])
            if len(selected) >= limit:
                break
        folder = gallery / kind; folder.mkdir(parents=True, exist_ok=True)
        for rank, row in enumerate(selected, 1):
            image = Image.open(reviewed_index[row["stem"]]).convert("RGB")
            draw = ImageDraw.Draw(image)
            record = predictions[row["stem"]]
            gt = load_yolo_labels(REVIEWED / "labels/test" / f"{row['stem']}.txt",
                                  record["width"], record["height"])
            for item in gt:
                draw.rectangle(item["box"], outline=(70, 255, 100), width=3)
                draw.text((item["box"][0], max(0, item["box"][1] - 13)),
                          f"GT {NAMES[item['class_id']]}", fill=(70, 255, 100))
            for item in record["detections"]:
                if item["confidence"] < 0.25:
                    continue
                color = COLORS[item["class_id"]]
                draw.rectangle(item["box"], outline=color, width=2)
                draw.text((item["box"][0], item["box"][1] + 2),
                          f"P {NAMES[item['class_id']]} {item['confidence']:.2f}", fill=color)
            image.save(folder / f"{rank:03d}_{row['stem']}.jpg", quality=90)


def analyze(args: argparse.Namespace) -> None:
    predictions = load_predictions(args.predictions)
    reviewed_images = image_index(REVIEWED / "images/test")
    raw_images = image_index(RAW / "images/test")
    common = sorted(set(predictions) & set(reviewed_images) & set(raw_images))
    if len(common) != len(predictions):
        print(f"Warning: analyzing {len(common)}/{len(predictions)} prediction records present in both datasets")
    all_errors: list[dict[str, Any]] = []
    gt_changes: list[dict[str, Any]] = []
    summary: dict[str, Counter] = {"raw": Counter(), "reviewed": Counter()}
    per_class: dict[str, dict[str, Counter]] = {
        version: {name: Counter() for name in NAMES} for version in ("raw", "reviewed")
    }
    for stem in common:
        record = predictions[stem]; width, height = record["width"], record["height"]
        diagnostic_predictions = [item for item in record["detections"]
                                  if item["confidence"] >= args.analysis_conf]
        labels = {
            "raw": load_yolo_labels(RAW / "labels/test" / f"{stem}.txt", width, height),
            "reviewed": load_yolo_labels(REVIEWED / "labels/test" / f"{stem}.txt", width, height),
        }
        gt_changes.extend(compare_gt(stem, labels["raw"], labels["reviewed"]))
        for version in ("raw", "reviewed"):
            errors, counts = classify_image(stem, diagnostic_predictions, labels[version],
                                            width, height, version)
            all_errors.extend(errors); summary[version].update(counts)
            for truth in labels[version]:
                per_class[version][NAMES[truth["class_id"]]]["gt"] += 1
            for row in errors:
                cls = row["gt_class"] or row["pred_class"]
                per_class[version][cls][row["error_type"]] += 1

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "errors.csv", all_errors)
    write_csv(args.out / "raw_vs_reviewed_gt.csv", gt_changes,
              ["stem", "change", "class", "iou", "raw_box", "reviewed_box"])
    report = {
        "prediction_cache": str(args.predictions.resolve()), "images": len(common),
        "analysis_confidence": args.analysis_conf,
        "matching": {"tp": "same class IoU >= 0.5", "localization": "same class 0.1 <= IoU < 0.5",
                     "class_confusion": "different class IoU >= 0.5"},
        "aggregate": {k: dict(v) for k, v in summary.items()},
        "per_class": {v: {c: dict(x) for c, x in classes.items()}
                      for v, classes in per_class.items()},
        "gt_change_counts": dict(Counter(row["change"] for row in gt_changes)),
        "notes": ["Counts are diagnostic at IoU 0.5; publication AP remains Ultralytics COCO-style AP.",
                  "The identical cached predictions are used for raw and reviewed labels."],
    }
    (args.out / "error_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    draw_gallery(args.out, all_errors, predictions, reviewed_images, args.gallery_limit)
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Saved analysis to {args.out}")


def sweep_val(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "resolution_sweep_val.csv"
    done: dict[tuple[int, bool], dict[str, Any]] = {}
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                done[(int(row["imgsz"]), row["augment"].lower() == "true")] = row
    model = YOLO(str(args.model))
    rows = list(done.values())
    for imgsz in args.sizes:
        for augment in ([False, True] if args.tta else [False]):
            if (imgsz, augment) in done:
                print(f"Skip completed VAL config imgsz={imgsz}, augment={augment}")
                continue
            started = time.perf_counter()
            metrics = model.val(data=str(REVIEWED / "data.yaml"), split="val", imgsz=imgsz,
                                conf=0.001, iou=args.nms_iou, device=args.device,
                                augment=augment, plots=False, verbose=False,
                                project=str(args.out / "ultralytics"),
                                name=f"val_{imgsz}_{'tta' if augment else 'plain'}")
            row = {"imgsz": imgsz, "augment": augment, "map50": float(metrics.box.map50),
                   "map50_95": float(metrics.box.map), "precision": float(metrics.box.mp),
                   "recall": float(metrics.box.mr), "elapsed_seconds": time.perf_counter() - started}
            rows.append(row); write_csv(output, rows)
            print(row)
    print(f"VAL-only sweep saved to {output}. Choose one config before running Test.")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", type=Path, default=MODEL)
    common.add_argument("--out", type=Path, default=OUT)
    common.add_argument("--device", default="cpu")
    common.add_argument("--nms-iou", type=float, default=0.7)
    p = sub.add_parser("predict-test", parents=[common])
    p.add_argument("--imgsz", type=int, default=640); p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--augment", action="store_true"); p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=predict)
    a = sub.add_parser("analyze", parents=[common])
    a.add_argument("--predictions", type=Path, default=OUT / "predictions_imgsz640.jsonl")
    a.add_argument("--gallery-limit", type=int, default=50)
    a.add_argument("--analysis-conf", type=float, default=0.25,
                   help="Operational confidence used for FP/FN counts; cache remains at low confidence for AP")
    a.set_defaults(func=analyze)
    s = sub.add_parser("sweep-val", parents=[common])
    s.add_argument("--sizes", type=int, nargs="+", default=[640, 960, 1280])
    s.add_argument("--tta", action="store_true"); s.set_defaults(func=sweep_val)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    args.func(args)
