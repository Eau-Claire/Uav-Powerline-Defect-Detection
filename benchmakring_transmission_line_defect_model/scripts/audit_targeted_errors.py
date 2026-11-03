#!/usr/bin/env python
"""Build a targeted visual audit pack for insulator and broken_strand errors."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts/finding_nemo_selected_checkpoints/yolo26m_epoch45.pt"
DEFAULT_DATA = ROOT / "data/test_main_v1/dataset.yaml"
DEFAULT_OUT = ROOT / "results/finding_nemo/targeted_audit_test_main_v1"
NAMES = {0: "insulator", 1: "insulator_broken", 2: "insulator_flashover", 3: "bird_nest", 4: "foreign_object", 5: "broken_strand"}
TARGETS = {0, 5}


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(aa[:, None] + bb[None, :] - inter, 1e-9, None)


def read_gt(label: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    classes, boxes = [], []
    if label.is_file():
        for line in label.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, xc, yc, bw, bh = map(float, parts[:5])
            classes.append(int(cls))
            boxes.append([(xc - bw / 2) * width, (yc - bh / 2) * height,
                          (xc + bw / 2) * width, (yc + bh / 2) * height])
    return np.asarray(classes, dtype=np.int64), np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def greedy_pairs(iou: np.ndarray, valid: np.ndarray, used_g: set[int], used_p: set[int]):
    candidates = [(float(iou[g, p]), g, p) for g, p in zip(*np.where(valid))
                  if g not in used_g and p not in used_p]
    for score, g, p in sorted(candidates, reverse=True):
        if g not in used_g and p not in used_p:
            used_g.add(g); used_p.add(p)
            yield score, g, p


def event(kind, cls, box, confidence="", iou="", counterpart=""):
    return {"error_type": kind, "class_id": int(cls), "class": NAMES[int(cls)],
            "confidence": confidence, "iou": iou, "counterpart": counterpart,
            "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3])}


def classify(gt_c, gt_b, pr_c, pr_b, pr_conf):
    ious = iou_matrix(gt_b, pr_b)
    used_g, used_p, events = set(), set(), []
    same = gt_c[:, None] == pr_c[None, :] if len(gt_c) and len(pr_c) else np.zeros_like(ious, dtype=bool)
    for score, g, p in greedy_pairs(ious, same & (ious >= .5), used_g, used_p):
        if int(gt_c[g]) in TARGETS:
            events.append(event("TP", gt_c[g], gt_b[g], float(pr_conf[p]), score))
    confusion = ((gt_c[:, None] == 0) & (pr_c[None, :] == 1)) | ((gt_c[:, None] == 1) & (pr_c[None, :] == 0)) if len(gt_c) and len(pr_c) else np.zeros_like(ious, dtype=bool)
    for score, g, p in greedy_pairs(ious, confusion & (ious >= .3), used_g, used_p):
        if int(gt_c[g]) == 0:
            events.append(event("CLASS_CONFUSION", gt_c[g], gt_b[g], float(pr_conf[p]), score, NAMES[int(pr_c[p])]))
    for score, g, p in greedy_pairs(ious, same & (ious >= .1), used_g, used_p):
        if int(gt_c[g]) in TARGETS:
            events.append(event("LOCALIZATION", gt_c[g], gt_b[g], float(pr_conf[p]), score))
    for g in range(len(gt_c)):
        if g not in used_g and int(gt_c[g]) in TARGETS:
            events.append(event("FN", gt_c[g], gt_b[g]))
    for p in range(len(pr_c)):
        if p not in used_p and int(pr_c[p]) in TARGETS:
            events.append(event("FP", pr_c[p], pr_b[p], float(pr_conf[p])))
    return events


def severity(events, cls):
    weights = {"FN": 2.0, "CLASS_CONFUSION": 1.5, "FP": 1.0, "LOCALIZATION": .5, "TP": 0.0}
    return sum(weights[e["error_type"]] for e in events if e["class_id"] == cls)


def draw_pack(image_path: Path, gt_c, gt_b, pr_c, pr_b, pr_conf, output: Path, title: str):
    im = cv2.imread(str(image_path))
    if im is None:
        return
    for cls, box in zip(gt_c, gt_b):
        if int(cls) not in TARGETS | {1}: continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 210, 0), 2)
        cv2.putText(im, f"GT {NAMES[int(cls)]}", (x1, max(15, y1 - 4)), 0, .48, (0, 210, 0), 1, cv2.LINE_AA)
    for cls, box, conf in zip(pr_c, pr_b, pr_conf):
        if int(cls) not in TARGETS | {1}: continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 80, 255), 2)
        cv2.putText(im, f"P {NAMES[int(cls)]} {conf:.2f}", (x1, min(im.shape[0] - 5, y2 + 16)), 0, .48, (0, 80, 255), 1, cv2.LINE_AA)
    cv2.rectangle(im, (0, 0), (im.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(im, title, (8, 20), 0, .55, (255, 255, 255), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), im)


def write_csv(path: Path, rows: list[dict]):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def class_aware_nms(classes: np.ndarray, boxes: np.ndarray, scores: np.ndarray, threshold: float):
    """Remove duplicate predictions within each class for audit visualization only."""
    keep = []
    for cls in np.unique(classes):
        indices = np.where(classes == cls)[0]
        chosen = nms(torch.as_tensor(boxes[indices]), torch.as_tensor(scores[indices]), threshold)
        keep.extend(indices[chosen.cpu().numpy()].tolist())
    keep = np.asarray(sorted(keep, key=lambda idx: float(scores[idx]), reverse=True), dtype=np.int64)
    return classes[keep], boxes[keep], scores[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=.10)
    ap.add_argument("--audit-nms-iou", type=float, default=.50,
                    help="Class-aware NMS IoU used only for the audit pack (default: 0.50)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--error-per-class", type=int, default=75)
    ap.add_argument("--tp-per-class", type=int, default=25)
    args = ap.parse_args()
    device = "0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    cfg = yaml.safe_load(args.dataset.resolve().read_text(encoding="utf-8"))
    root = Path(cfg["path"]); image_dir = root / cfg["test"]
    label_dir = Path(str(image_dir).replace("images", "labels"))
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if args.limit: images = images[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model.resolve()))
    summaries, all_events, cache = [], [], {}
    def prediction_pairs():
        for start in range(0, len(images), args.batch):
            chunk = images[start:start + args.batch]
            chunk_results = model.predict(source=[str(p) for p in chunk], imgsz=640, conf=args.conf,
                                          iou=.7, max_det=300, device=device, batch=args.batch,
                                          stream=False, verbose=False)
            yield from zip(chunk, chunk_results)
    for idx, (image_path, result) in enumerate(prediction_pairs(), 1):
        h, w = result.orig_shape
        gt_c, gt_b = read_gt(label_dir / f"{image_path.stem}.txt", w, h)
        if result.boxes is None:
            pr_c=np.empty(0,dtype=np.int64); pr_b=np.empty((0,4),dtype=np.float32); pr_conf=np.empty(0)
        else:
            pr_c=result.boxes.cls.cpu().numpy().astype(np.int64); pr_b=result.boxes.xyxy.cpu().numpy(); pr_conf=result.boxes.conf.cpu().numpy()
            pr_c, pr_b, pr_conf = class_aware_nms(pr_c, pr_b, pr_conf, args.audit_nms_iou)
        events = classify(gt_c, gt_b, pr_c, pr_b, pr_conf)
        cache[str(image_path)] = (gt_c, gt_b, pr_c, pr_b, pr_conf)
        counts = defaultdict(int)
        for e in events:
            counts[f"{e['class']}_{e['error_type']}"] += 1
            area = max(0., e["x2"]-e["x1"]) * max(0., e["y2"]-e["y1"]) / (w*h)
            all_events.append({"image": str(image_path), "width": w, "height": h, "relative_area": round(area, 8), **e})
        row = {"image": str(image_path), "insulator_severity": severity(events,0), "broken_strand_severity": severity(events,5)}
        for cls in ("insulator", "broken_strand"):
            for kind in ("TP","FP","FN","CLASS_CONFUSION","LOCALIZATION"):
                row[f"{cls}_{kind}"] = counts[f"{cls}_{kind}"]
        summaries.append(row)
        if idx % 100 == 0: print(f"Processed {idx}/{len(images)}", flush=True)
    selected=[]
    for cls_id, cls_name in ((0,"insulator"),(5,"broken_strand")):
        errors=sorted((r for r in summaries if r[f"{cls_name}_severity"]>0), key=lambda r:r[f"{cls_name}_severity"], reverse=True)[:args.error_per_class]
        controls=[r for r in summaries if r[f"{cls_name}_severity"]==0 and r[f"{cls_name}_TP"]>0][:args.tp_per_class]
        for rank, row in enumerate(errors+controls,1):
            category="error" if row[f"{cls_name}_severity"]>0 else "tp_control"
            record={"audit_class":cls_name,"selection":category,"rank":rank,**row,"review_verdict":"","review_note":""}
            selected.append(record)
            p=Path(row["image"]); vals=cache[str(p)]
            draw_pack(p,*vals,args.output/"previews"/cls_name/category/f"{rank:03d}_{p.stem}.jpg",f"{cls_name} | {category} | severity={row[f'{cls_name}_severity']:.1f}")
    write_csv(args.output/"image_summary.csv", summaries)
    write_csv(args.output/"error_events.csv", all_events)
    write_csv(args.output/"selected_for_review.csv", selected)
    meta={"model":str(args.model.resolve()),"dataset":str(args.dataset.resolve()),"images":len(images),"device":device,"conf":args.conf,
          "audit_nms_iou":args.audit_nms_iou,
          "matching":{"TP":"same class IoU >= 0.5","CLASS_CONFUSION":"insulator vs insulator_broken IoU >= 0.3","LOCALIZATION":"same class 0.1 <= IoU < 0.5"},
          "severity":"2*FN + 1.5*CLASS_CONFUSION + FP + 0.5*LOCALIZATION"}
    (args.output/"audit_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(f"Audit pack: {args.output.resolve()}")


if __name__ == "__main__": main()
