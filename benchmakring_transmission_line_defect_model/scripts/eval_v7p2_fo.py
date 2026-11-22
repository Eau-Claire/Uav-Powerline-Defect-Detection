"""Evaluate V7.2 candidates against seed85, with a foreign_object (FO) error breakdown.

For every model and split it writes:
  summary.csv / per_class.csv        official Ultralytics val metrics (same settings as V7.1 notebook)
  confusion_<model>_<split>.csv      GT class x predicted class (IoU >= --match-iou, one-to-one), plus missed / FP
  fo_outcomes_<model>_<split>.csv    one row per FO GT box: area bin, outcome, matched class, confidence, IoU
  fo_by_area.csv                     FO recall / FO->insulator_broken rate per area bin
  recall_by_size.csv                 recall per class and GT size (short side in a 640 reference)
  gallery/<model>_<split>/           images of FO GT boxes predicted as another class or missed
  gate.csv                           pass/fail against the baseline (only on val)

Test is evaluated only when --split test is given explicitly; choose checkpoints on val.

Example (Windows):
  python benchmakring_transmission_line_defect_model/scripts/eval_v7p2_fo.py ^
    --data benchmakring_transmission_line_defect_model/data/finding_nemo_6cls_v7p2_a_base_only/data_local.yaml ^
    --model seed85=weights/v7p1_seed85_curated_yolo26m_best.pt ^
    --model v7p1=runs/finding_v7p1_171126/yolo26m_finetune_v2_reviewed_181126/weights/best.pt ^
    --model v7p2b=runs/finding_v7p2/yolo26m_v7p2_b_fo_large/weights/best.pt ^
    --out benchmakring_transmission_line_defect_model/results/finding_v7p2/eval_val
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]
FO, IB = 4, 1
AREA_BINS = ((0.0, 0.01, "small_lt1pct"), (0.01, 0.05, "medium_1to5pct"), (0.05, 1.01, "large_ge5pct"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", action="append", required=True, help="name=path; the first one is the baseline")
    p.add_argument("--split", nargs="+", default=["val"], choices=["val", "test"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--imgsz-for", action="append", default=[],
                   help="name=size: evaluate one model at another image size (e.g. seed85_i960=960)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.25, help="confidence for the confusion analysis")
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--gallery", type=int, default=60, help="max FO error images per model/split")
    p.add_argument("--fo-tolerance", type=float, default=0.0, help="allowed FO AP50-95 drop on val for the gate")
    return p.parse_args(argv)


def area_bin(area):
    return next(name for lo, hi, name in AREA_BINS if lo <= area < hi)


def read_gt(path: Path):
    out = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            p = line.split()
            if len(p) == 5:
                c, x, y, w, h = int(float(p[0])), *map(float, p[1:])
                out.append((c, x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt, preds, thr):
    """Class-agnostic one-to-one matching by descending IoU. Returns {gt_idx: (pred_idx, iou)}."""
    pairs = sorted(((iou(g[1:], p[1:5]), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(preds)),
                   reverse=True)
    used_g, used_p, result = set(), set(), {}
    for v, gi, pi in pairs:
        if v < thr:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi); used_p.add(pi); result[gi] = (pi, v)
    return result


def resolve_split(data_yaml: Path, split: str):
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    images = root / cfg[split]  # e.g. <root>/images/val -> labels in <root>/labels/val
    labels = images.parent.parent / "labels" / images.name
    if not (images.is_dir() and labels.is_dir()):
        raise FileNotFoundError(f"{split}: expected {images} and {labels}")
    return images, labels


def draw(image_path: Path, gt_box, pred, label, dst: Path):
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        lw = max(2, W // 400)
        if pred is not None:  # red = prediction, drawn first so the green GT stays visible on top
            d.rectangle([pred[1] * W, pred[2] * H, pred[3] * W, pred[4] * H], outline=(230, 30, 30), width=lw * 2)
        d.rectangle([gt_box[1] * W, gt_box[2] * H, gt_box[3] * W, gt_box[4] * H], outline=(0, 220, 0), width=lw)
        font = ImageFont.load_default(size=max(16, W // 50))
        d.rectangle(d.textbbox((8, 8), label, font=font), fill=(0, 0, 0))
        d.text((8, 8), label, fill=(255, 255, 0), font=font)
        im.thumbnail((1280, 1280))
        im.save(dst, quality=88)


SIZE_BINS = ((0, 16, "<16px"), (16, 32, "16-32px"), (32, 64, "32-64px"), (64, 1e9, ">=64px"))


def size_bin(short_side_640):
    return next(name for lo, hi, name in SIZE_BINS if lo <= short_side_640 < hi)


def analyze(model, name, split, images_dir, labels_dir, a, out, imgsz):
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    conf_mat = defaultdict(Counter)  # gt class -> predicted class / "missed"
    fp = Counter()
    fo_rows = []
    size_hits = defaultdict(lambda: [0, 0])  # (class, size bin) -> [matched with right class, total]
    gallery = out / "gallery" / f"{name}_{split}"
    gallery.mkdir(parents=True, exist_ok=True)
    saved = 0
    for start in range(0, len(images), a.batch):
        chunk = images[start:start + a.batch]
        # rect=False: square letterbox, as in training and in the deployed endpoint.
        results = model.predict([str(p) for p in chunk], conf=a.conf, imgsz=imgsz, rect=False,
                                device=a.device, verbose=False)
        for img, r in zip(chunk, results):
            gt = read_gt(labels_dir / f"{img.stem}.txt")
            H, W = r.orig_shape
            b = r.boxes
            preds = [(int(c), *xy, float(s)) for c, xy, s in
                     zip(b.cls.tolist(), b.xyxyn.tolist(), b.conf.tolist())]
            m = match(gt, preds, a.match_iou)
            matched = {v[0] for v in m.values()}
            for pi, p in enumerate(preds):
                if pi not in matched:
                    fp[NAMES[p[0]]] += 1
            for gi, g in enumerate(gt):
                if gi in m:
                    pi, v = m[gi]
                    pc, pconf, pbox = NAMES[preds[pi][0]], preds[pi][5], preds[pi]
                else:
                    pc, pconf, v, pbox = "missed", "", "", None
                conf_mat[NAMES[g[0]]][pc] += 1
                # Size in a fixed 640 reference so every model bins the same GT box identically.
                side = min((g[3] - g[1]) * W, (g[4] - g[2]) * H) * 640 / max(W, H)
                hit = size_hits[(NAMES[g[0]], size_bin(side))]
                hit[0] += pc == NAMES[g[0]]
                hit[1] += 1
                if g[0] != FO:
                    continue
                area = (g[3] - g[1]) * (g[4] - g[2])
                fo_rows.append({"image": img.name, "area": round(area, 5), "area_bin": area_bin(area),
                                "outcome": pc, "confidence": pconf, "iou": round(v, 3) if v != "" else ""})
                if pc != "foreign_object" and saved < a.gallery:
                    draw(img, g, pbox, f"GT foreign_object -> {pc} {pconf if pconf == '' else round(pconf, 2)}",
                         gallery / f"{pc}__{img.stem}_{gi}.jpg")
                    saved += 1

    cols = NAMES + ["missed"]
    with (out / f"confusion_{name}_{split}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt\\pred"] + cols)
        for g in NAMES:
            w.writerow([g] + [conf_mat[g][c] for c in cols])
        w.writerow(["background_fp"] + [fp[c] for c in NAMES] + [""])
    with (out / f"fo_outcomes_{name}_{split}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "area", "area_bin", "outcome", "confidence", "iou"])
        w.writeheader(); w.writerows(fo_rows)

    by_bin = []
    for *_, bn in AREA_BINS + (("", "", "all"),):
        rows = [r for r in fo_rows if bn == "all" or r["area_bin"] == bn]
        n = len(rows)
        oc = Counter(r["outcome"] for r in rows)
        by_bin.append({"model": name, "split": split, "area_bin": bn, "fo_gt": n,
                       "recall_at_conf": round(oc["foreign_object"] / n, 4) if n else "",
                       "as_insulator_broken": oc["insulator_broken"], "as_other_class":
                       n - oc["foreign_object"] - oc["insulator_broken"] - oc["missed"], "missed": oc["missed"]})
    ib_as_fo = conf_mat["insulator_broken"]["foreign_object"]
    print(f"  [{name}/{split}] FO->IB={conf_mat['foreign_object']['insulator_broken']} "
          f"IB->FO={ib_as_fo} FO missed={conf_mat['foreign_object']['missed']} FO FP={fp['foreign_object']}")
    size_rows = [{"model": name, "split": split, "class": c, "size_bin_640": b, "gt": n,
                  "recall_at_conf": round(t / n, 4)}
                 for c in NAMES for *_, b in SIZE_BINS for t, n in [size_hits.get((c, b), (0, 0))] if n]
    return by_bin, size_rows


def main(argv=None):
    a = parse_args(argv)
    models = []
    for spec in a.model:
        name, _, path = spec.partition("=")
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"--model must be name=existing_path, got {spec}")
        models.append((name, Path(path)))
    imgsz_for = {}
    for spec in a.imgsz_for:
        name, _, size = spec.partition("=")
        if name not in {n for n, _ in models} or not size.isdigit():
            raise ValueError(f"--imgsz-for must be model_name=size for a given --model, got {spec}")
        imgsz_for[name] = int(size)
    a.out.mkdir(parents=True, exist_ok=True)
    summary, per_class, fo_bins, size_rows = [], [], [], []

    for name, ckpt in models:
        model = YOLO(str(ckpt))
        imgsz = imgsz_for.get(name, a.imgsz)
        for split in a.split:
            print(f"\n===== {name} / {split} (imgsz {imgsz}) =====")
            m = model.val(data=str(a.data), split=split, imgsz=imgsz, batch=a.batch, device=a.device,
                          project=str(a.out / "runs"), name=f"{name}_{split}", exist_ok=True,
                          plots=True, save_json=False, verbose=False)
            summary.append({"model": name, "split": split, "imgsz": imgsz, "precision": float(m.box.mp), "recall": float(m.box.mr),
                            "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)})
            for i, c in enumerate(m.box.ap_class_index.astype(int).tolist()):
                per_class.append({"model": name, "split": split, "class": m.names[c],
                                  "precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
                                  "AP50": float(m.box.ap50[i]), "AP50_95": float(m.box.ap[i])})
            images_dir, labels_dir = resolve_split(a.data, split)
            bins, sizes = analyze(model, name, split, images_dir, labels_dir, a, a.out, imgsz)
            fo_bins += bins
            size_rows += sizes

    def write(rows, fname):
        if not rows:
            return
        with (a.out / fname).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    write(summary, "summary.csv"); write(per_class, "per_class.csv"); write(fo_bins, "fo_by_area.csv")
    write(size_rows, "recall_by_size.csv")

    gate = []
    base = models[0][0]
    if "val" in a.split:
        S = {(r["model"], r["split"]): r for r in summary}
        P = {(r["model"], r["split"], r["class"]): r["AP50_95"] for r in per_class}
        for name, _ in models[1:]:
            d_map = S[name, "val"]["mAP50_95"] - S[base, "val"]["mAP50_95"]
            d_m50 = S[name, "val"]["mAP50"] - S[base, "val"]["mAP50"]
            # A class without GT in the split has no AP for any model: nothing to compare.
            present = [c for c in NAMES if (name, "val", c) in P and (base, "val", c) in P]
            d_fo = (P[name, "val", "foreign_object"] - P[base, "val", "foreign_object"]
                    if "foreign_object" in present else 0.0)
            better = [c for c in present if P[name, "val", c] > P[base, "val", c]]
            ok = d_map >= 0 and d_fo >= -a.fo_tolerance
            gate.append({"model": name, "baseline": base, "d_mAP50": round(d_m50, 4), "d_mAP50_95": round(d_map, 4),
                         "d_FO_AP50_95": round(d_fo, 4), "classes_improved": "|".join(better),
                         "gate": "PASS" if ok else "FAIL"})
        if gate:
            write(gate, "gate.csv")
            print("\nVal gate vs", base)
            for g in gate:
                print(f"  {g['model']:>12}: {g['gate']}  dmAP50-95={g['d_mAP50_95']:+.4f} "
                      f"dFO={g['d_FO_AP50_95']:+.4f} improved={g['classes_improved']}")
    (a.out / "eval_results.json").write_text(json.dumps({
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "data": str(a.data), "models": [list(map(str, m)) for m in models],
        "imgsz_for": imgsz_for, "summary": summary, "per_class": per_class, "fo_by_area": fo_bins,
        "recall_by_size": size_rows, "gate": gate}, indent=2), encoding="utf-8")
    print("\nSaved:", a.out)


def evaluate(*argv):
    """Notebook entry point: evaluate("--data", path, "--model", "a=b.pt", ...)."""
    main([str(x) for x in argv])


if __name__ == "__main__":
    main()
