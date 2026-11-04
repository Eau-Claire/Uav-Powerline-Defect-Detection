#!/usr/bin/env python
"""Compare model predictions against GT labels and list candidate label errors for review.

Issue types (one row per issue in issues.csv):
  missing      confident prediction with no overlapping GT (IoU < --miss-iou with every GT box)
  class_flip   confident prediction overlapping a GT box of another class, GT box not matched by its own class
  loose_box    GT matched by same-class prediction but IoU in [0.5, --tight-iou)
  unseen_gt    GT box with no prediction of any class at IoU >= 0.3 (possible wrong / spurious label)
  dup_gt       two GT boxes of the same class with IoU >= 0.8
  tiny_gt      GT box narrower than 4 px or smaller than 0.01% of the image
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


def xywh2xyxy(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    return np.stack([a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2], 1)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def read_gt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [l.split() for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    a = np.array(rows, dtype=float)
    return a[:, 0].astype(int), a[:, 1:5]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--pred-dir", required=True, type=Path)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--hi-conf", type=float, default=0.5)
    p.add_argument("--match-conf", type=float, default=0.25)
    p.add_argument("--miss-iou", type=float, default=0.3)
    p.add_argument("--tight-iou", type=float, default=0.75)
    p.add_argument("--gt-only", action="store_true", help="only run checks that need no predictions")
    a = p.parse_args()

    cfg = yaml.safe_load(a.data.read_text())
    names = cfg["names"] if isinstance(cfg["names"], list) else [cfg["names"][i] for i in sorted(cfg["names"])]
    root = Path(cfg["path"])
    issues, summary = [], {}
    for split in a.splits:
        lbl_dir = root / cfg[split].replace("images", "labels", 1)
        preds = {}
        if not a.gt_only:
            for line in (a.pred_dir / f"pred_{split}.jsonl").open():
                r = json.loads(line)
                preds[r["image"]] = r
            images = sorted(preds)
        else:
            images = sorted(x.name for x in (root / cfg[split]).glob("*.*"))
        cnt, img_hit = Counter(), defaultdict(set)
        iou_same = defaultdict(list)
        for name in images:
            gc, gb = read_gt(lbl_dir / (Path(name).stem + ".txt"))
            gx = xywh2xyxy(gb)
            cnt["images"] += 1
            cnt["gt_boxes"] += len(gc)

            def add(kind, cls, conf, box, extra=""):
                issues.append({"split": split, "image": name, "type": kind, "class": names[cls], "conf": conf,
                               "x": round(box[0], 5), "y": round(box[1], 5), "w": round(box[2], 5),
                               "h": round(box[3], 5), "extra": extra, "gt_classes": "|".join(sorted({names[c] for c in gc}))})
                cnt[f"{kind}:{names[cls]}"] += 1
                img_hit[kind].add(name)

            # GT-only checks
            if len(gc) > 1:
                m = iou(gx, gx)
                for i in range(len(gc)):
                    for j in range(i + 1, len(gc)):
                        if gc[i] == gc[j] and m[i, j] >= 0.8:
                            add("dup_gt", gc[i], "", gb[j], f"iou={m[i, j]:.2f}")
            if name in preds:
                W, H = preds[name]["wh"]
            else:
                W = H = 640
            for c, b in zip(gc, gb):
                if b[2] * W < 4 or b[3] * H < 4 or b[2] * b[3] < 1e-4:
                    add("tiny_gt", c, "", b, f"{b[2] * W:.1f}x{b[3] * H:.1f}px")
            if a.gt_only:
                continue

            r = preds[name]
            pc, pf, pb = np.array(r["cls"], int), np.array(r["conf"]), np.array(r["xywhn"]).reshape(-1, 4)
            px = xywh2xyxy(pb)
            m = iou(px, gx)  # [P, G]
            # same-class matches (greedy by conf)
            gt_matched = np.full(len(gc), -1)
            pred_matched = np.zeros(len(pc), bool)
            for pi in np.argsort(-pf):
                if pf[pi] < a.match_conf:
                    break
                cand = [(m[pi, g], g) for g in range(len(gc)) if gc[g] == pc[pi] and gt_matched[g] < 0 and m[pi, g] >= 0.5]
                if cand:
                    v, g = max(cand)
                    gt_matched[g] = pi
                    pred_matched[pi] = True
                    iou_same[names[gc[g]]].append(v)
                    if v < a.tight_iou and pf[pi] >= 0.6:
                        add("loose_box", gc[g], round(float(pf[pi]), 3), gb[g],
                            f"iou={v:.2f};pred={','.join(f'{x:.4f}' for x in pb[pi])}")
            for pi in range(len(pc)):
                if pred_matched[pi] or pf[pi] < a.hi_conf:
                    continue
                best = m[pi].max() if len(gc) else 0.0
                if best < a.miss_iou:
                    add("missing", pc[pi], round(float(pf[pi]), 3), pb[pi], f"max_iou={best:.2f}")
                else:
                    g = int(m[pi].argmax())
                    if gc[g] != pc[pi] and gt_matched[g] < 0 and best >= 0.5:
                        add("class_flip", pc[pi], round(float(pf[pi]), 3), pb[pi],
                            f"gt={names[gc[g]]};iou={best:.2f}")
            for g in range(len(gc)):
                if gt_matched[g] < 0 and (not len(pc) or m[:, g].max() < 0.3):
                    add("unseen_gt", gc[g], "", gb[g])
        summary[split] = {
            "counts": dict(sorted(cnt.items())),
            "images_with_issue": {k: len(v) for k, v in img_hit.items()},
            "matched_iou_median": {k: round(float(np.median(v)), 3) for k, v in iou_same.items()},
            "matched_iou_lt075_frac": {k: round(float(np.mean(np.array(v) < 0.75)), 3) for k, v in iou_same.items()},
        }
    out = a.pred_dir / ("gt_checks" if a.gt_only else "audit")
    out.mkdir(exist_ok=True)
    with (out / "issues.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(issues[0]) if issues else ["split"])
        w.writeheader()
        w.writerows(issues)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
