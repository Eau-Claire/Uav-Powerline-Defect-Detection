#!/usr/bin/env python
"""Detect images whose GT labels are mirrored relative to the pixels, by testing which transform of the
GT (identity / hflip / vflip / both) agrees best with model predictions. Writes flips.csv."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from label_audit_analyze import iou, read_gt, xywh2xyxy  # noqa: E402

TRANSFORMS = {
    "none": lambda b: b,
    "hflip": lambda b: np.c_[1 - b[:, 0], b[:, 1:]],
    "vflip": lambda b: np.c_[b[:, 0], 1 - b[:, 1], b[:, 2:]],
    "hvflip": lambda b: np.c_[1 - b[:, 0], 1 - b[:, 1], b[:, 2:]],
}


def score(gc, gb, pc, pf, pb, conf):
    """Mean over GT boxes of best same-class IoU with a prediction of conf >= conf."""
    keep = pf >= conf
    pc, pb = pc[keep], pb[keep]
    if not len(gc) or not len(pc):
        return 0.0
    m = iou(xywh2xyxy(gb), xywh2xyxy(pb))
    m[gc[:, None] != pc[None, :]] = 0
    return float(m.max(1).mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--pred-dir", required=True, type=Path)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--min-gain", type=float, default=0.3)
    p.add_argument("--min-score", type=float, default=0.5)
    a = p.parse_args()
    cfg = yaml.safe_load(a.data.read_text())
    root = Path(cfg["path"])
    out_rows = []
    for split in a.splits:
        lbl_dir = root / cfg[split].replace("images", "labels", 1)
        for line in (a.pred_dir / f"pred_{split}.jsonl").open():
            r = json.loads(line)
            gc, gb = read_gt(lbl_dir / (Path(r["image"]).stem + ".txt"))
            if not len(gc):
                continue
            pc, pf = np.array(r["cls"], int), np.array(r["conf"])
            pb = np.array(r["xywhn"]).reshape(-1, 4)
            s = {k: score(gc, f(gb), pc, pf, pb, a.conf) for k, f in TRANSFORMS.items()}
            best = max(s, key=s.get)
            if best != "none" and s[best] >= a.min_score and s[best] - s["none"] >= a.min_gain:
                out_rows.append({"split": split, "image": r["image"], "transform": best, "n_gt": len(gc),
                                 **{f"score_{k}": round(v, 3) for k, v in s.items()}})
    out = a.pred_dir / "audit/flips.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "image", "transform", "n_gt"] + [f"score_{k}" for k in TRANSFORMS])
        w.writeheader()
        w.writerows(out_rows)
    from collections import Counter
    print(out, Counter((r["split"], r["transform"]) for r in out_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
