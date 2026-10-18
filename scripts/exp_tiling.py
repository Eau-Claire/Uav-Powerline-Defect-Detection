#!/usr/bin/env python
"""Server-side tiling for tiny defects (insulator_broken chips): each val image is cut into 2x2 tiles with 20% overlap,
every tile is predicted at 640 (objects ~2x larger than in the full-image pass). From the tiles only small boxes
(area < SMALL of the image) that do not touch an inner tile edge are kept (a large object cut by a tile edge gives a
partial box). Output: data/label_audit/preflight/sys_<tag>.json with the tile boxes only (same format as
system_two_tier.run), so they can be merged per class with the full-image cache in an evaluation.
Usage: python scripts/exp_tiling.py [--weights W] [--tag T] [--split val_clean_v8]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
PF = REPO / "data/label_audit/preflight"
SMALL, OVERLAP, EDGE = 0.02, 0.2, 0.01


def tiles(W, H):
    tw, th = int(W * (0.5 + OVERLAP / 2)), int(H * (0.5 + OVERLAP / 2))
    return [(x, y, x + tw, y + th) for y in (0, H - th) for x in (0, W - tw)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(REPO / "artifacts/v6p5_v6_teachers/init/teacher_v5.pt"))
    p.add_argument("--tag", default="tile2x2_v5_orig")
    p.add_argument("--split", default="val_clean_v8")
    a = p.parse_args()
    imgs = sorted((REPO / "datasets/v6p5_v8/images" / a.split).glob("*.jpg"))
    m = YOLO(a.weights); out = {}; t0 = time.time()
    import cv2
    for k, im in enumerate(imgs):
        img = cv2.imread(str(im)); H, W = img.shape[:2]; T = tiles(W, H)
        res = m.predict([img[y1:y2, x1:x2] for x1, y1, x2, y2 in T], imgsz=640, conf=0.001, rect=False, verbose=False,
                        device=0, batch=4)
        boxes = []
        for (x1, y1, x2, y2), r in zip(T, res):
            for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                gx1, gy1, gx2, gy2 = b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1
                if (gx2 - gx1) * (gy2 - gy1) / (W * H) >= SMALL: continue
                inner = [(b[0] < EDGE * W and x1 > 0), (b[1] < EDGE * H and y1 > 0),
                         (b[2] > (x2 - x1) - EDGE * W and x2 < W), (b[3] > (y2 - y1) - EDGE * H and y2 < H)]
                if any(inner): continue
                boxes.append([int(c), round(float(s), 5), gx1 / W, gy1 / H, gx2 / W, gy2 / H])
        out[im.name] = boxes
        if k % 300 == 0 or k == len(imgs) - 1:
            el = time.time() - t0; print(f"{k + 1}/{len(imgs)} | {el / 60:.1f} min | còn ~{el / (k + 1) * (len(imgs) - k - 1) / 60:.1f} min", flush=True)
    (PF / f"sys_{a.tag}.json").write_text(json.dumps({"_weights": a.weights, "tiles": "2x2 overlap 0.2, small only", "dets": out}))
    print("saved", PF / f"sys_{a.tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
