#!/usr/bin/env python
"""Pi student: square 640x640 letterbox (current NCNN export) vs native-aspect input (rect letterbox, long side 640 =
what an NCNN export at the camera aspect, e.g. 640x480 for 4:3 drone frames, sees). Caches predictions like
system_two_tier.run (sys_<tag>.json) and prints Pi-only per-class AP50 on val_clean_v8 without leaked images, with a
paired bootstrap rect - square per model.
Usage: python scripts/exp_pi_aspect.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import eval_v8 as E  # noqa: E402

MODELS = {"n_v5": REPO / "artifacts/v6p5_v6_teachers/init/student_n_v5.pt",
          "gn_v6": REPO / "artifacts/v6p5_v7_gentle/runs/gn_v6/weights/best.pt",
          "s_v5": REPO / "artifacts/v6p5_v5_students/runs/student_s/weights/best.pt"}
SQUARE_CACHE = {"n_v5": "nb22_pi_n_v5_640", "gn_v6": "nb23_gn_v6_640"}


def predict(w, imgs, tag, rect):
    f = E.PF / f"sys_{tag}.json"
    if f.exists(): return json.loads(f.read_text())["dets"]
    m = YOLO(str(w)); out = {}; t0 = time.time()
    for k, im in enumerate(imgs):
        r = m.predict(str(im), imgsz=640, conf=0.001, rect=rect, verbose=False, device=0)[0]
        out[im.name] = [[int(c), float(s), *b] for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxyn.tolist())]
        if k % 500 == 0: print(f"  {tag}: {k}/{len(imgs)} {(time.time() - t0) / 60:.1f} min", flush=True)
    f.write_text(json.dumps({"_weights": str(w), "rect": rect, "dets": out})); return out


def main() -> int:
    imgs = sorted((E.DS / "images/val_clean_v8").glob("*.jpg")); leak = set((E.DS / "leaked_val.txt").read_text().split())
    NL = [p.name for p in imgs if p.name not in leak]; idx = {n: i for i, n in enumerate(NL)}; G = E.gt(NL)
    size = {p.name: __import__("PIL.Image", fromlist=["Image"]).open(p).size for p in imgs}
    drone = [n for n in NL if size[n][0] != size[n][1]]
    rng = np.random.default_rng(20261016)
    W = [np.bincount(rng.integers(0, len(NL), len(NL)), minlength=len(NL)).astype(float) for _ in range(1000)]
    for name, w in MODELS.items():
        D = {"square": E.dets(SQUARE_CACHE[name]) if name in SQUARE_CACHE else predict(w, imgs, f"pi_{name}_sq", False),
             "rect": predict(w, imgs, f"pi_{name}_rect", True)}
        R = {}
        for k, d in D.items():
            pre = [E.prep(d, G[c], c, set(NL), idx) for c in range(1, 6)]
            full = [E.wap(*x, np.ones(len(NL))) for x in pre]; R[k] = np.array([[E.wap(*x, ww) for x in pre] for ww in W])
            print(f"{name:6s} {k:6s} MEAN_5 {np.mean(full):.4f} | " + " ".join(f"{s} {v:.3f}" for s, v in zip(E.SHORT, full)), flush=True)
        d = R["rect"] - R["square"]; m = d.mean(1)
        print(f"{name:6s} rect − square: MEAN_5 {m.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}] "
              f"P>0 {(m > 0).mean():.0%} | " + " ".join(f"{s} {d[:, i].mean():+.3f}" for i, s in enumerate(E.SHORT)), flush=True)
    print(f"(ảnh không vuông = khung drone: {len(drone)}/{len(NL)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
