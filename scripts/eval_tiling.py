#!/usr/bin/env python
"""Evaluate server tiling (scripts/exp_tiling.py cache) in the two-tier system on val_clean_v8 without leaked images:
tile boxes are merged with the full-image server boxes per class (NMS IoU 0.5, higher score kept), for each subset of
classes; paired image bootstrap against the current server (v5_orig, FO@800). Prints per-class AP50 and differences.
Usage: python scripts/eval_tiling.py [--tag tile2x2_v5_orig] [--b 1000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import eval_v8 as E  # noqa: E402


def nms(boxes, thr=0.5):
    if not boxes: return []
    b = np.array(boxes, float); keep = []
    for c in np.unique(b[:, 0]):
        x = b[b[:, 0] == c]; x = x[np.argsort(-x[:, 1])]; alive = np.ones(len(x), bool)
        for i in range(len(x)):
            if not alive[i]: continue
            keep.append(x[i].tolist())
            if i + 1 < len(x):
                u = E.iou(x[i:i + 1, 2:], x[i + 1:, 2:])[0]; alive[i + 1:] &= u < thr
    return [[int(k[0]), *k[1:]] for k in keep]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="tile2x2_v5_orig")
    p.add_argument("--b", type=int, default=1000)
    a = p.parse_args()
    ALL = sorted(f.name for f in (E.DS / "images/val_clean_v8").glob("*.jpg"))
    leak = set((E.DS / "leaked_val.txt").read_text().split())
    NL = [n for n in ALL if n not in leak]; idx = {n: i for i, n in enumerate(NL)}; G = E.gt(NL)
    f = E.flagged(E.dets("nb22_pi_n_v5_640"), E.TAU)
    s6, s8, tl = E.dets("nb22_v5_orig_640"), E.dets("nb22_v5_orig_800"), E.dets(a.tag)

    def system(tile_cls):
        out = {}
        for n in NL:
            if n not in f: out[n] = []; continue
            base = [d for d in s6[n] if d[0] != 4] + [d for d in s8[n] if d[0] == 4]
            extra = [d for d in tl[n] if d[0] in tile_cls]
            out[n] = nms(base + extra) if extra else base
        return out

    cfg = {"hiện tại": set(), "tile broken": {1}, "tile FO": {4}, "tile bird_nest": {3}, "tile flashover": {2},
           "tile strand": {5}, "tile broken+FO": {1, 4}}
    rng = np.random.default_rng(20261016)
    W = [np.bincount(rng.integers(0, len(NL), len(NL)), minlength=len(NL)).astype(float) for _ in range(a.b)]
    R = {}
    for k, c in cfg.items():
        D = system(c); pre = [E.prep(D, G[cc], cc, set(NL), idx) for cc in range(1, 6)]
        full = [E.wap(*x, np.ones(len(NL))) for x in pre]; R[k] = np.array([[E.wap(*x, w) for x in pre] for w in W])
        print(f"{k:16s} MEAN_5 {np.mean(full):.4f} | " + " ".join(f"{s} {v:.3f}" for s, v in zip(E.SHORT, full)), flush=True)
    for k in list(cfg)[1:]:
        d = R[k] - R["hiện tại"]; m = d.mean(1)
        print(f"{k:16s} − hiện tại: MEAN_5 {m.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}] "
              f"P>0 {(m > 0).mean():.0%} | " + " ".join(f"{s} {d[:, i].mean():+.3f}" for i, s in enumerate(E.SHORT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
