#!/usr/bin/env python
"""Does a Pi student trained at a higher input size catch the tiny defects it misses at 640? (Pi n_v5 misses 32% of the
images whose largest insulator_broken is < 1% of the image, 7% otherwise.) Two arms, same recipe, only imgsz differs:
init student n_v5, distill from teacher v5_orig, dataset v6p5_v9 (no val leak, test_fo held out), lr 1e-4, no warmup
(lr 1e-3 + warmup broke the converged student before), mosaic off for the last 5 epochs, 20 epochs, seed fixed.
Eval: Pi-only on val_clean_v8 without leaked images, each arm at its own imgsz (square, as the NCNN export), per-class
AP50 + image-level hit rate at conf 0.25 split by defect size, paired bootstrap vs n_v5 and 800 vs 640.
Resume: a finished arm (epoch == -1 in last.pt) is skipped; an interrupted one restarts.  ~6 h on an RTX 4060 laptop.
Usage: python scripts/exp_student_res.py [--epochs 20] [--sizes 640 800]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import eval_v8 as E  # noqa: E402

OUT = REPO / "artifacts/exp_student_res"
INIT = REPO / "artifacts/v6p5_v6_teachers/init/student_n_v5.pt"
TEACHER = REPO / "artifacts/v6p5_v6_teachers/init/teacher_v5.pt"
DATA = REPO / "datasets/v6p5_v9/data.yaml"


def log(m):
    print(m, flush=True)
    with open(OUT / "progress.log", "a", encoding="utf-8") as f: f.write(f"{time.strftime('%m-%d %H:%M')} {m}\n")


def train(sz, epochs):
    run = OUT / f"n_{sz}"; w = run / "weights/last.pt"
    if w.exists() and torch.load(w, map_location="cpu", weights_only=False).get("epoch") == -1:
        log(f"[n_{sz}] đã xong"); return run / "weights/best.pt"
    shutil.rmtree(run, ignore_errors=True); m = YOLO(str(INIT)); t0 = time.time()
    m.add_callback("on_fit_epoch_end", lambda tr: log(f"[n_{sz}] epoch {tr.epoch + 1}/{tr.epochs} | {(time.time() - t0) / 60:.0f} min"))
    m.train(data=str(DATA), imgsz=sz, epochs=epochs, batch=8, lr0=1e-4, lrf=0.1, cos_lr=True, warmup_epochs=0,
            close_mosaic=5, optimizer="AdamW", seed=20261016, deterministic=True, cache="disk", workers=4, amp=True,
            val=False, plots=False, distill_model=str(TEACHER), project=str(OUT), name=f"n_{sz}", exist_ok=True)
    return run / "weights/best.pt"


def predict(w, imgs, sz, tag):
    f = E.PF / f"sys_{tag}.json"
    if f.exists(): return json.loads(f.read_text())["dets"]
    m = YOLO(str(w)); d = {}
    for im in imgs:
        r = m.predict(str(im), imgsz=sz, conf=0.001, rect=False, verbose=False, device=0)[0]
        d[im.name] = [[int(c), float(s), *b] for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxyn.tolist())]
    f.write_text(json.dumps({"_weights": str(w), "imgsz": sz, "dets": d})); return d


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--sizes", type=int, nargs="+", default=[640, 800])
    a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    W = {sz: train(sz, a.epochs) for sz in a.sizes}
    imgs = sorted((E.DS / "images/val_clean_v8").glob("*.jpg")); leak = set((E.DS / "leaked_val.txt").read_text().split())
    NL = [p.name for p in imgs if p.name not in leak]; idx = {n: i for i, n in enumerate(NL)}; G = E.gt(NL)
    D = {"n_v5@640": E.dets("nb22_pi_n_v5_640")} | {f"n_{sz}@{sz}": predict(w, imgs, sz, f"exp_student_n_{sz}") for sz, w in W.items()}
    rng = np.random.default_rng(20261016)
    Wb = [np.bincount(rng.integers(0, len(NL), len(NL)), minlength=len(NL)).astype(float) for _ in range(1000)]
    R, lines = {}, []
    for k, d in D.items():
        pre = [E.prep(d, G[c], c, set(NL), idx) for c in range(1, 6)]
        full = [E.wap(*x, np.ones(len(NL))) for x in pre]; R[k] = np.array([[E.wap(*x, w) for x in pre] for w in Wb])
        small = []
        for c in range(1, 6):   # image-level hit at conf 0.25 for images whose largest box of class c is small (< 1%)
            pos = [n for n in NL if len(G[c][n]) and max((b[2] - b[0]) * (b[3] - b[1]) for b in G[c][n]) < 0.01]
            small.append(np.mean([any(x[0] == c and x[1] >= 0.25 for x in d[n]) for n in pos]) if pos else float("nan"))
        lines.append(f"{k:10s} MEAN_5 {np.mean(full):.4f} | " + " ".join(f"{s} {v:.3f}" for s, v in zip(E.SHORT, full))
                     + " | bắt ảnh lỗi nhỏ @0.25: " + " ".join(f"{s} {v:.0%}" for s, v in zip(E.SHORT, small)))
    keys = list(D)
    for a_, b_ in [(k, "n_v5@640") for k in keys[1:]] + ([(keys[-1], keys[1])] if len(keys) > 2 else []):
        dd = R[a_] - R[b_]; m = dd.mean(1)
        lines.append(f"{a_} − {b_}: MEAN_5 {m.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}] P>0 {(m > 0).mean():.0%} | "
                     + " ".join(f"{s} {dd[:, i].mean():+.3f}" for i, s in enumerate(E.SHORT)))
    for l in lines: log(l)
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":   # Windows: DataLoader workers re-import this file
    raise SystemExit(main())
