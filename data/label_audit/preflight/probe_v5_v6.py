"""Cheap 'is v6 worth training' probe on a small GPU: fine-tune student n v5 for a few epochs on v5 and on v6 with the
same seed / schedule (low lr, no warmup: a converged model is only nudged), then compare both on val_clean_v6
(square letterbox, all images = Pi-only setting) with a paired image bootstrap. A weak signal (short run, small model):
it can show the direction of the label change (FO precision / AP), not the final gain.
Usage: python probe_v5_v6.py REPO_DIR [EPOCHS]  ->  probe_v5_v6.log / probe_v5_v6.csv (+ runs in artifacts/probe_v5_v6/)"""
import sys, time, json, shutil, torch, numpy as np, pandas as pd
from pathlib import Path
from ultralytics import YOLO
HERE = Path(__file__).resolve().parent
from system_two_tier import NAMES, ap, run
from system_two_tier_bootstrap import prep, wap
B, SEED = 2000, 20261015
def log(m):
    print(m, flush=True)
    with open(HERE/'probe_v5_v6.log', 'a', encoding='utf-8') as f: f.write(f"{time.strftime('%H:%M:%S')} {m}\n")
def main():
    R = Path(sys.argv[1]); EP = int(sys.argv[2]) if len(sys.argv) > 2 else 6; OUT = R/'artifacts/probe_v5_v6'
    init = R/'artifacts/v6p5_v5_students/init/student_n_v5.pt'; best = {}
    for tag in ['v5', 'v6']:
        w = OUT/tag/'weights/last.pt'
        done = w.exists() and torch.load(w, map_location='cpu', weights_only=False).get('epoch') == -1  # -1 = run finished
        if not done:
            shutil.rmtree(OUT/tag, ignore_errors=True)  # an interrupted run restarts from scratch (same seed)
            m = YOLO(str(init)); t0 = time.time()
            m.add_callback('on_fit_epoch_end', lambda tr: log(f"[{tag}] epoch {tr.epoch + 1}/{tr.epochs} | "
                           f"{(time.time() - t0) / 60:.1f} min"))
            m.train(data=str(R/f'datasets/v6p5_{tag}/data.yaml'), epochs=EP, batch=16, imgsz=640, lr0=1e-4, lrf=0.1,
                    cos_lr=True, warmup_epochs=0, optimizer='AdamW', close_mosaic=2, seed=SEED, deterministic=True,
                    cache='disk', workers=4, amp=True, val=False, plots=False, project=str(OUT), name=tag, exist_ok=True)
        best[tag] = w; log(f"[{tag}] trained -> {w}")
    D = R/'datasets/v6p5_v6'; imgs = sorted((D/'images/val_clean_v6').glob('*.jpg')); idx = {p.name: i for i, p in enumerate(imgs)}
    G = {c: {} for c in range(1, 6)}
    for im in imgs:
        x = np.array([l.split() for l in (D/'labels/val_clean_v6'/(im.stem + '.txt')).read_text().splitlines() if l.strip()], float).reshape(-1, 5)
        for c in range(1, 6):
            b = x[x[:, 0] == c][:, 1:]; G[c][im.name] = np.c_[b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2]
    P = {t: run(w, imgs, f'probe_{t}', device=0) for t, w in best.items()}
    P['v5_init'] = run(init, imgs, 'probe_v5_init', device=0)          # reference: the starting point
    keep = set(idx); pre = {(t, c): prep(P[t], G[c], c, keep, idx) for t in P for c in range(1, 6)}
    cls = lambda t, w: [wap(*pre[(t, c)], w) for c in range(1, 6)]
    full = {t: cls(t, np.ones(len(imgs))) for t in P}; rng = np.random.default_rng(SEED); S = {t: [] for t in P}
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(imgs), len(imgs)), minlength=len(imgs)).astype(float)
        for t in P: S[t].append(cls(t, w))
    S = {t: np.array(v) for t, v in S.items()}; rows = []
    for t in P:
        rows.append({'model': t, 'MEAN_5': np.mean(full[t]), **{NAMES[c]: full[t][c] for c in range(5)}})
        log(f"{t:8s} MEAN_5 {np.mean(full[t]):.4f} | " + ' '.join(f"{NAMES[c][:9]} {full[t][c]:.3f}" for c in range(5)))
    for name, k in [('MEAN_5', None), ('foreign_object', 3), ('insulator_flashover', 1)]:
        d = (S['v6'].mean(1) - S['v5'].mean(1)) if k is None else (S['v6'][:, k] - S['v5'][:, k])
        lo, hi = np.percentile(d, [2.5, 97.5]); f = (np.mean(full['v6']) - np.mean(full['v5'])) if k is None else full['v6'][k] - full['v5'][k]
        log(f"v6 - v5 {name:20s} {f:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]  P(>0) {(d > 0).mean():.1%}")
        rows.append({'model': f'v6-v5 {name}', 'MEAN_5': f, 'ci95_lo': lo, 'ci95_hi': hi, 'p_gt0': (d > 0).mean()})
    pd.DataFrame(rows).to_csv(HERE/'probe_v5_v6.csv', index=False)
if __name__ == "__main__":  # Windows: DataLoader workers re-import this file
    main()
