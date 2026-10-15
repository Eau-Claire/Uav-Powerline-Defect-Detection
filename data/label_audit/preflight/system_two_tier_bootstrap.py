"""Paired image bootstrap of two-tier MEAN_5 AP50 on val_clean (same models/caches as system_two_tier.py).
TP/FP of every box is computed once (greedy by conf within each image, as ap() does); each round only reweights images
by how often they were drawn, then the 101-point AP is recomputed with those weights (all-ones weights reproduce ap()).
Reports CI95 and P(MEAN_5 > 0.82) per server, plus the paired difference fo800 - teacher640.
Usage: python system_two_tier_bootstrap.py [same args as system_two_tier.py]. Writes system_two_tier_bootstrap.csv."""
import numpy as np
from system_two_tier import HERE, args, flagged, iou, load
B, TAU, SEED, TARGET = 2000, 0.05, 20261015, 0.82
def prep(dets, gts, c, keep, idx):  # -> image index, tp of every class-c box on kept images (conf-sorted), npos per image
    rows = []
    for n, d in dets.items():
        if n not in keep: continue
        g = gts.get(n, np.zeros((0, 4))); used = np.zeros(len(g), bool)
        for s, b in sorted([(s, b) for cc, s, *b in d if cc == c], key=lambda p: -p[0]):
            hit = 0
            if len(g):
                u = iou(np.array([b]), g)[0]; u[used] = -1; j = u.argmax()
                if u[j] >= 0.5: used[j] = True; hit = 1
            rows.append((idx[n], s, hit))
    rows.sort(key=lambda r: -r[1]); x = np.array(rows, float).reshape(-1, 3); npos = np.zeros(len(idx))
    for n, v in gts.items(): npos[idx[n]] = len(v)
    return x[:, 0].astype(int), x[:, 2], npos
def wap(img, tp, npos, w):  # weighted 101-point AP50
    N = (npos*w).sum()
    if N == 0: return 0.0
    ww = w[img]; c = np.cumsum(tp*ww); n = np.cumsum(ww); m = n > 0; rec, prec = c[m]/N, c[m]/n[m]
    mrec, mpre = np.r_[0, rec, 1], np.flip(np.maximum.accumulate(np.flip(np.r_[1, prec, 0])))
    x = np.linspace(0, 1, 101); return float(np.trapezoid(np.interp(x, mrec, mpre), x))
if __name__ == "__main__":
    a = args(); imgs, G, D, servers = load(a); idx = {p.name: i for i, p in enumerate(imgs)}; flag = flagged(D['pi'], TAU)
    pre = {(src, c): prep(D[src], G[c], c, flag, idx) for p in servers.values() for c, src in p.items()}
    score = lambda w: {s: np.mean([wap(*pre[(p[c], c)], w) for c in range(1, 6)]) for s, p in servers.items()}
    full = score(np.ones(len(imgs))); rng = np.random.default_rng(SEED); S = {s: [] for s in servers}
    for _ in range(B):
        for s, v in score(np.bincount(rng.integers(0, len(imgs), len(imgs)), minlength=len(imgs)).astype(float)).items(): S[s].append(v)
    S = {s: np.array(v) for s, v in S.items()}; S['fo800 - teacher640'] = S['fo800'] - S['teacher640']
    full['fo800 - teacher640'] = full['fo800'] - full['teacher640']; res = []
    for s, v in S.items():
        lo, hi = np.percentile(v, [2.5, 97.5]); diff = ' - ' in s; p = (v > 0).mean() if diff else (v > TARGET).mean()
        print(f"Two-tier @{TAU} {s:20s} MEAN_5 {full[s]:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]  P({'>0' if diff else f'>{TARGET}'}) {p:.1%}", flush=True)
        res.append({'setting': s, 'tau': TAU, 'MEAN_5_AP50': full[s], 'ci95_lo': lo, 'ci95_hi': hi, 'p_above': p, 'above': 0 if diff else TARGET, 'rounds': B})
    import pandas as pd; pd.DataFrame(res).to_csv(HERE/('system_two_tier_bootstrap.csv' if a.split == 'val_clean' else f'system_two_tier_bootstrap_{a.split}.csv'), index=False)
