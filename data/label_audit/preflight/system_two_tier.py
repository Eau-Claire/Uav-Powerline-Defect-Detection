"""System-level two-tier eval on val_clean. Pi = student v5 (NCNN, or its .pt — identical at square letterbox), flags an
image when any defect box (classes 1-5) has conf >= TAU; server = teacher v5 YOLO26m on flagged images, its boxes are the
final answer; unflagged images -> no detections. Reports per-class AP50 / AP50-95, MEAN_5 and the share of images sent
to the server for several TAU. Pi-only and teacher-only rows check the AP code against ultralytics.

Every model runs with a square 640 letterbox (rect=False), as the Pi does: model.predict() defaults to rect=True, which
made the first version of this script score the teacher at 0.815 instead of 0.822 (two-tier 0.813 -> 0.820).
Server "fo800": foreign_object boxes come from the teacher at 800 (FO is small: 0.687 -> 0.718), the other classes
from 640 (flashover / strand drop at 800). Uncertainty: system_two_tier_bootstrap.py.

Usage: python system_two_tier.py [--data DIR] [--pi WEIGHTS] [--teacher WEIGHTS] [--device cpu|0]
Predictions are cached as sys_<tag>.json, tagged with the weights path (another path -> recompute); delete them
after retraining a model in place."""
import argparse, json, numpy as np, torch
from pathlib import Path
from ultralytics import YOLO
S = Path('/teamspace/studios/this_studio'); A = S/'Uav-Powerline-Defect-Detection/artifacts/v6p5_v5/runs'
HERE = Path(__file__).resolve().parent; NAMES = ['insulator_broken', 'insulator_flashover', 'bird_nest', 'foreign_object', 'broken_strand']
TAUS = [0.01, 0.05, 0.1, 0.25, 0.4]
def iou(a, b):
    wh = np.clip(np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0, None)
    i = wh[..., 0]*wh[..., 1]; ar = lambda x: (x[:, 2]-x[:, 0])*(x[:, 3]-x[:, 1])
    return i/(ar(a)[:, None]+ar(b)[None]-i+1e-9)
def ap(preds, gts, thr):  # preds [(img, conf, box)], gts {img: boxes}; 101-point interpolated AP (ultralytics style)
    npos = sum(len(g) for g in gts.values())
    if not npos: return 0.0
    used = {k: np.zeros(len(v), bool) for k, v in gts.items()}; tp = []
    for img, c, b in sorted(preds, key=lambda p: -p[1]):
        g = gts.get(img, np.zeros((0, 4))); hit = 0
        if len(g):
            u = iou(np.array([b]), g)[0]; u[used[img]] = -1; j = u.argmax()
            if u[j] >= thr: used[img][j] = True; hit = 1
        tp.append(hit)
    if not tp: return 0.0
    c = np.cumsum(tp); rec, prec = c/npos, c/np.arange(1, len(tp)+1)
    mrec, mpre = np.r_[0, rec, 1], np.flip(np.maximum.accumulate(np.flip(np.r_[1, prec, 0])))
    x = np.linspace(0, 1, 101); return float(np.trapezoid(np.interp(x, mrec, mpre), x))
def run(weights, imgs, tag, imgsz=640, device='cpu'):
    cache = HERE/f'sys_{tag}.json'
    if cache.exists():
        d = json.load(open(cache))
        if d.get('_weights') == str(weights): return d['dets']
        print(f'{cache.name}: cache from {d.get("_weights")} != {weights} -> recompute', flush=True)
    m = YOLO(str(weights), task='detect'); out = {}
    for k, im in enumerate(imgs):
        r = m.predict(str(im), imgsz=imgsz, conf=0.001, rect=False, verbose=False, device=device)[0]
        out[im.name] = [[int(c), float(s), *b] for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxyn.tolist())]
        if k % 200 == 0: print(cache.name, k, len(imgs), flush=True)
    json.dump({'_weights': str(weights), 'dets': out}, open(cache, 'w')); return out
def load(a):
    """-> imgs, G {class: {img: xyxyn}}, dets {pi, t640, t800}, servers {name: {class: dets}}"""
    R = Path(a.data); imgs = sorted(p for p in (R/'images/val_clean').iterdir() if p.suffix == '.jpg'); G = {c: {} for c in range(1, 6)}
    for im in imgs:
        x = np.array([l.split() for l in (R/'labels/val_clean'/(im.stem+'.txt')).read_text().splitlines() if l.strip()], float).reshape(-1, 5)
        for c in range(1, 6):
            b = x[x[:, 0] == c][:, 1:]; G[c][im.name] = np.c_[b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2]
    D = {'pi': run(a.pi, imgs, 'pi', device=a.device), 't640': run(a.teacher, imgs, 'teacher', device=a.device),
         't800': run(a.teacher, imgs, 'teacher800', imgsz=800, device=a.device)}
    servers = {'teacher640': {c: 't640' for c in range(1, 6)}, 'fo800': {c: 't800' if c == 4 else 't640' for c in range(1, 6)}}
    return imgs, G, D, servers
def flagged(pi, tau): return {n for n, d in pi.items() if any(1 <= c <= 5 and s >= tau for c, s, *_ in d)}
def args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(S/'datasets/v6p5_v5'))
    p.add_argument('--pi', default=str(A/'student_distill/weights/best_ncnn_model'))
    p.add_argument('--teacher', default=str(A/'teacher/weights/best.pt'))
    p.add_argument('--device', default='0' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()
if __name__ == "__main__":
    a = args(); torch.set_num_threads(4); imgs, G, D, servers = load(a)
    def score(pick, keep):  # pick {class: source in D}
        r = {}
        for c in range(1, 6):
            P = [(n, s, b) for n, d in D[pick[c]].items() if keep(n) for cc, s, *b in d if cc == c]
            r[NAMES[c-1]] = (ap(P, G[c], 0.5), np.mean([ap(P, G[c], t) for t in np.arange(0.5, 0.96, 0.05)]))
        return r
    rows = [('Pi only', 1.0, score({c: 'pi' for c in range(1, 6)}, lambda n: True))]
    rows += [(f'Server {s} only (100% server)', 1.0, score(p, lambda n: True)) for s, p in servers.items()]
    for s, p in servers.items():
        for tau in TAUS:
            flag = flagged(D['pi'], tau); rows.append((f'Two-tier {s}, Pi flag conf >= {tau}', len(flag)/len(imgs), score(p, lambda n, f=flag: n in f)))
    res = []
    for name, share, r in rows:
        m50 = np.mean([v[0] for v in r.values()]); m95 = np.mean([v[1] for v in r.values()])
        print(f"{name:44s} server {share:5.1%} | MEAN_5 {m50:.4f} / {m95:.4f} | " + ' '.join(f"{k[:9]} {v[0]:.3f}/{v[1]:.3f}" for k, v in r.items()), flush=True)
        res.append({'setting': name, 'server_share': share, 'MEAN_5_AP50': m50, 'MEAN_5_AP50_95': m95, **{f'{k}_AP50': v[0] for k, v in r.items()}, **{f'{k}_AP50_95': v[1] for k, v in r.items()}})
    import pandas as pd; pd.DataFrame(res).to_csv(HERE/'system_two_tier.csv', index=False)
