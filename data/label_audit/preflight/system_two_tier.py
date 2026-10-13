"""System-level two-tier eval on val_clean (CPU). Pi = student v5 NCNN at conf 0.01, flags an image when any defect
box (classes 1-5) has conf >= TAU; server = teacher v5 YOLO26m on flagged images, its boxes are the final answer;
unflagged images -> no detections. Reports per-class AP50 / AP50-95, MEAN_5 and the share of images sent to the
server for several TAU. Pi-only and teacher-only rows check the AP code against ultralytics (0.779 / 0.816)."""
import json, numpy as np, torch
from pathlib import Path
from ultralytics import YOLO
R = Path('/teamspace/studios/this_studio/datasets/v6p5_v5'); A = Path('/teamspace/studios/this_studio/Uav-Powerline-Defect-Detection/artifacts/v6p5_v5/runs')
HERE = Path(__file__).resolve().parent; NAMES = ['insulator_broken', 'insulator_flashover', 'bird_nest', 'foreign_object', 'broken_strand']
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
def run(model, imgs, cache, **kw):
    if cache.exists(): return json.load(open(cache))
    out = {}
    for k, im in enumerate(imgs):
        r = model.predict(str(im), imgsz=640, conf=0.001, verbose=False, **kw)[0]
        out[im.name] = [[int(c), float(s), *b] for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxyn.tolist())]
        if k % 200 == 0: print(cache.name, k, len(imgs), flush=True)
    json.dump(out, open(cache, 'w')); return out
if __name__ == "__main__":
    torch.set_num_threads(4)
    imgs = sorted(p for p in (R/'images/val_clean').iterdir() if p.suffix == '.jpg'); G = {c: {} for c in range(1, 6)}
    for im in imgs:
        a = np.array([l.split() for l in (R/'labels/val_clean'/(im.stem+'.txt')).read_text().splitlines() if l.strip()], float).reshape(-1, 5)
        for c in range(1, 6):
            b = a[a[:, 0] == c][:, 1:]; G[c][im.name] = np.c_[b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2]
    pi = run(YOLO(str(A/'student_distill/weights/best_ncnn_model'), task='detect'), imgs, HERE/'sys_pi.json')
    te = run(YOLO(str(A/'teacher/weights/best.pt')), imgs, HERE/'sys_teacher.json', device='cpu')
    def score(dets, keep):
        r = {}
        for c in range(1, 6):
            P = [(n, s, b) for n, d in dets.items() if keep(n) for cc, s, *b in d if cc == c]
            r[NAMES[c-1]] = (ap(P, G[c], 0.5), np.mean([ap(P, G[c], t) for t in np.arange(0.5, 0.96, 0.05)]))
        return r
    rows = [('Pi only', 1.0, score(pi, lambda n: True)), ('Teacher only (100% server)', 1.0, score(te, lambda n: True))]
    for tau in [0.01, 0.05, 0.1, 0.25, 0.4]:
        flag = {n for n, d in pi.items() if any(1 <= c <= 5 and s >= tau for c, s, *_ in d)}
        rows.append((f'Two-tier, Pi flag conf >= {tau}', len(flag)/len(imgs), score(te, lambda n, f=flag: n in f)))
    res = []
    for name, share, r in rows:
        m50 = np.mean([v[0] for v in r.values()]); m95 = np.mean([v[1] for v in r.values()])
        print(f"{name:32s} server {share:5.1%} | MEAN_5 {m50:.3f} / {m95:.3f} | " + ' '.join(f"{k[:9]} {v[0]:.3f}/{v[1]:.3f}" for k, v in r.items()), flush=True)
        res.append({'setting': name, 'server_share': share, 'MEAN_5_AP50': m50, 'MEAN_5_AP50_95': m95, **{f'{k}_AP50': v[0] for k, v in r.items()}, **{f'{k}_AP50_95': v[1] for k, v in r.items()}})
    import pandas as pd; pd.DataFrame(res).to_csv(HERE/'system_two_tier.csv', index=False)
