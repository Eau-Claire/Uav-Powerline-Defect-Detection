"""Two-tier FO with the teacher as the server-side verifier. The Pi (student NCNN, conf >= 0.01) flags images;
the server runs teacher v5 (YOLO26m) on the flagged images only. A: teacher re-scores the Pi boxes,
B: sqrt(Pi conf x teacher conf), C: the server's own FO boxes on flagged images. CPU only."""
import json, numpy as np, torch
from pathlib import Path
from ultralytics import YOLO
import fo_verify as F
T = '/teamspace/studios/this_studio/Uav-Powerline-Defect-Detection/artifacts/v6p5_v5/runs/teacher/weights/best.pt'
if __name__ == "__main__":
    torch.set_num_threads(4)
    preds = json.load(open(F.HERE/'fo_verify_preds.json'))
    imgs = sorted(p for p in (F.R/'images/val_clean').iterdir() if p.suffix == '.jpg'); gts = {}
    for im in imgs:
        a = np.array([l.split() for l in (F.R/'labels/val_clean'/(im.stem+'.txt')).read_text().splitlines() if l.strip()], float).reshape(-1, 5)
        a = a[a[:, 0] == F.FO][:, 1:]; gts[im.name] = np.c_[a[:, :2]-a[:, 2:]/2, a[:, :2]+a[:, 2:]/2]
    flagged = sorted({p['image'] for p in preds}); m = YOLO(T); tb = {}
    for k, n in enumerate(flagged):
        r = m.predict(str(F.R/'images/val_clean'/n), imgsz=640, conf=0.001, classes=[F.FO], device='cpu', verbose=False)[0]
        tb[n] = (r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy())
    for p in preds:
        b, c = tb[p['image']]
        p['t'] = float(c[F.iou(np.array([p['box']]), b)[0] >= 0.5].max()) if len(b) and (F.iou(np.array([p['box']]), b)[0] >= 0.5).any() else 0.0
    server = [(n, float(c), list(b)) for n in flagged for b, c in zip(*tb[n])]
    print(f'FO GT {sum(len(g) for g in gts.values())} | Pi flags {len(flagged)} of {len(imgs)} images ({len(preds)} boxes) -> server runs the teacher on {len(flagged)/len(imgs):.0%} of the images')
    for name, P in [('Pi only (conf)', [(p['image'], p['conf'], p['box']) for p in preds]),
                    ('A: teacher re-scores Pi boxes', [(p['image'], p['t'], p['box']) for p in preds if p['t'] > 0]),
                    ('B: sqrt(Pi x teacher)', [(p['image'], np.sqrt(p['conf']*p['t']), p['box']) for p in preds if p['t'] > 0]),
                    ('C: teacher boxes on flagged images', server)]:
        ap, rmax, pr = F.ap_pr(P, gts)
        print(f'{name:36s} AP50 {ap:.3f} | max recall {rmax:.3f} | precision @ recall 0.5/0.6/0.7/0.8: {pr}', flush=True)
