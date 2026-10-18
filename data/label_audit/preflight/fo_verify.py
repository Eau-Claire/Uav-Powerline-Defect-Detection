"""Idea 4 (two-tier): the Pi model (student v5 NCNN) proposes foreign_object at a low confidence, a server-side
zero-shot model (CLIP ViT-B/32 on the padded crop) re-scores each box. Compares FO AP50 / precision-recall on
val_clean before and after verification. CPU only. Writes fo_verify_preds.json + prints a summary."""
import json, sys
import numpy as np, torch, clip
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
R = Path('/teamspace/studios/this_studio/datasets/v6p5_v5')
NCNN = '/teamspace/studios/this_studio/Uav-Powerline-Defect-Detection/artifacts/v6p5_v5/runs/student_distill/weights/best_ncnn_model'
FO, HERE = 4, Path(__file__).resolve().parent
POS = ['a balloon', 'a kite', 'a plastic bag', 'a piece of cloth', 'a banner or flag', 'a toy', 'trash hanging on a wire',
       'a foreign object hanging on a power line']
NEG = ['a person', 'a worker', 'a porcelain insulator', 'a glass insulator', 'a power line cable', 'a steel lattice tower',
       'a concrete pole', 'a bird nest', 'a bird', 'tree branches and leaves', 'vines on a pole', 'the sky', 'a building',
       'a road', 'grass', 'electrical equipment', 'a warning sphere marker on a power line']
def iou(a, b):
    wh = np.clip(np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0, None)
    i = wh[..., 0]*wh[..., 1]; ar = lambda x: (x[:, 2]-x[:, 0])*(x[:, 3]-x[:, 1])
    return i/(ar(a)[:, None]+ar(b)[None]-i+1e-9)
def ap_pr(preds, gts):
    npos = sum(len(g) for g in gts.values()); used = {k: np.zeros(len(v), bool) for k, v in gts.items()}; tp = []
    for img, s, b in sorted(preds, key=lambda p: -p[1]):
        g = gts[img]; hit = 0
        if len(g):
            u = iou(np.array([b]), g)[0]; j = u.argmax()
            if u[j] >= 0.5 and not used[img][j]: used[img][j] = True; hit = 1
        tp.append(hit)
    tp = np.array(tp); c = np.cumsum(tp); rec = c/npos; prec = c/np.arange(1, len(tp)+1)
    mrec, mpre = np.r_[0, rec, 1], np.flip(np.maximum.accumulate(np.flip(np.r_[1, prec, 0])))
    x = np.linspace(0, 1, 101)
    def p_at(r): k = np.where(rec >= r)[0]; return float(prec[k].max()) if len(k) else 0.0
    return float(np.trapezoid(np.interp(x, mrec, mpre), x)), float(rec.max()), {r: round(p_at(r), 3) for r in (0.5, 0.6, 0.7, 0.8)}
if __name__ == "__main__":
    torch.set_num_threads(4)
    cache = HERE/'fo_verify_preds.json'
    imgs = sorted(p for p in (R/'images/val_clean').iterdir() if p.suffix == '.jpg')
    gts = {}
    for im in imgs:
        a = np.array([l.split() for l in (R/'labels/val_clean'/(im.stem+'.txt')).read_text().splitlines() if l.strip()], float).reshape(-1, 5)
        a = a[a[:, 0] == FO][:, 1:]; gts[im.name] = np.c_[a[:, :2]-a[:, 2:]/2, a[:, :2]+a[:, 2:]/2]
    if cache.exists():
        preds = json.load(open(cache))
    else:
        det = YOLO(NCNN, task='detect'); preds = []
        for k, im in enumerate(imgs):
            r = det.predict(str(im), imgsz=640, conf=0.01, classes=[FO], verbose=False)[0]
            for b, c in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist()): preds.append({'image': im.name, 'box': b, 'conf': c})
            if k % 300 == 0: print('detect', k, len(imgs), flush=True)
        model, prep = clip.load('ViT-B/32', device='cpu')
        with torch.no_grad():
            t = model.encode_text(clip.tokenize([f'a photo of {p}' for p in POS + NEG])); t /= t.norm(dim=-1, keepdim=True)
            for k, p in enumerate(preds):
                im = Image.open(R/'images/val_clean'/p['image']).convert('RGB'); W, H = im.size
                x0, y0, x1, y1 = p['box']; pw, ph = 0.15*(x1-x0), 0.15*(y1-y0)
                crop = im.crop((max(0, x0-pw)*W, max(0, y0-ph)*H, min(1, x1+pw)*W, min(1, y1+ph)*H))
                f = model.encode_image(prep(crop)[None]); f /= f.norm(dim=-1, keepdim=True)
                pr = (100*f@t.T).softmax(-1)[0]
                p['p_fo'] = float(pr[:len(POS)].sum()); p['p_person'] = float(pr[len(POS):len(POS)+2].sum())
                if k % 500 == 0: print('clip', k, len(preds), flush=True)
        json.dump(preds, open(cache, 'w'))
    print(f'FO GT {sum(len(g) for g in gts.values())} | candidates {len(preds)} (conf >= 0.01)')
    rows = []
    for name, score, minconf in [('Pi only (conf)', lambda p: p['conf'], 0.01),
                                 ('Pi conf x CLIP p_fo', lambda p: p['conf']*p['p_fo'], 0.01),
                                 ('Pi conf x sqrt(CLIP)', lambda p: p['conf']*np.sqrt(p['p_fo']), 0.01),
                                 ('Pi conf>=0.05, drop CLIP person', lambda p: p['conf'] if p['p_person'] < 0.5 else 0, 0.05)]:
        P = [(p['image'], score(p), p['box']) for p in preds if p['conf'] >= minconf]
        ap, rmax, pr = ap_pr([x for x in P if x[1] > 0], gts)
        print(f'{name:34s} AP50 {ap:.3f} | max recall {rmax:.3f} | precision @ recall 0.5/0.6/0.7/0.8: {pr}')
