"""Fuse Grounding DINO + OWLv2 insulator proposals -> accepted boxes per image (fused_{split}.jsonl).
Rule: GDINO box (score >= G_MIN) survives NMS, is not a group box (contains >= 2 other GDINO boxes), is not near the
whole image, does not coincide with a bird_nest / FO / person GT box, and is confirmed by OWLv2 (IoU >= 0.5,
score >= O_MIN) or has GDINO score >= G_HIGH. Usage: python fuse.py [G_MIN O_MIN G_HIGH] [--bench] [--sheet N] [--auto]
--auto: grid-search the thresholds on the benchmark (max recall with precision >= 0.85), write insulator_add.csv."""
import json, sys, random
import numpy as np
from pathlib import Path
R = Path('/teamspace/studios/this_studio/datasets/v6p5_v4')
args = [a for a in sys.argv[1:] if not a.startswith('--')]
G_MIN, O_MIN, G_HIGH = (map(float, args[:3]) if len(args) >= 3 else (0.3, 0.15, 0.45))
P_TARGET = 0.85
def iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    wh = np.clip(np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0, None)
    i = wh[..., 0]*wh[..., 1]; ar = lambda x: (x[:, 2]-x[:, 0])*(x[:, 3]-x[:, 1])
    return i/(ar(a)[:, None]+ar(b)[None]-i+1e-9), i/(ar(b)[None]+1e-9)  # IoU, intersection over b's area
from functools import lru_cache
def gt(split, image, classes):
    return _gt(split, image, tuple(classes))
@lru_cache(maxsize=None)
def _gt(split, image, classes):
    L = np.array([l.split() for l in (R/'labels'/split/(Path(image).stem+'.txt')).read_text().splitlines() if l.strip()], float).reshape(-1, 5)
    L = L[np.isin(L[:, 0], classes)]
    return np.c_[L[:, 1:3]-L[:, 3:]/2, L[:, 1:3]+L[:, 3:]/2]
def fuse(r, split):
    g = np.array(r['gdino'], float).reshape(-1, 4); gs = np.array(r['gdino_s'])
    o = np.array(r['owl'], float).reshape(-1, 4); os_ = np.array(r['owl_s'])
    k = gs >= G_MIN; g, gs = g[k], gs[k]
    order = np.argsort(-gs); g, gs = g[order], gs[order]; keep = []
    for i in range(len(g)):  # NMS
        if all(iou(g[i], g[j])[0][0, 0] < 0.6 for j in keep): keep.append(i)
    g, gs = g[keep], gs[keep]
    if len(g):
        _, ioa = iou(g, g)  # ioa[i, j] = part of box j inside box i
        np.fill_diagonal(ioa, 0)
        area = (g[:, 2]-g[:, 0])*(g[:, 3]-g[:, 1])
        ok = ((ioa >= 0.8).sum(1) < 2) & (area < 0.85)
        g, gs = g[ok], gs[ok]
    if len(g):
        other = gt(split, r['image'], [3, 4, 6])  # nest, FO, person
        if len(other): k = iou(g, other)[0].max(1) < 0.5; g, gs = g[k], gs[k]
    if len(g):
        oo = o[os_ >= O_MIN]
        conf = iou(g, oo)[0].max(1) >= 0.5 if len(oo) else np.zeros(len(g), bool)
        k = conf | (gs >= G_HIGH); g, gs = g[k], gs[k]
    if len(g) > 1:  # nested duplicates: drop a box that wraps a smaller accepted box (same insulator + bracket)
        _, ioa = iou(g, g); np.fill_diagonal(ioa, 0)
        area = (g[:, 2]-g[:, 0])*(g[:, 3]-g[:, 1])
        wraps = ((ioa >= 0.8) & (area[:, None] >= 1.5*area[None, :])).any(1)
        g, gs = g[~wraps], gs[~wraps]
    return g, gs
def bench(rows):
    tp = fp = npos = 0
    for r in rows:
        if not r['benchmark']: continue
        g, _ = fuse(r, 'train'); G = gt('train', r['image'], [0]); npos += len(G)
        if len(g) and len(G):
            hit = iou(g, G)[0].max(1) >= 0.5; tp += int(hit.sum()); fp += int((~hit).sum())
        else: fp += len(g)
    return tp/max(tp+fp, 1), tp/max(npos, 1)
if __name__ == "__main__" and '--auto' in sys.argv:
    import csv, itertools
    rows = {s: [json.loads(l) for l in open(f'pred_{s}.jsonl')] for s in ('train', 'val')}
    res = []
    for G_MIN, O_MIN, G_HIGH in itertools.product([0.25, 0.3, 0.35, 0.4], [0.1, 0.15, 0.2, 0.3], [0.4, 0.45, 0.5, 0.6, 1.1]):
        p, r = bench(rows['train']); res.append((p, r, G_MIN, O_MIN, G_HIGH))
    ok = [x for x in res if x[0] >= P_TARGET]
    p, r, G_MIN, O_MIN, G_HIGH = max(ok, key=lambda x: (x[1], x[0])) if ok else max(res)
    print(f'chosen G_MIN {G_MIN} O_MIN {O_MIN} G_HIGH {G_HIGH} -> benchmark P {p:.3f} R {r:.3f} ({len(ok)}/{len(res)} settings reach P>={P_TARGET})')
    n_img = n_box = 0
    safe = p >= 0.75  # below this the proposals would add more noise than they fix: add nothing
    if not safe: print(f'benchmark precision {p:.3f} < 0.75 -> no insulator boxes added')
    with open('insulator_add.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['split', 'image', 'x', 'y', 'w', 'h', 'gdino_score'])
        for split, out_split in ((('train', 'train'), ('val', 'val_clean')) if safe else ()):
            for r_ in rows[split]:
                if r_['benchmark']: continue
                g, gs = fuse(r_, split)
                n_img += bool(len(g)); n_box += len(g)
                for b, sc in zip(g, gs):
                    w.writerow([out_split, r_['image'], round((b[0]+b[2])/2, 6), round((b[1]+b[3])/2, 6), round(b[2]-b[0], 6), round(b[3]-b[1], 6), round(float(sc), 4)])
    json.dump({'G_MIN': G_MIN, 'O_MIN': O_MIN, 'G_HIGH': G_HIGH, 'bench_P': p, 'bench_R': r, 'applied': safe, 'images': n_img, 'boxes': n_box,
               'grid': [list(x) for x in res]}, open('fuse_choice.json', 'w'), indent=1)
    print(f'insulator_add.csv: {n_box} boxes on {n_img} images')
    sys.exit(0)
if __name__ == "__main__":
    out = {}
    for split in ('train', 'val'):
        f = Path(f'pred_{split}.jsonl')
        if not f.exists(): continue
        out[split] = [(r, *fuse(r, split)) for r in map(json.loads, f.open())]
    if '--bench' in sys.argv:  # precision / recall vs GT on images that have insulator labels
        tp = fp = npos = 0
        for r, g, _ in out['train']:
            if not r['benchmark']: continue
            G = gt('train', r['image'], [0]); npos += len(G)
            if len(g) and len(G):
                m = iou(g, G)[0]; hit = m.max(1) >= 0.5; tp += int(hit.sum()); fp += int((~hit).sum())
            else: fp += len(g)
        print(f'G_MIN {G_MIN} O_MIN {O_MIN} G_HIGH {G_HIGH} | benchmark P {tp/max(tp+fp,1):.3f} R {tp/max(npos,1):.3f} (GT {npos})')
    if '--sheet' in sys.argv:
        from PIL import Image, ImageDraw
        n = int(sys.argv[sys.argv.index('--sheet')+1]); random.seed(n)
        rows = random.sample([x for x in out['train'] if not x[0]['benchmark']], 24)
        S = Image.new('RGB', (6*260, 4*260), 'black')
        for k, (r, g, gs) in enumerate(rows):
            im = Image.open(R/'images/train'/r['image']).convert('RGB'); W, H = im.size; d = ImageDraw.Draw(im); lw = max(2, W//150)
            for b, s in zip(g, gs): d.rectangle([b[0]*W, b[1]*H, b[2]*W, b[3]*H], outline='lime', width=lw); d.text((b[0]*W+3, b[1]*H+3), f'{s:.2f}', fill='lime')
            im.thumbnail((255, 255)); ImageDraw.Draw(im).text((3, 3), str(k), fill='yellow'); S.paste(im, ((k % 6)*260, (k//6)*260))
        S.save(f'sheet_{n}.jpg', quality=85); print('sheet', n)
    for s, v in out.items():
        nb = [x for x in v if not x[0]['benchmark']]
        print(f'{s}: {len(nb)} images without insulator label -> {sum(len(g) > 0 for _, g, _ in nb)} get boxes, {sum(len(g) for _, g, _ in nb)} boxes')
