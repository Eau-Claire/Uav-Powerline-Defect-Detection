"""FO review for dataset v6 (train). Builds numbered contact sheets for visual review:
  G = every foreign_object GT box in train; F = teacher v5 FO boxes (conf >= F_CONF) matching no FO GT (unlabelled FO?).
Each tile: context crop x3, green = GT, red = teacher box, big ID; text = teacher conf, COCO person conf over the box
(yolo26s COCO, IoU >= 0.3), SYN for v65 synthetic pastes. FO definition (README v2): a real foreign object; NOT a
person, electrical equipment, vegetation/vines, empty region, or a drawing.
Needs the teacher predictions from notebook v6p5-21 (artifacts/v6p5_v6_data/pred_train_640.jsonl).
Usage: python make_sheets.py REPO_DIR  ->  items.csv + sheets/{G,F}_NN.jpg; decisions go to decisions.csv."""
import json, sys, cv2, numpy as np, pandas as pd
from pathlib import Path
from ultralytics import YOLO
HERE = Path(__file__).resolve().parent; FO, F_CONF, TILE, COLS, ROWS = 4, 0.3, 200, 8, 6
def iou(a, b):
    wh = np.clip(np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0, None)
    i = wh[..., 0]*wh[..., 1]; ar = lambda x: (x[:, 2]-x[:, 0])*(x[:, 3]-x[:, 1])
    return i/(ar(a)[:, None]+ar(b)[None]-i+1e-9)
def tile(img, box, color, others, iid, text):
    H, W = img.shape[:2]; im = img.copy(); x1, y1, x2, y2 = box*[W, H, W, H]
    cx, cy, s = (x1+x2)/2, (y1+y2)/2, max(x2-x1, y2-y1, 0.04*max(W, H))*1.5
    X1, Y1, X2, Y2 = int(max(cx-s, 0)), int(max(cy-s, 0)), int(min(cx+s, W)), int(min(cy+s, H))
    for b, col in [*others, (box, color)]:
        b = b*[W, H, W, H]; cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, max(2, W//400))
    c = im[Y1:Y2, X1:X2]; f = TILE/max(c.shape[:2]); c = cv2.resize(c, (max(1, int(c.shape[1]*f)), max(1, int(c.shape[0]*f))))
    out = np.full((TILE+18, TILE, 3), 40, np.uint8); out[:c.shape[0], :c.shape[1]] = c
    cv2.rectangle(out, (0, 0), (62, 22), (0, 0, 0), -1); cv2.putText(out, iid, (2, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(out, text[:34], (2, TILE+13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out
if __name__ == "__main__":
    REPO = Path(sys.argv[1]); DS = REPO/'datasets/v6p5_v5'; IMG, LAB = DS/'images/train', DS/'labels/train'
    pred = {d['img']: np.array(d['boxes'], float).reshape(-1, 6) for d in map(json.loads, open(REPO/'artifacts/v6p5_v6_data/pred_train_640.jsonl', encoding='utf-8'))}
    items = []
    for name, P in pred.items():
        g = np.array([l.split() for l in (LAB/(Path(name).stem+'.txt')).read_text().splitlines() if l.strip()], float).reshape(-1, 5)
        gf = g[g[:, 0] == FO, 1:]; gf = np.c_[gf[:, :2]-gf[:, 2:]/2, gf[:, :2]+gf[:, 2:]/2]
        pf = P[P[:, 0] == FO]; pf = pf[np.argsort(-pf[:, 1])]
        U = iou(gf, pf[:, 2:]) if len(gf) and len(pf) else np.zeros((len(gf), len(pf)))
        for j, b in enumerate(gf):
            k = U[j].argmax() if len(pf) else -1; ok = k >= 0 and U[j, k] >= 0.5
            items.append(dict(group='G', img=name, box=b, teacher=float(pf[k, 1]) if ok else 0.0,
                              tbox=pf[k, 2:] if k >= 0 and U[j, k] > 0 else None))
        for i, (c, s, *b) in enumerate(pf):
            if s >= F_CONF and not (len(gf) and U[:, i].max() >= 0.5):
                items.append(dict(group='F', img=name, box=np.array(b), teacher=float(s), tbox=None))
    imgs = sorted({it['img'] for it in items}); person = {}
    coco = YOLO(str(REPO/'legacy/notebooks/v6p5/yolo26s.pt'))
    for k in range(0, len(imgs), 32):
        for n, r in zip(imgs[k:k+32], coco.predict([str(IMG/n) for n in imgs[k:k+32]], classes=[0], conf=0.25, rect=False, verbose=False)):
            person[n] = np.c_[r.boxes.conf.cpu().numpy(), r.boxes.xyxyn.cpu().numpy()]
    for it in items:
        p = person[it['img']]; u = iou(it['box'][None], p[:, 1:])[0] if len(p) else np.zeros(0)
        it['person'] = float(p[u >= 0.3, 0].max()) if (u >= 0.3).any() else 0.0
        it['syn'] = 'syn' in it['img']
    df = pd.DataFrame(items)
    df = pd.concat([df[df.group == 'G'].sort_values(['syn', 'person', 'teacher'], ascending=[True, False, True]),
                    df[df.group == 'F'].sort_values('teacher', ascending=False)])
    df['id'] = [f"{g}{i:03d}" for g, i in zip(df.group, df.groupby('group').cumcount() + 1)]
    (HERE/'sheets').mkdir(exist_ok=True); cache = {}
    for g in ['G', 'F']:
        sub = df[df.group == g]; per = COLS*ROWS
        for s in range(0, len(sub), per):
            tl = []
            for _, r in sub.iloc[s:s+per].iterrows():
                img = cache.get(r.img); img = img if img is not None else cv2.imread(str(IMG/r.img)); cache = {r.img: img}
                others = [(r.tbox, (0, 0, 255))] if r.tbox is not None else []
                txt = f"t{r.teacher:.2f}" + (f" P{r.person:.2f}" if r.person else "") + (" SYN" if r.syn else "") + f" {Path(r.img).stem[:12]}"
                tl.append(tile(img, r.box, (0, 200, 0) if g == 'G' else (0, 0, 255), others, r.id, txt))
            tl += [np.full_like(tl[0], 40)]*(-len(tl) % COLS)
            cv2.imwrite(str(HERE/f'sheets/{g}_{s//per + 1:02d}.jpg'), np.vstack([np.hstack(tl[i:i+COLS]) for i in range(0, len(tl), COLS)]),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
    df.assign(box=df.box.map(lambda b: ' '.join(f'{v:.5f}' for v in b))).drop(columns='tbox').to_csv(HERE/'items.csv', index=False)
    print(df.groupby('group').agg(n=('id', 'size'), syn=('syn', 'sum'), person_overlap=('person', lambda s: (s > 0).sum())))
