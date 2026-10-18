"""Upper bound of the person class for foreign_object: teacher v2 FO predictions on v4 val_clean, AP50 with and
without the FO boxes that sit on a COCO person (conf>=0.8, intersection/pred-area >= 0.5)."""
import json, numpy as np
from pathlib import Path
from ultralytics import YOLO
R = Path('/teamspace/studios/this_studio/datasets/v6p5_v4')
A = Path('/teamspace/studios/this_studio/Uav-Powerline-Defect-Detection')
def xyxy(b): b = np.asarray(b, float).reshape(-1, 4); return np.c_[b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2]
def inter(a, b):
    wh = np.clip(np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0, None)
    return wh[..., 0]*wh[..., 1]
def area(a): return (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1])
def ap50(preds, gts):
    npos = sum(len(g) for g in gts.values()); used = {k: np.zeros(len(v), bool) for k, v in gts.items()}; tp = []
    for img, c, b in sorted(preds, key=lambda p: -p[1]):
        g = gts[img]; hit = 0
        if len(g):
            i = inter(b[None], g)[0]; iou = i/(area(b[None])+area(g)-i); j = iou.argmax()
            if iou[j] >= 0.5 and not used[img][j]: used[img][j] = True; hit = 1
        tp.append(hit)
    ctp = np.cumsum(tp); rec = ctp/npos; prec = ctp/np.arange(1, len(tp)+1)
    mrec, mpre = np.r_[0, rec, 1], np.flip(np.maximum.accumulate(np.flip(np.r_[1, prec, 0])))
    x = np.linspace(0, 1, 101); return float(np.trapezoid(np.interp(x, mrec, mpre), x)), npos
if __name__ == "__main__":
    PER = {r['image']: r for r in map(json.loads, open(A/'data/label_audit/person_all/pred_val.jsonl'))}
    m = YOLO(str(A/'artifacts/v6p5_v2/runs/teacher/weights/best.pt'))  # 5 classes, FO = 3
    gts, preds = {}, []
    imgs = sorted(p for p in (R/'images/val_clean').iterdir() if p.suffix == '.jpg')
    for im in imgs:
        g = np.array([l.split() for l in (R/'labels/val_clean'/(im.stem+'.txt')).read_text().splitlines() if l.strip()], float).reshape(-1, 5)
        gts[im.name] = xyxy(g[g[:, 0] == 4][:, 1:])
    for im in imgs:  # one image per call: a list input is treated as one batch (OOM)
        r = m.predict(str(im), imgsz=640, conf=0.001, classes=[3], device=0, verbose=False)[0]; n = im.name
        for b, c in zip(r.boxes.xywhn.cpu().numpy(), r.boxes.conf.cpu().numpy()): preds.append((n, float(c), xyxy(b)[0]))
    base, npos = ap50(preds, gts)
    kept = []
    for img, c, b in preds:
        pr = PER.get(img); pb = xyxy([x for x, f in zip(pr['xywhn'], pr['conf']) if f >= .8]) if pr else np.zeros((0, 4))
        if len(pb) and (inter(b[None], pb)[0]/area(b[None])).max() >= 0.5: continue
        kept.append((img, c, b))
    filt, _ = ap50(kept, gts)
    print(f'FO GT {npos} | AP50 teacher v2 {base:.3f} -> without FO-on-person {filt:.3f} (+{filt-base:.3f}) | removed {len(preds)-len(kept)} of {len(preds)} preds')
