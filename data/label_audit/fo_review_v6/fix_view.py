"""Full-image view for user-flagged boxes: every GT box (class colour + name), the reviewed box (thick yellow) and COCO
person proposals (yolo26s, conf >= 0.3, magenta, numbered). Usage: python fix_view.py REPO_DIR ID [ID ...] -> sheets/fix_*.jpg"""
import sys, cv2, numpy as np, pandas as pd
from pathlib import Path
from ultralytics import YOLO
HERE = Path(__file__).resolve().parent
NAMES = ['insulator', 'broken', 'flashover', 'bird_nest', 'FO', 'strand', 'person']
COL = [(160, 160, 160), (0, 140, 255), (255, 0, 255), (0, 200, 255), (0, 220, 0), (255, 200, 0), (255, 80, 80)]
if __name__ == "__main__":
    R = Path(sys.argv[1]); IMG, LAB = R/'datasets/v6p5_v5/images/train', R/'datasets/v6p5_v5/labels/train'
    d = pd.read_csv(HERE/'decisions.csv').set_index('id'); coco = YOLO(str(R/'legacy/notebooks/v6p5/yolo26s.pt')); tiles, props = [], []
    for iid in sys.argv[2:]:
        r = d.loc[iid]; im = cv2.imread(str(IMG/r.img)); H, W = im.shape[:2]; t = max(2, W//300)
        for l in (LAB/(Path(r.img).stem+'.txt')).read_text().splitlines():
            c, x, y, w, h = map(float, l.split()); c = int(c)
            p1, p2 = (int((x-w/2)*W), int((y-h/2)*H)), (int((x+w/2)*W), int((y+h/2)*H))
            cv2.rectangle(im, p1, p2, COL[c], t); cv2.putText(im, NAMES[c], (p1[0], max(p1[1]-4, 12)), 0, W/1400, COL[c], t)
        b = np.array([float(v) for v in r.box.split()])*[W, H, W, H]
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), t*3)
        res = coco.predict(str(IMG/r.img), classes=[0], conf=0.3, verbose=False)[0]
        for k, (s, pb) in enumerate(zip(res.boxes.conf.tolist(), res.boxes.xyxyn.tolist())):
            q = np.array(pb)*[W, H, W, H]; cv2.rectangle(im, (int(q[0]), int(q[1])), (int(q[2]), int(q[3])), (255, 0, 200), max(1, t//2))
            cv2.putText(im, f"p{k} {s:.2f}", (int(q[0]), int(q[3])+int(18*W/1000)), 0, W/1200, (255, 0, 200), t)
            props.append(dict(id=iid, img=r.img, p=k, conf=round(s, 3), box=' '.join(f'{v:.5f}' for v in pb)))
        f = 560/max(H, W); im = cv2.resize(im, (int(W*f), int(H*f))); out = np.full((600, 560, 3), 30, np.uint8)
        out[:im.shape[0], :im.shape[1]] = im; cv2.putText(out, f"{iid} ({r.decision})", (4, 592), 0, 0.6, (0, 255, 255), 2); tiles.append(out)
    tiles += [np.full_like(tiles[0], 30)]*(-len(tiles) % 4)
    for k in range(0, len(tiles), 8):
        cv2.imwrite(str(HERE/f'sheets/fix_{k//8+1}.jpg'), np.vstack([np.hstack(tiles[i:i+4]) for i in range(k, min(k+8, len(tiles)), 4)]))
    pd.DataFrame(props).to_csv(HERE/'fix_person_proposals.csv', index=False); print(len(props), 'person proposals')
