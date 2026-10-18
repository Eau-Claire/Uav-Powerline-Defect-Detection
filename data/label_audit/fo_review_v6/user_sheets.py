"""Sheets for the human check of decisions.csv: one titled block per decision that changes a label (person / bird_nest /
drop / add_fo), per rule-based keep (branch on line, hornet nest, rechecked) and a 24-box sample of the dropped synthetic
images. Usage: python user_sheets.py REPO_DIR -> sheets/review_NN.jpg"""
import sys, cv2, numpy as np, pandas as pd
from pathlib import Path
from make_sheets import HERE, TILE, COLS, tile
ORDER = [('person', 'FO -> PERSON (cong nhan)'), ('bird_nest', 'FO -> BIRD_NEST (to chim)'),
         ('drop:vegetation/vines', 'BO BOX: cay co / day leo moc tren cot, duoi dat'),
         ('drop:empty/equipment', 'BO BOX: trong / chi co thiet bi dien'), ('drop:ambiguous', 'BO BOX: khong ro la gi'),
         ('drop:group of people', 'BO BOX: ca nhom nguoi'), ('drop:drawing', 'BO BOX: hinh ve'),
         ('add_fo', 'THEM NHAN FO (teacher thay, chua co nhan)'),
         ('keep:branch on line', 'GIU FO: canh / day leo vat tren day, tren thiet bi (theo tra loi cua ban)'),
         ('keep:hornet nest', 'GIU FO: to ong vo ve (theo tra loi cua ban)'),
         ('keep:rechecked', 'GIU FO: o "khong chac" da xem lai o 400 px'),
         ('syn_image', 'LOAI CA ANH: 24 mau trong 161 anh tong hop')]
if __name__ == "__main__":
    IMG = Path(sys.argv[1])/'datasets/v6p5_v5/images/train'; d = pd.read_csv(HERE/'decisions.csv'); blocks = []
    for key, title in ORDER:
        sub = d[d.decision == key]
        if key == 'syn_image': sub = sub[sub.group == 'G'].sample(24, random_state=0)
        if not len(sub): continue
        tl = []
        for _, r in sub.iterrows():
            box = np.array([float(v) for v in r.box.split()])
            tl.append(tile(cv2.imread(str(IMG/r.img)), box, (0, 200, 0) if r.group == 'G' else (0, 0, 255), [], r.id,
                           f"t{r.teacher:.2f}" + (f" P{r.person:.2f}" if r.person else "")))
        tl += [np.full_like(tl[0], 40)]*(-len(tl) % COLS)
        head = np.full((34, TILE*COLS, 3), 15, np.uint8)
        cv2.putText(head, f"{title}  ({len(sub)})", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        blocks += [head] + [np.hstack(tl[i:i+COLS]) for i in range(0, len(tl), COLS)]
    page, n, h = [], 1, 0
    for b in blocks + [None]:
        if b is None or (h + b.shape[0] > 1700 and page):
            cv2.imwrite(str(HERE/f'sheets/review_{n:02d}.jpg'), np.vstack(page), [cv2.IMWRITE_JPEG_QUALITY, 88]); n += 1; page, h = [], 0
        if b is not None: page.append(b); h += b.shape[0]
    print(f"{n - 1} trang review")
