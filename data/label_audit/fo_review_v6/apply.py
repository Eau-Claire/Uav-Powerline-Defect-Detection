"""Apply decisions.csv to YOLO label lines of one train image (used by scripts/label_audit_build.py and for previews).
G rows are matched to their FO line by the box in items.csv (the box as labelled); person / bird_nest change the class
(and take the box from decisions.csv, which may be corrected), drop:* removes the line, keep* leaves it; add_* rows append
a new line; syn_image drops the whole image (returns None)."""
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
CLS = {'person': 6, 'bird_nest': 3, 'add_fo': 4, 'add_person': 6, 'add_bird_nest': 3}
def load():
    d = pd.read_csv(HERE/'decisions.csv'); orig = pd.read_csv(HERE/'items.csv').set_index('id').box
    d['orig'] = d.id.map(orig); return {k: g for k, g in d.groupby('img')}
def xyxy(s): return np.array([float(v) for v in s.split()])
def apply(lines, rows):
    """lines: label lines ('c x y w h'); rows: decisions of this image -> (new lines or None, list of changes)."""
    if rows is None: return lines, []
    if (rows.decision == 'syn_image').any(): return None, ['syn_image']
    lab = [l.split() for l in lines if l.strip()]; out, used, log = [], set(), []
    box = np.array([[float(v) for v in l[1:]] for l in lab]).reshape(-1, 4)
    box = np.c_[box[:, :2] - box[:, 2:]/2, box[:, :2] + box[:, 2:]/2]
    act = {}
    for r in rows[rows.group == 'G'].itertuples():
        o = xyxy(r.orig); j = [i for i, l in enumerate(lab) if l[0] == '4' and i not in act and np.abs(box[i] - o).max() < 2e-4]
        assert len(j) == 1, (r.id, r.img, len(j)); act[j[0]] = r
    for i, l in enumerate(lab):
        r = act.get(i)
        if r is None or r.decision.startswith('keep'): out.append(' '.join(l)); continue
        if r.decision.startswith('drop'): log.append(f'{r.id} {r.decision}'); continue
        b = xyxy(r.box); out.append(f"{CLS[r.decision]} {(b[0]+b[2])/2:.6f} {(b[1]+b[3])/2:.6f} {b[2]-b[0]:.6f} {b[3]-b[1]:.6f}")
        log.append(f'{r.id} FO->{r.decision}')
    for r in rows[rows.decision.str.startswith('add_')].itertuples():
        b = xyxy(r.box); out.append(f"{CLS[r.decision]} {(b[0]+b[2])/2:.6f} {(b[1]+b[3])/2:.6f} {b[2]-b[0]:.6f} {b[3]-b[1]:.6f}")
        log.append(f'{r.id} {r.decision}')
    return out, log
