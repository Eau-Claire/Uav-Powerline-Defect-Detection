#!/usr/bin/env python
"""Build dataset v6p5_v7 from v6p5_v6: automatic box-consistency fixes on TRAIN + a short human review queue.

Why: the mAP50 -> mAP50-95 gap comes from inconsistent box extents (scripts/audit_v6_boxes.py): one box per broken
shed vs one box per crack, loose / tight boxes per source, ~900 visible insulators never labelled. The v5 teacher
(trained on all of it) predicts the AVERAGE convention, so where it is confident and clearly matches one label,
its box is the consistent one. Rules (train only, teacher v5 @640 from artifacts/audit_v6_boxes/pred_train.json):

  R1 refine   GT (any class) <-> teacher box of the same class, conf >= 0.8, IoU in [0.5, 0.75), mutual best match
              (clearly-off boxes only; boxes already within 0.75 are left as the human drew them)
              -> GT box := teacher box
  R2 merge    >= 2 GT boxes of class 1/2 (broken / flashover) each >= 80% inside ONE teacher box of that class
              (conf >= 0.6) -> replaced by that teacher box   ("one box per damaged shed")
  R3 add_ins  teacher insulator conf >= 0.75 with no GT insulator IoU >= 0.3 -> add insulator box
Not automated (-> review queue, prioritised, capped): defect boxes the teacher never finds (missed_gt), confident
teacher defects with no label (missing_label, minus teacher FO-on-person errors), class conflicts (wrong_class).

`val`, `val_clean`, `val_clean_v6` are copied UNCHANGED: refining val with the teacher would inflate the score.
val_clean_v6 problems go to the review queue instead; a reviewed val_v7 is a separate, later step.
Images and .npy caches are hard-linked; v6 is never modified.

  .venv/bin/python scripts/build_v7.py      -> datasets/v6p5_v7, artifacts/review_v7/
"""
import json, os, shutil
from collections import Counter
from pathlib import Path

import numpy as np, pandas as pd, yaml
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
V6, V7 = REPO / "datasets/v6p5_v6", REPO / "datasets/v6p5_v7"
AUD = REPO / "artifacts/audit_v6_boxes"; REV = REPO / "artifacts/review_v7"
NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand", "person"]
COL = [(0, 200, 255), (255, 128, 0), (255, 0, 255), (0, 200, 0), (255, 255, 0), (255, 0, 0), (128, 0, 255)]


def iou(a, b):
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    wh = np.clip(np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0, None)
    i = wh[..., 0] * wh[..., 1]; ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return i / (ar(a)[:, None] + ar(b)[None] - i + 1e-9)


def ioa(a, b):  # share of each a inside each b
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    wh = np.clip(np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0, None)
    return wh[..., 0] * wh[..., 1] / ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + 1e-9)[:, None]


def read(p):
    g = np.array([l.split() for l in p.read_text().splitlines() if l.strip()], float).reshape(-1, 5)
    return g[:, 0].astype(int), np.c_[g[:, 1:3] - g[:, 3:5] / 2, g[:, 1:3] + g[:, 3:5] / 2]


def write(p, cls, box):
    box = np.clip(box, 0, 1); c = (box[:, :2] + box[:, 2:]) / 2; wh = box[:, 2:] - box[:, :2]
    p.write_text("".join(f"{k} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for k, (x, y), (w, h) in zip(cls, c, wh) if w > 0 and h > 0))


def fix_image(gc, gb, P):
    d = np.array(P, float).reshape(-1, 6); dc, ds, db = d[:, 0].astype(int), d[:, 1], d[:, 2:]
    gc, gb = gc.copy(), gb.copy(); keep = np.ones(len(gc), bool); log = []
    # R2 merge fragments first (they would otherwise each fail R1)
    add_c, add_b = [], []
    for c in (1, 2):
        cand = np.where((dc == c) & (ds >= 0.6))[0]; gi = np.where((gc == c) & keep)[0]
        if not len(cand) or len(gi) < 2: continue
        A = ioa(gb[gi], db[cand])
        for k in np.argsort(-ds[cand]):
            grp = gi[(A[:, k] >= 0.8) & keep[gi]]
            if len(grp) >= 2:
                keep[grp] = False; add_c.append(c); add_b.append(db[cand[k]]); log.append(("R2_merge", NAMES[c], len(grp)))
    # R1 refine: mutual best same-class match
    U = iou(gb, db)
    for j in np.where(keep)[0]:
        same = (dc == gc[j]) & (ds >= 0.8)
        if not same.any(): continue
        u = np.where(same, U[j], -1); k = int(u.argmax())
        if 0.5 <= u[k] < 0.75 and int(np.argmax(np.where(gc == gc[j], U[:, k], -1))) == j:
            gb[j] = db[k]; log.append(("R1_refine", NAMES[gc[j]], round(float(u[k]), 2)))
    # R3 add unlabelled insulators
    gi = gb[(gc == 0)]
    for k in np.where((dc == 0) & (ds >= 0.75))[0]:
        if not len(gi) or iou(db[k:k + 1], gi).max() < 0.3:
            add_c.append(0); add_b.append(db[k]); gi = np.r_[gi.reshape(-1, 4), db[k:k + 1]]; log.append(("R3_add_ins", "insulator", 1))
    cls = np.r_[gc[keep], np.array(add_c, int)]; box = np.r_[gb[keep], np.array(add_b).reshape(-1, 4)]
    return cls, box, log


def tile(split, image, box, text, w=360):
    im = Image.open(V6 / "images" / split / image).convert("RGB"); W, H = im.size; dr = ImageDraw.Draw(im)
    gc, gb = read(V6 / "labels" / split / (Path(image).stem + ".txt"))
    for c, b in zip(gc, gb): dr.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=COL[c], width=max(2, W // 400))
    b = np.array(box); cx, cy = (b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H
    s = max((b[2] - b[0]) * W, (b[3] - b[1]) * H) * 3 + 64
    dr.rectangle([b[0] * W - 3, b[1] * H - 3, b[2] * W + 3, b[3] * H + 3], outline=(255, 255, 255), width=max(3, W // 300))
    t = im.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))).resize((w, w))
    ImageDraw.Draw(t).rectangle([0, 0, w, 16], fill=(0, 0, 0)); ImageDraw.Draw(t).text((3, 2), text, fill=(255, 255, 255))
    return t


def sheets(df, name, per=24):
    for p in range(0, len(df), per):
        part = df.iloc[p:p + per]; cols = 6; sh = Image.new("RGB", (cols * 360, 360 * ((len(part) + cols - 1) // cols)), "white")
        for i, r in enumerate(part.itertuples()):
            sh.paste(tile(r.split, r.image, r.box, f"#{r.id} {r.cls} {r.flag} c{r.conf}"), ((i % cols) * 360, (i // cols) * 360))
        sh.save(REV / f"{name}_{p // per + 1:02d}.jpg", quality=85)


if __name__ == "__main__":
    if V7.exists(): shutil.rmtree(V7)
    REV.mkdir(parents=True, exist_ok=True)
    for f in REV.glob("*.jpg"): f.unlink()
    P = json.loads((AUD / "pred_train.json").read_text())
    stats, changes = Counter(), []
    for split in ["train", "val", "val_clean", "val_clean_v6"]:
        (V7 / "images" / split).mkdir(parents=True); (V7 / "labels" / split).mkdir(parents=True)
        for im in (V6 / "images" / split).iterdir():
            os.link(im, V7 / "images" / split / im.name)
        for lab in (V6 / "labels" / split).glob("*.txt"):
            if split != "train": shutil.copy2(lab, V7 / "labels" / split / lab.name); continue
            gc, gb = read(lab); cls, box, log = fix_image(gc, gb, P.get(lab.stem + ".jpg", []))
            write(V7 / "labels" / split / lab.name, cls, box)
            for rule, c, v in log: stats[(rule, c)] += 1; changes.append({"image": lab.stem + ".jpg", "rule": rule, "cls": c, "detail": v})
    shutil.copytree(V6 / "annotations", V7 / "annotations")   # COCO json = v6 (sizes / ids); labels/*.txt are the truth
    for y in ["data.yaml", "data_val_clean.yaml", "data_val_clean_v6.yaml"]:
        cfg = yaml.safe_load((V6 / y).read_text()); cfg["path"] = str(V7); (V7 / y).write_text(yaml.safe_dump(cfg, sort_keys=False))
    pd.DataFrame(changes).to_csv(V7 / "changes_v7.csv", index=False)
    n = sum(1 for _ in (V7 / "labels/train").glob("*.txt"))
    cnt = Counter(int(l.split()[0]) for f in (V7 / "labels/train").glob("*.txt") for l in f.read_text().splitlines() if l.strip())
    man = {"source": "v6p5_v6", "rules": ["R1_refine", "R2_merge", "R3_add_ins"], "teacher": "v6p5_v5 teacher @640",
           "changes": {f"{r}:{c}": v for (r, c), v in sorted(stats.items())},
           "train_boxes": {NAMES[k]: v for k, v in sorted(cnt.items())}, "train_images": n, "val": "unchanged from v6"}
    (V7 / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man["changes"], indent=1)); print("train boxes v7:", man["train_boxes"])

    # ---- review queue: only what a human must decide, highest value first, capped ----
    fl = pd.read_csv(AUD / "flags.csv"); fl["box"] = fl.box.apply(json.loads)
    D = fl[fl.cls.isin(NAMES[1:6])].copy()
    # teacher-error filter: FO predicted on a labelled person is the known v5 confusion, not a missing label
    def on_person(r):
        gc, gb = read(V6 / "labels" / r.split / (Path(r.image).stem + ".txt"))
        return bool(len(gb[gc == 6]) and iou(np.array([r.box]), gb[gc == 6]).max() >= 0.3)
    D = D[~((D.flag == "missing_label") & (D.cls == "foreign_object") & D.apply(on_person, axis=1))]
    D = D[(D.flag != "missing_label") | (D.conf >= 0.7)]
    D = D[D.flag.isin(["missing_label", "missed_gt", "wrong_class"])]
    # priority: val first (it is the yardstick), then rare classes, then teacher confidence
    rare = {"foreign_object": 0, "insulator_flashover": 1, "broken_strand": 2, "insulator_broken": 3, "bird_nest": 4}
    D["prio"] = (D.split != "val_clean_v6") * 10 + D.cls.map(rare)
    D = D.sort_values(["prio", "conf"], ascending=[True, False])
    q_val = D[D.split == "val_clean_v6"]; q_tr = D[(D.split == "train") & (D.cls.isin(["foreign_object", "insulator_flashover"]))].head(120)
    Q = pd.concat([q_val, q_tr]).reset_index(drop=True); Q.insert(0, "id", range(1, len(Q) + 1))
    Q["decision"] = ""   # fill: ok (label is right) | fix (move/resize) | drop (delete box) | add (add the predicted box) | relabel:<class>
    Q[["id", "split", "image", "cls", "flag", "conf", "best_iou", "box", "decision"]].to_csv(REV / "review_queue.csv", index=False)
    for (sp, fg), part in Q.groupby(["split", "flag"], sort=False):
        sheets(part, f"{sp}_{fg}")
    print(f"review queue: {len(Q)} (val_clean_v6 {len(q_val)}, train FO/flashover {len(q_tr)}) -> {REV}")
    print(Q.groupby(["split", "flag", "cls"]).size().to_string())
