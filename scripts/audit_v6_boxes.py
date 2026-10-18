"""Box-quality audit of v6 (train + val_clean_v6): where do labels and the v5 teacher disagree?

Why: the AP50 -> AP50-95 gap (~0.3-0.5) is box extent, and per-source analysis shows mixed labelling conventions
(broken_strand: long-span boxes on 640x640 phone photos vs tight break-point boxes on 640x480 drone close-ups;
flashover: tiny top patch vs whole-shed region). Flags, per GT / prediction (teacher @640, conf >= 0.001):
  loose_box      teacher agrees on the class (conf >= 0.5) but best IoU in [0.3, 0.7)  -> extent convention / sloppy box
  missed_gt      no same-class prediction with IoU >= 0.3 and conf >= 0.1                -> wrong class, invisible, or tiny
  missing_label  teacher conf >= 0.6, no GT of that class with IoU >= 0.3               -> unlabelled object
  wrong_class    teacher conf >= 0.6 on another class with IoU >= 0.6 to the GT           -> class confusion / relabel
The teacher was trained on train, so train flags are the labels it REFUSED to fit -> high-precision review queue.

  .venv/bin/python scripts/audit_v6_boxes.py           (writes artifacts/audit_v6_boxes/{flags.csv, summary.txt, sheets/})
"""
import json, sys
from pathlib import Path

import numpy as np, pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
DS = REPO / "datasets/v6p5_v6"; OUT = REPO / "artifacts/audit_v6_boxes"; (OUT / "sheets").mkdir(parents=True, exist_ok=True)
TEACHER = REPO / "artifacts/v6p5_v6_teachers/init/teacher_v5.pt"
N = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand", "person"]
COL = [(0, 200, 255), (255, 128, 0), (255, 0, 255), (0, 200, 0), (255, 255, 0), (255, 0, 0), (128, 0, 255)]


def iou(a, b):
    wh = np.clip(np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0, None)
    i = wh[..., 0] * wh[..., 1]; ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return i / (ar(a)[:, None] + ar(b)[None] - i + 1e-9)


def predict(split, imgs):
    cache = OUT / f"pred_{split}.json"
    if cache.is_file(): return json.loads(cache.read_text())
    m = YOLO(str(TEACHER)); out = {}
    for k in range(0, len(imgs), 32):
        for im, r in zip(imgs[k:k + 32], m.predict([str(p) for p in imgs[k:k + 32]], imgsz=640, conf=0.001, rect=False,
                                                    verbose=False, device=0, batch=32)):
            out[im.name] = [[int(c), float(s), *b] for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxyn.tolist())]
        if k % 3200 == 0: print(split, k, len(imgs), flush=True)
    cache.write_text(json.dumps(out)); return out


def audit(split):
    coco = json.loads((DS / f"annotations/instances_{split}.json").read_text())
    size = {Path(i["file_name"]).name: (i["width"], i["height"]) for i in coco["images"]}
    imgs = sorted((DS / "images" / split).glob("*.jpg")); P = predict(split, imgs); rows = []
    for im in imgs:
        W, H = size.get(im.name, (0, 0)); src = f"{W}x{H}"
        g = np.array([l.split() for l in (DS / "labels" / split / f"{im.stem}.txt").read_text().splitlines() if l.strip()], float).reshape(-1, 5)
        gc, gb = g[:, 0].astype(int), np.c_[g[:, 1:3] - g[:, 3:5] / 2, g[:, 1:3] + g[:, 3:5] / 2]
        d = np.array(P[im.name], float).reshape(-1, 6); dc, ds, db = d[:, 0].astype(int), d[:, 1], d[:, 2:]
        U = iou(gb, db) if len(gb) and len(db) else np.zeros((len(gb), len(db)))
        for j in range(len(gb)):
            same = (dc == gc[j]); u = U[j] if len(db) else np.zeros(0)
            best = (u * same).max() if same.any() else 0.0
            conf_at = ds[same & (u >= 0.3)].max() if (same & (u >= 0.3)).any() else 0.0
            base = dict(split=split, image=im.name, source=src, cls=N[gc[j]], box=gb[j].round(4).tolist(), best_iou=round(float(best), 3),
                        conf=round(float(conf_at), 3))
            if conf_at >= 0.5 and 0.3 <= best < 0.7: rows.append({**base, "flag": "loose_box"})
            elif conf_at < 0.1: rows.append({**base, "flag": "missed_gt"})
            other = (~same) & (ds >= 0.6) & (u >= 0.6)
            if other.any():
                k = np.argmax(np.where(other, ds, 0)); rows.append({**base, "flag": "wrong_class", "pred_cls": N[dc[k]], "conf": round(float(ds[k]), 3)})
        for k in np.where(ds >= 0.6)[0]:
            if not ((gc == dc[k]) & ((U[:, k] if len(gb) else np.zeros(0)) >= 0.3)).any():
                rows.append(dict(split=split, image=im.name, source=src, cls=N[dc[k]], box=db[k].round(4).tolist(), best_iou=0.0,
                                 conf=round(float(ds[k]), 3), flag="missing_label"))
    return pd.DataFrame(rows)


def sheet(df, name, n=16):
    df = df.sort_values("conf", ascending=False).head(n); tiles = []
    for _, r in df.iterrows():
        im = Image.open(DS / "images" / r.split / r.image).convert("RGB"); W, H = im.size; dr = ImageDraw.Draw(im)
        lab = DS / "labels" / r.split / (Path(r.image).stem + ".txt")
        for l in lab.read_text().splitlines():
            c, x, y, w, h = l.split(); x, y, w, h = map(float, (x, y, w, h))
            dr.rectangle([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H], outline=COL[int(c)], width=2)
        b = r.box; dr.rectangle([b[0] * W - 4, b[1] * H - 4, b[2] * W + 4, b[3] * H + 4], outline=(255, 255, 255), width=4)
        t = im.resize((400, int(400 * H / W))).crop((0, 0, 400, 300)); ImageDraw.Draw(t).text((4, 4), f"{r.cls} {r.flag} c{r.conf}", fill=(255, 255, 255))
        tiles.append(t)
    if not tiles: return
    sh = Image.new("RGB", (1600, 300 * ((len(tiles) + 3) // 4)), "white")
    for i, t in enumerate(tiles): sh.paste(t, ((i % 4) * 400, (i // 4) * 300))
    sh.save(OUT / "sheets" / f"{name}.jpg", quality=80)


if __name__ == "__main__":
    dfs = []
    for split in ["val_clean_v6", "train"]:
        df = audit(split); dfs.append(df); print(split, "flags:", len(df), flush=True)
    df = pd.concat(dfs, ignore_index=True); df.to_csv(OUT / "flags.csv", index=False)
    lines = []
    for split in ["train", "val_clean_v6"]:
        d = df[df.split == split]
        lines.append(f"\n== {split}: flags by class x type\n" + pd.crosstab(d.cls, d.flag, margins=True).to_string())
        lines.append(f"\n== {split}: loose_box + missed_gt by source (top)\n"
                     + pd.crosstab([d.cls, d.source], d.flag).query("loose_box + missed_gt >= 20").to_string())
    (OUT / "summary.txt").write_text("\n".join(lines)); print("\n".join(lines))
    for cls in N[1:6] + ["insulator"]:
        for fl in ["loose_box", "missed_gt", "missing_label", "wrong_class"]:
            sheet(df[(df.split == "train") & (df.cls == cls) & (df.flag == fl)], f"train_{cls}_{fl}")
    print("sheets:", OUT / "sheets")
