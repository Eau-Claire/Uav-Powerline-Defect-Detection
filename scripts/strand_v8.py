"""v8 convention for broken_strand: box tight around the frayed / broken strands (+5% margin), not the whole wire span.

71% of broken_strand boxes on the 640x640 source (phone photos) are "long" (aspect >= 3 or span >= 60% of the image)
while the drone sources box the break point (aspect ~1.3). Model-assisted conversion:
  1. localizer: YOLO26s, 1 class, trained ONLY on images whose broken_strand boxes are all compact (+ no val-leaked
     train images) -> it learns "where the fray is" from the tight convention
  2. each long box (train and val_clean_v6): crop the box + 15% context, predict, keep fray boxes centred inside it
       found  (conf >= 0.5, union < 70% of the long box)  -> replaced by the fray boxes (+5% margin)
       keep   (union >= 70%: the long box was already the fray)
       review (best conf 0.2-0.5, or nothing found)      -> human queue (val) / kept as is (train)
Outputs artifacts/strand_v8/{proposals_<split>.json, sheet_<split>_<decision>_NN.jpg, localizer/}

  .venv/bin/python scripts/strand_v8.py
"""
import json, os, random, shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
DS = REPO / "datasets/v6p5_v6"; OUT = REPO / "artifacts/strand_v8"; LDS = OUT / "localizer_ds"
LEAK_TRAIN = {t for _, ts in json.loads((REPO / "artifacts/review_v7/val_train_near_dups.json").read_text()) for t, _ in ts}
PAD = 0.05
random.seed(20261015)


def sizes(split):
    coco = json.loads((DS / f"annotations/instances_{split}.json").read_text())
    return {Path(i["file_name"]).name: (i["width"], i["height"]) for i in coco["images"]}


def strand_boxes(split):
    """-> {image: [(row_index, xywhn, is_long)]} for broken_strand (class 5)"""
    S, out = sizes(split), {}
    for f in (DS / "labels" / split).glob("*.txt"):
        W, H = S.get(f.stem + ".jpg", (1, 1)); rows = []
        for i, l in enumerate(f.read_text().splitlines()):
            p = l.split()
            if not p or p[0] != "5": continue
            x, y, w, h = map(float, p[1:]); asp = max(w * W, h * H) / max(1.0, min(w * W, h * H))
            rows.append((i, [x, y, w, h], asp >= 3 or max(w, h) >= 0.6))
        if rows: out[f.stem + ".jpg"] = rows
    return out


def build_localizer_ds(B):
    if (LDS / "data.yaml").is_file(): return LDS / "data.yaml"
    imgs = [n for n, rows in B.items() if not any(r[2] for r in rows) and n not in LEAK_TRAIN]
    random.shuffle(imgs); nv = max(40, len(imgs) // 10)
    for split, part in [("val", imgs[:nv]), ("train", imgs[nv:])]:
        (LDS / "images" / split).mkdir(parents=True, exist_ok=True); (LDS / "labels" / split).mkdir(parents=True, exist_ok=True)
        for n in part:
            os.link(DS / "images/train" / n, LDS / "images" / split / n)
            (LDS / "labels" / split / (Path(n).stem + ".txt")).write_text("".join(f"0 {x} {y} {w} {h}\n" for _, (x, y, w, h), _ in B[n]))
    (LDS / "data.yaml").write_text(f"path: {LDS}\ntrain: images/train\nval: images/val\nnames:\n  0: broken_strand\n")
    print(f"localizer ds: train {len(imgs) - nv}, val {nv} images", flush=True)
    return LDS / "data.yaml"


def convert(split, B, model):
    S = sizes(split); props = {}
    for n, rows in B.items():
        W, H = S[n]; im = None
        for i, (x, y, w, h), is_long in rows:
            if not is_long: continue
            if im is None: im = Image.open(DS / "images" / split / n).convert("RGB")
            x0, y0, x1, y1 = (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H
            cx0, cy0 = max(0, x0 - 0.15 * w * W), max(0, y0 - 0.15 * h * H)
            cx1, cy1 = min(W, x1 + 0.15 * w * W), min(H, y1 + 0.15 * h * H)
            r = model.predict(im.crop((int(cx0), int(cy0), int(cx1), int(cy1))), imgsz=640, conf=0.05, verbose=False)[0]
            cand = []
            for s, b in zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                b = [b[0] + int(cx0), b[1] + int(cy0), b[2] + int(cx0), b[3] + int(cy0)]
                mx, my = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if x0 <= mx <= x1 and y0 <= my <= y1:
                    b = [max(b[0], x0), max(b[1], y0), min(b[2], x1), min(b[3], y1)]; cand.append((s, b))
            good = [b for s, b in cand if s >= 0.5]; best = max([s for s, _ in cand], default=0.0)
            if good:
                m = np.zeros((int(H), int(W)), bool)
                for b in good: m[int(b[1]):int(b[3]) + 1, int(b[0]):int(b[2]) + 1] = True
                share = m[int(y0):int(y1) + 1, int(x0):int(x1) + 1].mean()
                if share >= 0.7: dec, new = "keep", None
                else:
                    dec = "found"; new = []
                    for b in good:
                        px, py = max(2, PAD * (b[2] - b[0])), max(2, PAD * (b[3] - b[1]))
                        new.append([max(0, b[0] - px) / W, max(0, b[1] - py) / H, min(W, b[2] + px) / W, min(H, b[3] + py) / H])
            else:
                dec, new = "review", None
            props.setdefault(n, []).append({"row": i, "old": [x0 / W, y0 / H, x1 / W, y1 / H], "new": new, "decision": dec,
                                            "best_conf": round(best, 3)})
    return props


def sheets(split, props):
    for dec in ["found", "keep", "review"]:
        items = [(n, p) for n, ps in props.items() for p in ps if p["decision"] == dec]
        random.shuffle(items)
        for k in range(0, min(len(items), 96), 24):
            sh = Image.new("RGB", (6 * 320, 320 * 4), "white")
            for j, (n, p) in enumerate(items[k:k + 24]):
                im = Image.open(DS / "images" / split / n).convert("RGB"); W, H = im.size; dr = ImageDraw.Draw(im); lw = max(2, W // 200)
                o = p["old"]; dr.rectangle([o[0] * W, o[1] * H, o[2] * W, o[3] * H], outline=(255, 0, 0), width=lw)
                for b in p["new"] or []: dr.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=(0, 255, 0), width=lw)
                t = im.resize((320, int(320 * H / W))).crop((0, 0, 320, 320))
                ImageDraw.Draw(t).rectangle([0, 0, 320, 14], fill=(0, 0, 0)); ImageDraw.Draw(t).text((3, 1), f"{dec} c{p['best_conf']}", fill=(255, 255, 255))
                sh.paste(t, ((j % 6) * 320, (j // 6) * 320))
            sh.save(OUT / f"sheet_{split}_{dec}_{k // 24 + 1:02d}.jpg", quality=80)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    Btr = strand_boxes("train"); data = build_localizer_ds(Btr)
    best = OUT / "localizer/weights/best.pt"
    if not best.is_file():
        YOLO("yolo26s.pt").train(data=str(data), epochs=60, imgsz=640, batch=32, project=str(OUT), name="localizer", exist_ok=True,
                                 seed=20261015, plots=False, workers=8, patience=20, verbose=False)
    model = YOLO(str(best))
    for split, B in [("val_clean_v6", strand_boxes("val_clean_v6")), ("train", Btr)]:
        props = convert(split, B, model)
        (OUT / f"proposals_{split}.json").write_text(json.dumps(props))
        from collections import Counter
        print(split, Counter(p["decision"] for ps in props.values() for p in ps), flush=True)
        sheets(split, props)
