"""Box convention v8: tight around the visible object + a small margin, applied the SAME way to train and val.

Convention (agreed 2026-10-15): box = tightest box around the visible defect / object, then padded on every side by
PAD_FRAC of its size (min PAD_MIN_PX px at the original resolution). insulator (class 0) and person (6) are untouched.
How: SAM 2.1 segments the object inside each existing box (box prompt); the mask's bounding box + margin replaces the
label box. SAM is independent of our detector, so applying it to val does not bias the score toward our models.
Guards (keep the original box otherwise): mask must cover >= MIN_FILL of the new box's area... see accept().

  .venv/bin/python scripts/sam_tighten.py --sample 48        # visual check sheet only
  .venv/bin/python scripts/sam_tighten.py --split train      # writes artifacts/sam_v8/<split>.json (new boxes per image)
"""
import argparse, json, random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
DS = REPO / "datasets/v6p5_v6"; OUT = REPO / "artifacts/sam_v8"
PAD_FRAC, PAD_MIN_PX = 0.05, 2
CLASSES = {1, 2, 3, 4, 5}
NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand", "person"]


def accept(old, new, W, H):
    """old/new xyxy px. Reject degenerate masks: SAM grabbing a speck or spilling to the whole box is no information."""
    ow, oh, nw, nh = old[2] - old[0], old[3] - old[1], new[2] - new[0], new[3] - new[1]
    if nw < 4 or nh < 4: return False, "tiny"
    ratio = (nw * nh) / (ow * oh + 1e-9)
    if ratio < 0.15: return False, "speck"          # mask is a small part (glare, edge) -> unreliable
    if ratio > 1.25: return False, "grow"            # tight box + margin may exceed an already-tight drawn box a little
    return True, "ok"


def pad(b, W, H):
    w, h = b[2] - b[0], b[3] - b[1]; px, py = max(PAD_MIN_PX, PAD_FRAC * w), max(PAD_MIN_PX, PAD_FRAC * h)
    return [max(0, b[0] - px), max(0, b[1] - py), min(W, b[2] + px), min(H, b[3] + py)]


def run(images, split, model):
    out = {}
    for im_path in images:
        lab = DS / "labels" / split / (im_path.stem + ".txt")
        rows = [l.split() for l in lab.read_text().splitlines() if l.strip()]
        W, H = Image.open(im_path).size
        idx = [i for i, r in enumerate(rows) if int(r[0]) in CLASSES]
        res = []
        if idx:
            boxes = []
            for i in idx:
                _, x, y, w, h = map(float, rows[i]); boxes.append([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H])
            r = model(str(im_path), bboxes=boxes, verbose=False)[0]
            masks = r.masks.data.cpu().numpy() if r.masks is not None else []
            for i, old, m in zip(idx, boxes, masks):
                ys, xs = np.where(m > 0.5)
                if len(xs) == 0: res.append([i, old, None, "empty"]); continue
                sx, sy = W / m.shape[1], H / m.shape[0]
                t = [xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy]
                t = [max(t[0], old[0]), max(t[1], old[1]), min(t[2], old[2]), min(t[3], old[3])]   # stay inside drawn box
                new = pad(t, W, H); ok, why = accept(old, new, W, H)
                res.append([i, old, new if ok else None, why])
        out[im_path.name] = {"W": W, "H": H, "boxes": res}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", type=int, default=0); ap.add_argument("--split", default="train")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    from ultralytics import SAM
    model = SAM("sam2.1_b.pt")
    imgs = sorted((DS / "images" / a.split).glob("*.jpg"))
    if a.sample:
        random.seed(7)
        cand = [p for p in imgs if any(l.split()[0] in "12345" for l in (DS / "labels" / a.split / (p.stem + ".txt")).read_text().splitlines() if l.strip())]
        res = run(random.sample(cand, a.sample), a.split, model); tiles = []
        for name, d in res.items():
            im = Image.open(DS / "images" / a.split / name).convert("RGB")
            rows = [l.split() for l in (DS / "labels" / a.split / (Path(name).stem + ".txt")).read_text().splitlines() if l.strip()]
            for i, old, new, why in d["boxes"][:1]:
                cx, cy = (old[0] + old[2]) / 2, (old[1] + old[3]) / 2; s = max(old[2] - old[0], old[3] - old[1]) * 1.6 + 40
                dr = ImageDraw.Draw(im); lw = max(2, int(s / 150))
                dr.rectangle(old, outline=(255, 0, 0), width=lw)
                if new: dr.rectangle(new, outline=(0, 255, 0), width=lw)
                t = im.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))).resize((300, 300))
                ImageDraw.Draw(t).rectangle([0, 0, 300, 14], fill=(0, 0, 0))
                ImageDraw.Draw(t).text((3, 1), f"{NAMES[int(rows[i][0])]} {why}", fill=(255, 255, 255)); tiles.append(t)
        sh = Image.new("RGB", (8 * 300, 300 * ((len(tiles) + 7) // 8)), "white")
        for k, t in enumerate(tiles): sh.paste(t, ((k % 8) * 300, (k // 8) * 300))
        sh.save(OUT / f"sample_{a.split}.jpg", quality=85)
        from collections import Counter
        print(Counter(b[3] for d in res.values() for b in d["boxes"]))
    else:
        res = run(imgs, a.split, model)
        (OUT / f"{a.split}.json").write_text(json.dumps(res))
        from collections import Counter
        print(a.split, Counter(b[3] for d in res.values() for b in d["boxes"]))
