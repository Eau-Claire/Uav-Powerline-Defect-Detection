"""Prototype: diffusion-inpainted foreign objects (FO, class 4) on real TRAIN backgrounds.

Copy-paste FO was dropped from v5/v6 ("pasted object never fitted"). Here SDXL-inpainting draws the object INTO the
scene (light, blur, perspective match). Placement = next to an insulator (class 0) box, like dataset/synthetic.py.
Label = bbox of the pixels the model actually changed inside the mask; auto-QA = teacher v5 must see an FO there.

  .venv/bin/python scripts/proto_diffusion_fo.py --n 16 --out artifacts/diffusion_fo_proto
Output: <out>/images/*.jpg, <out>/labels/*.txt (YOLO, original labels + new FO), <out>/meta.csv, <out>/sheet.jpg
"""
import argparse, csv, random
from pathlib import Path

import numpy as np, torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
DS = REPO / "datasets/v6p5_v6"
PROMPTS = [
    "a torn white plastic bag tangled on a high voltage power line, aerial drone photo, realistic",
    "a black plastic bag caught on an electric transmission line insulator, drone photo, realistic",
    "a red kite stuck on overhead power line wires, aerial photo, realistic",
    "a faded blue tarp cloth hanging on a power line, drone inspection photo, realistic",
    "a shiny foil balloon caught on electric wires, aerial drone photo, realistic",
    "a piece of rope and fabric debris hanging from a power line, drone photo, realistic",
]
NEG = "cartoon, drawing, text, watermark, blurry, deformed, bird, nest, person"


def pick_backgrounds(rng, n):
    out = []
    files = sorted((DS / "labels/train").glob("*.txt")); rng.shuffle(files)
    for f in files:
        rows = [l.split() for l in f.read_text().splitlines() if l.strip()]
        cls = [int(r[0]) for r in rows]
        if 4 in cls or 6 in cls or f.stem.startswith("v65_"): continue           # no existing FO / person
        ins = [list(map(float, r[1:])) for r in rows if int(r[0]) == 0 and 0.04 < float(r[3]) < 0.4]
        if ins and (DS / "images/train" / f"{f.stem}.jpg").is_file():
            out.append((f.stem, rows, rng.choice(ins)))
        if len(out) >= n: break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16); ap.add_argument("--out", default="artifacts/diffusion_fo_proto")
    ap.add_argument("--seed", type=int, default=20261015); ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--strength", type=float, default=0.99)
    a = ap.parse_args()
    out = REPO / a.out; (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True)
    rng = random.Random(a.seed)

    from diffusers import AutoPipelineForInpainting
    pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                     torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)
    from ultralytics import YOLO
    teacher = YOLO(str(REPO / "artifacts/v6p5_v6_teachers/init/teacher_v5.pt"))

    meta = []
    for i, (stem, rows, (xc, yc, w, h)) in enumerate(pick_backgrounds(rng, a.n)):
        img = Image.open(DS / "images/train" / f"{stem}.jpg").convert("RGB"); W, H = img.size
        # object box: beside the insulator (left/right/above), ~0.5-1.0x insulator size
        s = rng.uniform(0.8, 1.4); ow = max(w * W * s, 64); oh = max(ow * rng.uniform(0.7, 1.3), 64)
        side = rng.choice(["left", "right", "above"])
        cx = xc * W + {"left": -(w * W + ow) / 2, "right": (w * W + ow) / 2, "above": 0}[side]
        cy = yc * H + ({"above": -(h * H + oh) / 2}.get(side, rng.uniform(-0.3, 0.3) * h * H))
        x0, y0, x1, y1 = [int(v) for v in (cx - ow / 2, cy - oh / 2, cx + ow / 2, cy + oh / 2)]
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 - x0 < 32 or y1 - y0 < 32: continue
        # 1024 crop around the object (SDXL native res); scale down if the object is large
        side_px = int(min(max(256, 2.2 * max(x1 - x0, y1 - y0)), W, H))   # zoom in: mask ~1/3-1/2 of the crop, else SDXL just repaints background
        cx0 = int(np.clip((x0 + x1) / 2 - side_px / 2, 0, W - side_px)); cy0 = int(np.clip((y0 + y1) / 2 - side_px / 2, 0, H - side_px))
        crop = img.crop((cx0, cy0, cx0 + side_px, cy0 + side_px)).resize((1024, 1024), Image.LANCZOS)
        k = 1024 / side_px; mb = [int((x0 - cx0) * k), int((y0 - cy0) * k), int((x1 - cx0) * k), int((y1 - cy0) * k)]
        mask = Image.new("L", (1024, 1024), 0); ImageDraw.Draw(mask).ellipse(mb, fill=255)
        prompt = rng.choice(PROMPTS)
        g = torch.Generator("cuda").manual_seed(a.seed + i)
        gen = pipe(prompt=prompt, negative_prompt=NEG, image=crop, mask_image=mask, num_inference_steps=a.steps,
                   strength=a.strength, guidance_scale=9.0, generator=g).images[0].resize((1024, 1024))
        # keep generated pixels only inside the (feathered) mask -> rest of the image is untouched
        soft = mask.filter(__import__("PIL.ImageFilter", fromlist=["x"]).GaussianBlur(6))
        merged = Image.composite(gen, crop, soft)
        diff = np.abs(np.asarray(merged, float) - np.asarray(crop, float)).mean(2) * (np.asarray(mask) > 0)
        ys, xs = np.where(diff > 25)
        if len(xs) < 200: meta.append({"i": i, "stem": stem, "status": "no_object"}); continue
        bx = [np.percentile(xs, 1), np.percentile(ys, 1), np.percentile(xs, 99), np.percentile(ys, 99)]
        patch = merged.resize((side_px, side_px), Image.LANCZOS); full = img.copy(); full.paste(patch, (cx0, cy0))
        fb = [cx0 + bx[0] / k, cy0 + bx[1] / k, cx0 + bx[2] / k, cy0 + bx[3] / k]
        name = f"dfo_{i:04d}_{stem}"
        full.save(out / "images" / f"{name}.jpg", quality=95)
        lab = [" ".join(r) for r in rows] + [f"4 {(fb[0] + fb[2]) / 2 / W:.6f} {(fb[1] + fb[3]) / 2 / H:.6f} "
                                             f"{(fb[2] - fb[0]) / W:.6f} {(fb[3] - fb[1]) / H:.6f}"]
        (out / "labels" / f"{name}.txt").write_text("\n".join(lab) + "\n")
        # auto-QA: does the v5 teacher see an FO overlapping the new box?
        r = teacher.predict(full, imgsz=800, conf=0.05, verbose=False)[0]
        best = 0.0
        for c, s_, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            ix = max(0, min(b[2], fb[2]) - max(b[0], fb[0])) * max(0, min(b[3], fb[3]) - max(b[1], fb[1]))
            iou = ix / ((b[2] - b[0]) * (b[3] - b[1]) + (fb[2] - fb[0]) * (fb[3] - fb[1]) - ix + 1e-9)
            if int(c) == 4 and iou > 0.3: best = max(best, s_)
        meta.append({"i": i, "stem": stem, "status": "ok", "name": name, "prompt": prompt, "teacher_fo_conf": round(best, 3),
                     "box": [round(v) for v in fb]})
        # preview tile: 512 crop with the box drawn
        t = merged.copy(); ImageDraw.Draw(t).rectangle(bx, outline=(255, 0, 0), width=3)
        t.resize((384, 384)).save(out / f"tile_{i:04d}.jpg"); crop.resize((384, 384)).save(out / f"tile_{i:04d}_orig.jpg")
        print(f"{i:3d} {stem} conf={best:.2f} | {prompt[:50]}", flush=True)

    with open(out / "meta.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=sorted({k for m in meta for k in m})); wr.writeheader(); wr.writerows(meta)
    tiles = sorted(p for p in out.glob("tile_*.jpg") if "_orig" not in p.name)
    if tiles:
        cols = 4; rows_ = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 768, rows_ * 384), "white")
        for j, p in enumerate(tiles):
            sheet.paste(Image.open(str(p).replace(".jpg", "_orig.jpg")), ((j % cols) * 768, (j // cols) * 384))
            sheet.paste(Image.open(p), ((j % cols) * 768 + 384, (j // cols) * 384))
        sheet.save(out / "sheet.jpg", quality=85)
    ok = [m for m in meta if m["status"] == "ok"]
    print(f"done: {len(ok)}/{len(meta)} generated, teacher sees FO (conf>=0.25): {sum(m['teacher_fo_conf'] >= 0.25 for m in ok)}")


if __name__ == "__main__":
    main()
