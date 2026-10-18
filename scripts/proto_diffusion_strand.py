"""Prototype: diffusion-inpainted broken_strand (class 5) on real DRONE images (640x480 source, no strand label).

broken_strand is 77% phone/ground photos; the drone domain has only 360 train boxes (AP50 0.56). Here a conductor is
found with Canny + Hough (long thin line), a zoomed crop is taken so the mask is ~1/3 of it (SDXL ignores tiny masks),
the mask is a band ALONG the wire, and SDXL-inpainting draws a frayed section. Label = bbox of changed pixels.

  .venv/bin/python scripts/proto_diffusion_strand.py --n 16 --out artifacts/diffusion_strand_proto
"""
import argparse, json, math, random
from pathlib import Path

import cv2, numpy as np, torch
from PIL import Image, ImageDraw, ImageFilter

REPO = Path(__file__).resolve().parents[1]
DS = REPO / "datasets/v6p5_v6"
PROMPTS = [
    "close-up of a damaged overhead power line conductor, several broken aluminum strands sticking out, frayed cable, drone photo",
    "frayed steel-aluminium transmission line cable with broken outer strands splayed outward, aerial inspection photo",
    "power line conductor with unravelled broken wire strands, realistic drone inspection photo",
]
NEG = "bird, nest, plastic bag, cloth, text, cartoon, drawing, new cable, thick rope, blurry"


def find_wire(img):
    g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY); e = cv2.Canny(g, 60, 160)
    L = cv2.HoughLinesP(e, 1, np.pi / 180, 80, minLineLength=min(img.size) // 3, maxLineGap=8)
    if L is None: return None
    L = sorted(L.reshape(-1, 4).tolist(), key=lambda l: -math.hypot(l[2] - l[0], l[3] - l[1]))
    return L[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="artifacts/diffusion_strand_proto"); ap.add_argument("--seed", type=int, default=20261015)
    a = ap.parse_args(); out = REPO / a.out; (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True)
    rng = random.Random(a.seed)
    coco = json.loads((DS / "annotations/instances_train.json").read_text())
    drone = [Path(i["file_name"]).name for i in coco["images"] if (i["width"], i["height"]) == (640, 480)]
    rng.shuffle(drone)
    from diffusers import AutoPipelineForInpainting
    pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                     torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)
    tiles, meta, i = [], [], 0
    for name in drone:
        if len(tiles) >= a.n: break
        lab = DS / "labels/train" / (Path(name).stem + ".txt"); rows = [l for l in lab.read_text().splitlines() if l.strip()]
        if any(r.split()[0] in ("4", "5") for r in rows): continue
        img = Image.open(DS / "images/train" / name).convert("RGB"); W, H = img.size
        line = find_wire(img)
        if line is None: continue
        x0, y0, x1, y1 = line; t = rng.uniform(0.3, 0.7); px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        ang = math.atan2(y1 - y0, x1 - x0)
        side = 192; cx0 = int(np.clip(px - side / 2, 0, W - side)); cy0 = int(np.clip(py - side / 2, 0, H - side))
        crop = img.crop((cx0, cy0, cx0 + side, cy0 + side)).resize((1024, 1024), Image.LANCZOS); k = 1024 / side
        # band along the wire: length ~45% of the crop, width ~14%
        qx, qy = (px - cx0) * k, (py - cy0) * k; hl, hw = 230, 72
        dx, dy, nx, ny = math.cos(ang), math.sin(ang), -math.sin(ang), math.cos(ang)
        poly = [(qx + s * dx * hl + u * nx * hw, qy + s * dy * hl + u * ny * hw) for s, u in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        mask = Image.new("L", (1024, 1024), 0); ImageDraw.Draw(mask).polygon(poly, fill=255)
        g = torch.Generator("cuda").manual_seed(a.seed + i); i += 1
        gen = pipe(prompt=rng.choice(PROMPTS), negative_prompt=NEG, image=crop, mask_image=mask, num_inference_steps=40,
                   strength=0.99, guidance_scale=8.0, generator=g).images[0].resize((1024, 1024))
        merged = Image.composite(gen, crop, mask.filter(ImageFilter.GaussianBlur(4)))
        diff = np.abs(np.asarray(merged, float) - np.asarray(crop, float)).mean(2) * (np.asarray(mask) > 0)
        ys, xs = np.where(diff > 30)
        if len(xs) < 300: continue
        bx = [np.percentile(xs, 1), np.percentile(ys, 1), np.percentile(xs, 99), np.percentile(ys, 99)]
        full = img.copy(); full.paste(merged.resize((side, side), Image.LANCZOS), (cx0, cy0))
        fb = [cx0 + bx[0] / k, cy0 + bx[1] / k, cx0 + bx[2] / k, cy0 + bx[3] / k]
        stem = f"dbs_{len(tiles):04d}_{Path(name).stem}"; full.save(out / "images" / f"{stem}.jpg", quality=95)
        (out / "labels" / f"{stem}.txt").write_text("\n".join(rows + [f"5 {(fb[0] + fb[2]) / 2 / W:.6f} {(fb[1] + fb[3]) / 2 / H:.6f} "
                                                                     f"{(fb[2] - fb[0]) / W:.6f} {(fb[3] - fb[1]) / H:.6f}"]) + "\n")
        a1 = crop.resize((320, 320)); a2 = merged.copy(); ImageDraw.Draw(a2).rectangle(bx, outline=(255, 0, 0), width=4)
        tiles.append((a1, a2.resize((320, 320)), img.resize((320, 240))))
        meta.append({"name": stem, "src": name, "box": [round(v) for v in fb]}); print(len(tiles), name, flush=True)
    sh = Image.new("RGB", (4 * 640, 320 * ((len(tiles) + 3) // 4)), "white")
    for j, (o, m, _) in enumerate(tiles): sh.paste(o, ((j % 4) * 640, (j // 4) * 320)); sh.paste(m, ((j % 4) * 640 + 320, (j // 4) * 320))
    sh.save(out / "sheet.jpg", quality=85); (out / "meta.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    main()
