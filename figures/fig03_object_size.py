"""Fig 3 - Object size per class: short box side in pixels after letterboxing to 640.

Explains why broken_strand / foreign_object are hard and motivates the 8 px
operational rule. Reads YOLO labels + image headers (PIL, no decoding).

    python figures/fig03_object_size.py                     # test_main_v1
    python figures/fig03_object_size.py --split curated_val_clean
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import (BLUE, CLASSES, CLASS_LABEL, DATA, DOUBLE_COL, INK_2, apply_style, panel_title,
                    save, yolo_label_rows)

SPLITS = {
    "test_main_v1": (DATA / "test_main_v1" / "labels" / "test", DATA / "test_main_v1" / "images" / "test"),
    "curated_val_clean": (DATA / "curated_combined" / "labels" / "val_clean",
                          DATA / "curated_combined" / "images" / "val_clean"),
    "curated_train": (DATA / "curated_combined" / "labels" / "train",
                      DATA / "curated_combined" / "images" / "train"),
}
IMGSZ = 640


def short_sides(label_dir, image_dir):
    sizes, out = {}, {c: [] for c in range(len(CLASSES))}
    for stem, cls, _xc, _yc, w, h in yolo_label_rows(label_dir):
        if stem not in sizes:
            img = next((p for ext in (".jpg", ".jpeg", ".png", ".JPG") if (p := image_dir / f"{stem}{ext}").exists()), None)
            if img is None:
                sizes[stem] = None
                continue
            with Image.open(img) as im:
                sizes[stem] = im.size
        if sizes[stem] is None or cls not in out:
            continue
        W, H = sizes[stem]
        s = IMGSZ / max(W, H)
        out[cls].append(min(w * W, h * H) * s)
    return {c: np.asarray(v) for c, v in out.items()}


def main(split="test_main_v1"):
    apply_style()
    data = short_sides(*SPLITS[split])
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 3.5), sharex=True)
    bins = np.logspace(np.log10(1), np.log10(IMGSZ), 36)
    for ax, (cid, cname) in zip(axes.flat, enumerate(CLASSES)):
        v = data[cid]
        ax.hist(v, bins=bins, color=BLUE, edgecolor="white", linewidth=0.5)
        ax.set_xscale("log")
        for thr in (8, 32):
            ax.axvline(thr, color=INK_2, lw=0.8, ls=(0, (3, 2)))
        lt8 = (v < 8).mean() * 100 if len(v) else 0
        lt32 = (v < 32).mean() * 100 if len(v) else 0
        med = np.median(v) if len(v) else 0
        panel_title(ax, f"{CLASS_LABEL[cname]} (n={len(v):,})",
                    f"median {med:.0f} px · <8 px {lt8:.1f}% · <32 px {lt32:.1f}%")
        ax.grid(axis="x", visible=False)
    for ax in axes[1]:
        ax.set_xticks([2, 8, 32, 128, 512], ["2", "8", "32", "128", "512"])
    fig.supxlabel(f"Short side of box after letterbox to {IMGSZ} px (log scale)", fontsize=8, y=0.0)
    fig.supylabel("Boxes", fontsize=8, x=0.0)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, f"fig03_object_size_{split}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_main_v1", choices=list(SPLITS))
    main(ap.parse_args().split)
