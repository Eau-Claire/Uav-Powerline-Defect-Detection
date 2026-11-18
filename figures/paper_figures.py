"""Figures for paper_draft/main_with_ablation.tex -> figures/out/07_paper/ (+ copied to paper_draft/figures/).

paper_workflow          Sec. 3.1: the five-stage data-centric workflow, as listed in the paper
paper_curation_modules  Sec. 3.4: Modules I-IV, rules A-H and the thresholds stated in the paper
paper_ablation          Sec. 5.4: YOLO26m module-level ablation on val_clean
                        (a) overall mAP, (b) class-wise AP50 as absolute values

The ablation numbers are those of figures/data/loo_val_clean.csv (transcribed from
docs/RESEARCH_REPORT_LOO_CURATION_ABLATION.md, tables 3.1 and 3.2). Absolute values
are drawn on purpose: the reference run (Modules I+II) differs from the LOO runs by
more than one module, so deltas against it would suggest single-module effects.
"""
import shutil

import matplotlib.pyplot as plt
import numpy as np

from common import (BLUE, CLASSES, CLASS_LABEL, DOUBLE_COL, GRID, INK, INK_2, LOCAL_DATA, ORANGE,
                    RAW_GRAY, ROOT, SURFACE, apply_style, read_csv, save)
import diagrams
from diagrams import anchor, arrow, box, canvas
from helpers import heatmap

PAPER_FIG_DIR = ROOT / "paper_draft" / "figures"


def publish(name):
    paths = save(plt.gcf(), name)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copy2(p, PAPER_FIG_DIR / p.name)


# --------------------------------------------------------------------------- workflow
def fig_workflow():
    fig, ax = canvas(7.16, 1.55)
    steps = [
        ("1 · Ingestion", ["VOC XML, COCO JSON,", "YOLO txt → YOLO", "six-class taxonomy"], "data"),
        ("2 · Deduplication", ["pHash / dHash", "near-duplicate", "Hamming d ≤ 4"], "process"),
        ("3 · Geometric check", ["valid coordinates,", "positive area,", "flag w/h > 100"], "process"),
        ("4 · 8-rule curation", ["Modules I–IV", "training split only", "human-verified"], "process"),
        ("5 · Benchmarking", ["Raw vs Curated", "6 detectors, frozen", "val / test splits"], "eval"),
    ]
    w, g = 1.3, 0.135
    prev = None
    for i, (t, lines, role) in enumerate(steps):
        b = box(ax, 0.08 + i * (w + g), 1.47, w, 0.74, t, lines, role=role, body_size=6.0, title_size=6.9)
        if prev:
            arrow(ax, anchor(prev, "r"), anchor(b, "l"))
        prev = b
    ax.text(0.08, 0.6, "Six sources: Zheng et al., IDID, CPLID, VPMBGI, InsuFault-Net, FOLT-Drone  →  "
                        "22,618 images (train 18,334 · val 2,272 · test 2,012).",
            fontsize=6.2, color=INK_2, va="top")
    ax.text(0.08, 0.36, "Validation and test annotations are never modified by stage 4.",
            fontsize=6.2, color=INK_2, va="top", style="italic")
    publish("paper_workflow")


# --------------------------------------------------------------------------- curation
def fig_curation():
    fig, ax = canvas(7.16, 2.4)
    raw = box(ax, 0.08, 2.32, 0.88, 1.08, "Raw train", ["18,334 img", "25,807 boxes"], role="data")
    mods = [
        ("Module I", "Model-assisted audit", [
            "A  flip correction", "B  missing targets:", "    conf ≥ 0.60, IoU < 0.30,", "    human-verified",
            "F  drop boxes < 4×4 px"]),
        ("Module II", "Cross-model & synthetic", [
            "C  person vs. object:", "    IoU ≥ 0.50, conf ≥ 0.60,", "    human-reviewed",
            "D  purge implausible", "    synthetic objects"]),
        ("Module III", "Quarantine & review", [
            "E  quarantine", "    unsupported labels", "G  contact-sheet review:", "    keep / fix / add /",
            "    remove"]),
        ("Module IV", "SAM 2.1 refinement", [
            "H  tighten insulator", "    and foreign-object", "    boxes when", "    0.50 ≤ IoU < 0.85"]),
    ]
    w, g, x0 = 1.25, 0.1, 1.08
    prev = raw
    for i, (name, sub, lines) in enumerate(mods):
        b = box(ax, x0 + i * (w + g), 2.32, w, 1.08, name, [sub, ""] + lines, role="process", body_size=5.6)
        arrow(ax, anchor(prev, "r"), anchor(b, "l"))
        prev = b
    out = box(ax, x0 + 4 * (w + g), 2.32, 0.62, 1.08, "Curated", ["17,903", "img", "23,204", "boxes"], role="data",
              body_size=5.6)
    arrow(ax, anchor(prev, "r"), anchor(out, "l"))
    judge = box(ax, x0, 0.92, 2.64, 0.42, "YOLO11n judge (mAP50 = 0.7564)",
                ["predictions are evidence, not ground truth"], role="note", dashed=True, body_size=5.6)
    for i in (0, 1):
        bx = x0 + i * (w + g) + w / 2
        arrow(ax, (bx, judge["y"]), (bx, 1.24), dashed=True)
    sam = box(ax, x0 + 3 * (w + g), 0.92, w, 0.42, "SAM 2.1", ["segmentation proposals"], role="note",
              dashed=True, body_size=5.6)
    arrow(ax, (sam["x"] + w / 2, sam["y"]), (sam["x"] + w / 2, 1.24), dashed=True)
    ax.text(0.08, 0.3, "All rules operate on the training partition only. Image-level exclusions: "
                        "100 (C) + 239 (D) + 92 (E) = 431 images.",
            fontsize=6.0, color=INK_2, va="top", style="italic")
    publish("paper_curation_modules")


# --------------------------------------------------------------------------- ablation
def fig_ablation():
    d = read_csv(LOCAL_DATA / "loo_val_clean.csv")
    order = ["m2_c_d", "m_134", "m_124", "m_234"]
    labels = {"m2_c_d": "Reference (I+II)", "m_134": "w/o Module II", "m_124": "w/o Module III",
              "m_234": "w/o Module I"}
    d = d.set_index("stage").loc[order]
    colors = {"m2_c_d": RAW_GRAY, "m_134": BLUE, "m_124": ORANGE, "m_234": "#1baf7a"}
    markers = {"m2_c_d": "o", "m_134": "s", "m_124": "D", "m_234": "^"}

    fig = plt.figure(figsize=(DOUBLE_COL, 2.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.75], wspace=0.62)
    ax = fig.add_subplot(gs[0])
    metrics = [("map50", "mAP$_{50}$"), ("map50_95", "mAP$_{50-95}$")]
    y = np.arange(len(metrics))[::-1]
    offs = np.linspace(0.24, -0.24, len(order))
    for yi, (col, lab) in zip(y, metrics):
        ax.axhspan(yi - 0.38, yi + 0.38, color="#f6f5f2", lw=0, zorder=0)
        for off, k in zip(offs, order):
            v = d.loc[k, col] * 100
            ax.plot(v, yi + off, linestyle="none", marker=markers[k], color=colors[k], markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=0.7, zorder=3)
            ax.text(v + 0.5, yi + off, f"{v:.1f}", va="center", fontsize=6.0, color=INK)
    ax.set_yticks(y, [lab for _, lab in metrics])
    ax.set_xlim(41, 87)
    ax.set_xlabel("Score (%)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("(a) Overall, YOLO26m on val_clean")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], linestyle="none", marker=markers[k], color=colors[k],
                              markeredgecolor=SURFACE, markersize=6, label=labels[k]) for k in order],
              loc="upper left", bbox_to_anchor=(0, -0.28), ncol=2, fontsize=6.0, handletextpad=0.2,
              columnspacing=0.8)

    ax2 = fig.add_subplot(gs[1])
    cols = [f"{c}_ap50" for c in CLASSES]
    heatmap(ax2, d[cols].values, [labels[k] for k in order], [CLASS_LABEL[c] for c in CLASSES],
            fig=fig, vmin=55, vmax=100)
    ax2.set_title("(b) Class-wise AP$_{50}$ (%)")
    ax2.set_xticks(range(len(CLASSES)), ["Insulator", "Broken\ninsulator", "Flash-\nover", "Bird\nnest",
                                         "Foreign\nobject", "Broken\nstrand"], rotation=0, ha="center")
    fig.text(0.995, -0.1, "LOO runs each keep three of the four modules (Module IV always included); "
                           "the reference lacks Modules III and IV.",
             ha="right", va="top", fontsize=5.9, color=INK_2)
    publish("paper_ablation")


def monochrome_diagrams():
    """Black-and-white boxes for the method diagrams (print-safe, no hue)."""
    dark, mid = "#3d3d3b", "#6b6a66"
    diagrams.BORDER = mid
    for role in diagrams.ROLE:
        diagrams.ROLE[role] = (dark if role != "note" else mid, "#ffffff")


def main():
    apply_style()
    monochrome_diagrams()
    plt.rcParams["axes.grid"] = False
    fig_workflow()
    fig_curation()
    plt.rcParams["axes.grid"] = True
    fig_ablation()


if __name__ == "__main__":
    main()
