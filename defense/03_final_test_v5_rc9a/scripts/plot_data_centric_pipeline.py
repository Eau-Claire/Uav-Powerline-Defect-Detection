#!/usr/bin/env python3
"""Create the publication-ready data-centric benchmark pipeline figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "finding_nemo" / "operational_min8_v5" / "figures"
OUTPUT_STEM = OUT_DIR / "figure_data_centric_pipeline"

NAVY = "#17365d"
BLUE = "#dcebf7"
GREEN = "#dff1e6"
ORANGE = "#fbe6d4"
PURPLE = "#e9e1f4"
GRAY = "#eef1f4"
RED = "#a33a32"


def box(
    ax,
    x,
    y,
    width,
    height,
    title,
    subtitle,
    facecolor,
    edgecolor=NAVY,
    title_size=11.2,
    subtitle_size=8.8,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.66,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        x + width / 2,
        y + height * 0.32,
        subtitle,
        ha="center",
        va="center",
        fontsize=subtitle_size,
        color="#333333",
        linespacing=1.25,
    )


def arrow(ax, start, end, color=NAVY, style="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            linestyle=style,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(15.5, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "Data-centric benchmark and controlled evaluation pipeline",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=NAVY,
    )

    top_y, w, h = 0.72, 0.145, 0.15
    xs = [0.02, 0.185, 0.35, 0.515, 0.68, 0.845]
    stages = [
        ("1. Ingestion", "Six public datasets\nVOC · COCO · YOLO", GRAY),
        ("2. Harmonization", "Six-class taxonomy\nformat conversion", BLUE),
        ("3. Safeguards", "Deduplication\ngeometric validation", GREEN),
        ("4. Curation", "Eight rules · four modules\ntraining partition only", ORANGE),
        ("5. Paired training", "Raw vs Curated\nsix architectures", PURPLE),
        ("6. Evaluation", "Shared annotations\naccuracy · class · efficiency", BLUE),
    ]
    for x, (title, subtitle, color) in zip(xs, stages):
        box(ax, x, top_y, w, h, title, subtitle, color, title_size=9.7, subtitle_size=7.8)
    for left, right in zip(xs[:-1], xs[1:]):
        arrow(ax, (left + w, top_y + h / 2), (right, top_y + h / 2))

    ax.text(
        0.5,
        0.655,
        "Eight-rule curation modules",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
    )
    module_y, module_w, module_h = 0.43, 0.21, 0.15
    module_xs = [0.055, 0.285, 0.515, 0.745]
    modules = [
        ("Module I · A, B, F", "Flip correction\nmissing-target recovery\ntiny-box pruning", BLUE),
        ("Module II · C, D", "Person/object disambiguation\nsynthetic-artifact filtering", ORANGE),
        ("Module III · E, G", "Quarantine\ncontact-sheet human review", GREEN),
        ("Module IV · H", "SAM 2.1-assisted\nbounding-box refinement", PURPLE),
    ]
    for x, (title, subtitle, color) in zip(module_xs, modules):
        box(ax, x, module_y, module_w, module_h, title, subtitle, color)
    arrow(ax, (0.587, top_y), (0.50, 0.655), color="#b46a24", style="--")

    selection_y = 0.17
    box(
        ax,
        0.13,
        selection_y,
        0.29,
        0.13,
        "Validation checkpoint selection",
        "Primary selection metric: validation mAP50\nno test metrics used for selection",
        GRAY,
    )
    box(
        ax,
        0.58,
        selection_y,
        0.29,
        0.13,
        "Finding Nemo evaluation roles",
        "test_main_v1: primary independent benchmark\nOperational min-8 V5: exploratory, model-assisted",
        GRAY,
    )

    ax.text(
        0.5,
        0.075,
        "Frozen reference annotations are shared across compared models within each evaluation protocol.",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=RED,
    )
    ax.text(
        0.5,
        0.038,
        "Curation never modifies validation or test annotations used within a controlled comparison.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
