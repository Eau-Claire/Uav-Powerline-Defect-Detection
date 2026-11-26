#!/usr/bin/env python3
"""Build the publication-ready RC-3 per-class Raw/Curated V5 figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "finding_nemo" / "operational_min8_v5"
INPUT_CSV = RESULT_DIR / "per_class_summary.csv"
OUTPUT_STEM = RESULT_DIR / "figures" / "figure_rc3_per_class_raw_curated_v5"

MODEL_ORDER = [
    "yolo11n",
    "yolo11m",
    "yolo26n",
    "yolo26m",
    "rtdetr50",
    "rtdetr101",
]
MODEL_LABELS = [
    "YOLO11n",
    "YOLO11m",
    "YOLO26n",
    "YOLO26m",
    "RT-DETR-R50",
    "RT-DETR-R101",
]
CLASSES = [
    "insulator",
    "insulator_broken",
    "insulator_flashover",
    "bird_nest",
    "foreign_object",
    "broken_strand",
]
CLASS_LABELS = [
    "Insulator",
    "Broken\ninsulator",
    "Flashover",
    "Bird nest",
    "Foreign\nobject",
    "Broken\nstrand",
]


def matrix(frame: pd.DataFrame, suffix: str) -> np.ndarray:
    indexed = frame.set_index("Model").reindex(MODEL_ORDER)
    return indexed[[f"{name}_{suffix}" for name in CLASSES]].to_numpy(dtype=float)


def annotate_heatmap(ax: plt.Axes, values: np.ndarray) -> None:
    magnitude = max(abs(values.min()), abs(values.max()))
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            color = "white" if abs(value) > magnitude * 0.55 else "#171717"
            ax.text(
                col,
                row,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=9.3,
                fontweight="semibold",
                color=color,
            )


def draw_delta_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    norm: TwoSlopeNorm,
) -> None:
    image = ax.imshow(values, cmap="RdBu", norm=norm, aspect="auto")
    annotate_heatmap(ax, values)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=12)
    ax.set_xticks(range(len(CLASS_LABELS)), CLASS_LABELS, fontsize=9.5)
    ax.set_yticks(range(len(MODEL_LABELS)), MODEL_LABELS, fontsize=10.5)
    ax.tick_params(length=0)
    for edge in np.arange(-0.5, len(CLASSES), 1):
        ax.axvline(edge, color="white", linewidth=2)
    for edge in np.arange(-0.5, len(MODEL_ORDER), 1):
        ax.axhline(edge, color="white", linewidth=2)
    return image


def draw_yolo26m_bars(
    ax: plt.Axes,
    raw: np.ndarray,
    curated: np.ndarray,
    title: str,
) -> None:
    x = np.arange(len(CLASSES))
    width = 0.36
    raw_pct = raw * 100
    curated_pct = curated * 100
    bars_raw = ax.bar(
        x - width / 2,
        raw_pct,
        width,
        label="Raw",
        color="#8a9baa",
        edgecolor="white",
        linewidth=0.8,
    )
    bars_curated = ax.bar(
        x + width / 2,
        curated_pct,
        width,
        label="Curated",
        color="#1775c1",
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=12)
    ax.set_xticks(x, CLASS_LABELS, fontsize=9.5)
    ax.set_ylabel("AP (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#dfdfdf", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    for bars in (bars_raw, bars_curated):
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8.5, rotation=90)


def main() -> None:
    data = pd.read_csv(INPUT_CSV)
    raw = data[data["Train"] == "Raw"]
    curated = data[data["Train"] == "Curated"]
    if len(raw) != 6 or len(curated) != 6:
        raise ValueError("Expected six Raw and six Curated model rows")

    raw_ap50 = matrix(raw, "AP50")
    curated_ap50 = matrix(curated, "AP50")
    raw_ap5095 = matrix(raw, "AP50_95")
    curated_ap5095 = matrix(curated, "AP50_95")
    delta_ap50 = (curated_ap50 - raw_ap50) * 100
    delta_ap5095 = (curated_ap5095 - raw_ap5095) * 100

    bound = max(np.abs(delta_ap50).max(), np.abs(delta_ap5095).max())
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 160})
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    image = draw_delta_heatmap(
        axes[0, 0], delta_ap50, "(a) Change in AP@0.5 (Curated − Raw, pp)", norm
    )
    draw_delta_heatmap(
        axes[0, 1],
        delta_ap5095,
        "(b) Change in AP@0.5:0.95 (Curated − Raw, pp)",
        norm,
    )
    colorbar = fig.colorbar(image, ax=axes[0, :], shrink=0.82, pad=0.02)
    colorbar.set_label("Percentage-point change", fontsize=11)

    yolo26m_index = MODEL_ORDER.index("yolo26m")
    draw_yolo26m_bars(
        axes[1, 0],
        raw_ap50[yolo26m_index],
        curated_ap50[yolo26m_index],
        "(c) YOLO26m class-level AP@0.5",
    )
    draw_yolo26m_bars(
        axes[1, 1],
        raw_ap5095[yolo26m_index],
        curated_ap5095[yolo26m_index],
        "(d) YOLO26m class-level AP@0.5:0.95",
    )

    fig.suptitle(
        "Per-class effects of training-data curation on Finding Nemo operational min-8 V5",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.012,
        "Common evaluation set: 1,871 images, 2,467 boxes. "
        "Exploratory model-assisted V5 annotations; not the primary independent test.",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
