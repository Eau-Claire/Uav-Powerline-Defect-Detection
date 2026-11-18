"""Fig 4 - Data-centric gain: Raw vs Curated training, six architectures (seed 42).

Evaluated on val_clean (dataset_curated.yaml maps `test` -> images/val_clean).
Sources: results_yolo_model/ and results_rtdetr_best/ benchmark_test_results.json.
Fig 5 - Per-class AP change (Curated - Raw), same runs.
"""
import matplotlib.pyplot as plt
import pandas as pd

from common import (BENCH, CLASSES, CLASS_LABEL, DOUBLE_COL, MODEL_LABEL, MODEL_ORDER, apply_style,
                    benchmark_json, save)
from helpers import caption as add_caption, dumbbell, heatmap


def load():
    df = pd.concat([benchmark_json(BENCH / "results_yolo_model" / "benchmark_test_results.json"),
                    benchmark_json(BENCH / "results_rtdetr_best" / "benchmark_test_results.json")])
    df = df[df["model"].isin(MODEL_ORDER)]
    raw = df[df.regime == "raw"].set_index("model").reindex(MODEL_ORDER).dropna(how="all")
    cur = df[df.regime == "curated"].set_index("model").reindex(raw.index)
    return raw, cur


def fig_overall(raw, cur):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.1), sharey=True)
    labels = [MODEL_LABEL[m] for m in raw.index]
    for ax, met, title in zip(axes, ("map50", "map50_95", "mean5"),
                              ("mAP@0.5 (%)", "mAP@0.5:0.95 (%)", "MEAN_5 defect AP@0.5 (%)")):
        dumbbell(ax, labels, raw[met].values, cur[met].values)
        ax.set_title(title)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.13), ncol=2, handletextpad=0.2)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig04_raw_vs_curated_val_clean")


def fig_per_class(raw, cur):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.4))
    rows = [MODEL_LABEL[m] for m in raw.index]
    cols = [CLASS_LABEL[c] for c in CLASSES]
    for ax, suffix, title in zip(axes, ("ap50", "ap50_95"), ("Δ AP@0.5 (pp)", "Δ AP@0.5:0.95 (pp)")):
        d = cur[[f"{c}_{suffix}" for c in CLASSES]].values - raw[[f"{c}_{suffix}" for c in CLASSES]].values
        heatmap(ax, d, rows, cols, fmt="{:+.1f}", diverging=True, fig=fig)
        ax.set_title(f"{title}, Curated − Raw")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig05_per_class_gain_val_clean")


def fig_per_class_raw_curated(raw, cur):
    """Absolute per-class scores for Raw and Curated in one directly comparable figure."""
    rows = [MODEL_LABEL[m] for m in raw.index]
    cols = [CLASS_LABEL[c] for c in CLASSES]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.8))
    for j, (suffix, metric) in enumerate((("ap50", "AP@0.5"),
                                          ("ap50_95", "AP@0.5:0.95"))):
        columns = [f"{c}_{suffix}" for c in CLASSES]
        raw_values = raw[columns].values
        curated_values = cur[columns].values
        # One scale per metric across both regimes, so colour intensity is comparable vertically.
        vmin = min(raw_values.min(), curated_values.min()) * 100
        vmax = max(raw_values.max(), curated_values.max()) * 100
        for i, (regime, values) in enumerate((("Raw", raw_values), ("Curated", curated_values))):
            heatmap(axes[i, j], values, rows, cols, fig=fig, vmin=vmin, vmax=vmax,
                    cbar_label=f"{metric} (%)")
            axes[i, j].set_title(f"{regime} training — {metric} (%)")
    fig.tight_layout(h_pad=1.5, w_pad=1.4)
    add_caption(fig, "Evaluated on: val_clean (same frozen split for Raw and Curated)")
    save(fig, "fig05_per_class_raw_curated_val_clean")


def main():
    apply_style()
    raw, cur = load()
    fig_overall(raw, cur)
    fig_per_class_raw_curated(raw, cur)
    fig_per_class(raw, cur)


if __name__ == "__main__":
    main()
