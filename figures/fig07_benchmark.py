"""Fig 7-9, 12 - Six-detector benchmark of the selected Finding Nemo checkpoints.

One evaluation folder -> four figures:
  fig07_benchmark_<tag>      dot plot of P / R / F1 / mAP50 / mAP50-95 / MEAN_5
  fig08_tradeoff_<tag>       accuracy vs latency and vs GFLOPs
  fig09_per_class_<tag>      per-class AP50 and AP50-95 heatmaps
  fig12_pr_points_<tag>      precision-recall operating points with iso-F1 curves

Default protocols (all under results_ablation/finding_nemo/):
  test_main_v1      selected_checkpoints_eval_test_main_v1  (primary independent test)
  reviewed_test     selected_checkpoints_eval               (human-reviewed 6-class test)
  val               selected_checkpoints_eval_val           (validation, for reference)
  op_min8_v5        operational_min8_v5_all                 (exploratory, model-assisted)

    python figures/fig07_benchmark.py                 # all protocols
    python figures/fig07_benchmark.py --only test_main_v1
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import (CLASSES, CLASS_LABEL, DOUBLE_COL, GRID, INK, INK_2, MODEL_LABEL, NEMO, SINGLE_COL,
                    apply_style, eval_summary, model_legend_handles, model_marker, save)
from helpers import caption as add_caption, heatmap, place_labels

PROTOCOLS = {
    "test_main_v1": ("selected_checkpoints_eval_test_main_v1", "test_main_v1 (1,931 images, independent)"),
    "reviewed_test": ("selected_checkpoints_eval", "Human-reviewed 6-class test"),
    "val": ("selected_checkpoints_eval_val", "Validation"),
    "op_min8_v5": ("operational_min8_v5_all", "Operational min-8 v5 (exploratory, model-assisted)"),
}

METRICS = [("Precision", "Precision"), ("Recall", "Recall"), ("F1-Score", "F1"),
           ("mAP@0.5", "mAP@0.5"), ("mAP@0.5:0.95", "mAP@0.5:0.95"), ("MEAN_5", "MEAN_5 (defects)")]


def fig_dotplot(df, tag, caption):
    """Small multiples: one panel per metric, own x-range, models as rows."""
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 3.3), sharey=True)
    y = np.arange(len(df))[::-1]
    for ax, (col, lab) in zip(axes.flat, METRICS):
        v = df[col].values * 100
        for yi, (_, r), vi in zip(y, df.iterrows(), v):
            ax.plot([v.min() - 1.5, vi], [yi, yi], color=GRID, lw=1, zorder=1)
            ax.plot(vi, yi, linestyle="none", zorder=3, **model_marker(r["model"]))
            best = vi == v.max()
            ax.text(vi + 0.5, yi, f"{vi:.1f}", va="center", fontsize=6.3,
                    color=INK, fontweight="bold" if best else "normal")
        ax.set_xlim(v.min() - 1.5, v.max() + 3.2)
        ax.set_title(f"{lab} (%)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    for ax in axes[:, 0]:
        ax.set_yticks(y, [MODEL_LABEL[m] for m in df["model"]])
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    add_caption(fig, caption)
    save(fig, f"fig07_benchmark_{tag}")


def fig_tradeoff(df, tag, caption):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.4), sharey=True)
    for ax, xcol, xlabel, logx in ((axes[0], "Latency (ms)", "Latency per image (ms, batch 1)", False),
                                   (axes[1], "GFLOPS", "GFLOPs (log scale)", True)):
        for _, r in df.iterrows():
            ax.plot(r[xcol], r["mAP@0.5:0.95"] * 100, linestyle="none", zorder=3, **model_marker(r["model"]))
        ax.set_xlabel(xlabel)
        if logx:
            ax.set_xscale("log")
        ax.margins(x=0.22, y=0.15)
        place_labels(ax, df[xcol].values, df["mAP@0.5:0.95"].values * 100,
                     [MODEL_LABEL[m] for m in df["model"]])
    axes[0].set_ylabel("mAP@0.5:0.95 (%)")
    axes[0].set_title("Accuracy vs latency")
    axes[1].set_title("Accuracy vs compute")
    fig.tight_layout(w_pad=1.2)
    add_caption(fig, caption)
    save(fig, f"fig08_tradeoff_{tag}")


def fig_per_class(df, tag, caption):
    rows = [MODEL_LABEL[m] for m in df["model"]]
    cols = [CLASS_LABEL[c] for c in CLASSES]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))
    for ax, suf, title in ((axes[0], "AP50", "AP@0.5 (%)"), (axes[1], "AP50_95", "AP@0.5:0.95 (%)")):
        heatmap(ax, df[[f"{c}_{suf}" for c in CLASSES]].values, rows, cols, fig=fig)
        ax.set_title(title)
    fig.tight_layout(w_pad=1.5)
    add_caption(fig, caption)
    save(fig, f"fig09_per_class_{tag}")


def fig_pr_points(df, tag, caption):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.8))
    p, r = df["Precision"].values * 100, df["Recall"].values * 100
    lo, hi = min(p.min(), r.min()) - 4, max(p.max(), r.max()) + 4
    rr = np.linspace(lo, hi, 200)
    for f1 in np.arange(0.60, 0.95, 0.05) * 100:
        pp = f1 * rr / (2 * rr - f1)
        ok = (pp > 0) & (pp <= 100)
        ax.plot(rr[ok], pp[ok], color=GRID, lw=0.9, zorder=1)
        k = np.argmin(np.abs(pp - hi + 1.5)) if np.any(ok) else None
        if k is not None and lo < rr[k] < hi:
            ax.text(rr[k], hi - 1.2, f"F1={f1:.0f}", fontsize=6, color=INK_2, ha="center")
    for _, row in df.iterrows():
        ax.plot(row["Recall"] * 100, row["Precision"] * 100, linestyle="none", zorder=3, **model_marker(row["model"]))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    place_labels(ax, r, p, [MODEL_LABEL[m] for m in df["model"]])
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Precision–recall operating points")
    fig.tight_layout()
    add_caption(fig, caption)
    save(fig, f"fig12_pr_points_{tag}")


def main(only=None):
    apply_style()
    for tag, (folder, caption) in PROTOCOLS.items():
        if only and tag != only:
            continue
        path = NEMO / folder
        if not (path / "benchmark_summary.csv").exists():
            print(f"  skip {tag}: {path} missing")
            continue
        df = eval_summary(path)
        caption = f"Evaluated on: {caption}"
        fig_dotplot(df, tag, caption)
        fig_tradeoff(df, tag, caption)
        if "insulator_AP50" in df:
            fig_per_class(df, tag, caption)
        fig_pr_points(df, tag, caption)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(PROTOCOLS))
    main(ap.parse_args().only)
