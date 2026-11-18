"""Fig 17 - Deployment cost of the six detectors: params, GFLOPs, latency, FPS, VRAM.

Source: results_ablation/finding_nemo/selected_checkpoints_eval_test_main_v1/benchmark_summary.csv
(latency measured by the unified evaluator, batch 1, same GPU for all models).
"""
import matplotlib.pyplot as plt
import numpy as np

from common import (DOUBLE_COL, INK, MODEL_LABEL, NEMO, SURFACE, apply_style, eval_summary, model_color,
                    MODEL_SIZE, save)
from helpers import caption as add_caption

PANELS = [("Params (M)", "Parameters (M)", "{:.1f}"), ("GFLOPS", "GFLOPs", "{:.0f}"),
          ("Latency (ms)", "Latency (ms)", "{:.1f}"), ("FPS", "Throughput (FPS)", "{:.0f}"),
          ("VRAM (MB)", "Peak VRAM (MB)", "{:.0f}")]


def main():
    apply_style()
    df = eval_summary(NEMO / "selected_checkpoints_eval_test_main_v1")
    y = np.arange(len(df))[::-1]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(DOUBLE_COL, 2.0), sharey=True)
    for ax, (col, title, fmt) in zip(axes, PANELS):
        vals = df[col].values
        for yi, (_, r), v in zip(y, df.iterrows(), vals):
            c = model_color(r["model"])
            small = MODEL_SIZE[r["model"]] == "small"
            ax.barh(yi, v, height=0.62, color=SURFACE if small else c, edgecolor=c, linewidth=1.2,
                    hatch="//////" if small else None)
            ax.text(v + vals.max() * 0.03, yi, fmt.format(v), va="center", fontsize=6.3, color=INK)
        ax.set_xlim(0, vals.max() * 1.38)
        ax.set_title(title, fontsize=7.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    axes[0].set_yticks(y, [MODEL_LABEL[m] for m in df["model"]])
    plt.rcParams["hatch.linewidth"] = 0.5
    fig.tight_layout(w_pad=0.5)
    add_caption(fig, "Hatched = small variant (n / R50). Latency, FPS and VRAM from the unified evaluator on test_main_v1.")
    save(fig, "fig17_hardware_profile")


if __name__ == "__main__":
    main()
