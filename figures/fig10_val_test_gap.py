"""Fig 10 - Generalisation gap: validation vs human-reviewed test vs raw independent test.

Same selected checkpoints in all three evaluations (results_ablation/finding_nemo/):
  selected_checkpoints_eval_val, selected_checkpoints_eval, selected_checkpoints_eval_test_main_v1
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common import (BLUE, DOUBLE_COL, GRID, INK_3, MODEL_LABEL, NEMO, ORANGE, SURFACE, apply_style,
                    eval_summary, save)

PROTOCOLS = [
    ("selected_checkpoints_eval_val", "Validation", INK_3, "o"),
    ("selected_checkpoints_eval", "Human-reviewed test", BLUE, "s"),
    ("selected_checkpoints_eval_test_main_v1", "test_main_v1 (independent)", ORANGE, "D"),
]


def main():
    apply_style()
    dfs = [eval_summary(NEMO / f).set_index("model") for f, *_ in PROTOCOLS]
    models = [m for m in dfs[0].index if all(m in d.index for d in dfs)]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.3), sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, met, title in zip(axes, ("mAP@0.5", "mAP@0.5:0.95", "MEAN_5"),
                              ("mAP@0.5 (%)", "mAP@0.5:0.95 (%)", "MEAN_5 defect AP@0.5 (%)")):
        vals = np.array([[d.loc[m, met] * 100 for d in dfs] for m in models])
        for yi, row in zip(y, vals):
            ax.plot([row.min(), row.max()], [yi, yi], color=GRID, lw=2.4, solid_capstyle="round", zorder=1)
        for k, (_, lab, col, mk) in enumerate(PROTOCOLS):
            ax.scatter(vals[:, k], y, s=30, color=col, marker=mk, edgecolor=SURFACE, linewidth=0.7,
                       zorder=3 + k, label=lab)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.margins(x=0.08)
    axes[0].set_yticks(y, [MODEL_LABEL[m] for m in models])
    handles = [Line2D([0], [0], linestyle="none", marker=mk, color=col, markeredgecolor=SURFACE,
                      markersize=6, label=lab) for _, lab, col, mk in PROTOCOLS]
    fig.legend(handles=handles, loc="upper left", ncol=3, bbox_to_anchor=(0.07, 1.07), handletextpad=0.2)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig10_val_test_gap")


if __name__ == "__main__":
    main()
