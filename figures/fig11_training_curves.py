"""Fig 11 - Finding Nemo training curves for the six detectors (50 epochs).

Source: results_ablation/finding_nemo/final_comparison/finding_nemo_all_epochs.csv.
Validation mAP panels include RT-DETR; loss panels are YOLO only because the
RT-DETR log reuses Ultralytics column names with different loss definitions.
The star marks the row flagged `selected_best_row`.
"""
import matplotlib.pyplot as plt

from common import (DOUBLE_COL, INK, MODEL_FAMILY, MODEL_LABEL, MODEL_ORDER, NEMO, SURFACE, apply_style,
                    model_color, model_legend_handles, model_linestyle, norm_model, read_csv, save)

PANELS = [
    ("metrics/mAP50(B)", "Validation mAP@0.5 (%)", True, 100),
    ("metrics/mAP50-95(B)", "Validation mAP@0.5:0.95 (%)", True, 100),
    ("train/box_loss", "Train box loss (YOLO)", False, 1),
    ("val/box_loss", "Validation box loss (YOLO)", False, 1),
]


def main():
    apply_style()
    df = read_csv(NEMO / "final_comparison" / "finding_nemo_all_epochs.csv")
    df["model"] = df["model"].map(norm_model)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.0), sharex=True)
    for ax, (col, title, with_detr, scale) in zip(axes.flat, PANELS):
        for m in MODEL_ORDER:
            if not with_detr and MODEL_FAMILY[m] == "RT-DETR":
                continue
            d = df[df.model == m].sort_values("epoch")
            if d.empty or d[col].isna().all():
                continue
            ax.plot(d["epoch"], d[col] * scale, color=model_color(m), ls=model_linestyle(m), lw=1.3)
            best = d[d["selected_best_row"].astype(str).str.lower() == "true"]
            if not best.empty:
                b = best.iloc[0]
                ax.plot(b["epoch"], b[col] * scale, marker="*", markersize=8, color=model_color(m),
                        markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=5)
        ax.set_title(title)
        if with_detr:
            ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    fig.legend(handles=model_legend_handles(), loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.04),
               columnspacing=1.0, handletextpad=0.2)
    fig.text(0.995, 0.0, "★ = selected checkpoint row; dashed = small variant (n / R50), solid = large (m / R101)",
             ha="right", va="top", fontsize=6.3, color=INK)
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    save(fig, "fig11_training_curves")


if __name__ == "__main__":
    main()
