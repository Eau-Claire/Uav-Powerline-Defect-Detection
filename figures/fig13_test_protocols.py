"""Fig 13 - How the score of each detector moves across test protocols / audit versions.
Fig 14 - YOLO26m per-class AP@0.5 across operational min-8 audit versions (v1 -> v5).

All folders under results_ablation/finding_nemo/. v3-v5 are model-assisted
exploratory label audits and are shaded to keep them visually separate from the
independent test.
"""
import matplotlib.pyplot as plt
import numpy as np

from common import (CLASSES, CLASS_LABEL, DOUBLE_COL, INK_2, MODEL_LABEL, MODEL_ORDER, NEMO,
                    apply_style, eval_summary, model_color, model_legend_handles, model_linestyle,
                    model_marker, save)
from helpers import heatmap

PROTOCOLS = [
    ("selected_checkpoints_eval_test_main_v1", "Main\nv1", False),
    ("selected_checkpoints_eval", "Re-\nviewed", False),
    ("operational_min8_v3_all", "Op.\nv3", True),
    ("operational_min8_v4_all", "Op.\nv4", True),
    ("operational_min8_v5_all", "Op.\nv5", True),
]

YOLO26M_VERSIONS = [
    ("operational_min8_yolo26m", "v1 (original)"),
    ("operational_min8_v3_all", "v3 (+broken_strand fix)"),
    ("operational_min8_v4_all", "v4 (partial consensus)"),
    ("operational_min8_v5_all", "v5 (+missing insulators)"),
]


def fig_protocols():
    tables = [eval_summary(NEMO / f).set_index("model") for f, *_ in PROTOCOLS]
    x = np.arange(len(PROTOCOLS))
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.6))
    for ax, met, title in zip(axes, ("mAP@0.5", "mAP@0.5:0.95", "MEAN_5"),
                              ("mAP@0.5 (%)", "mAP@0.5:0.95 (%)", "MEAN_5 (%)")):
        first_expl = next(i for i, (*_, e) in enumerate(PROTOCOLS) if e)
        ax.axvspan(first_expl - 0.4, len(PROTOCOLS) - 0.6, color="#f3f2ee", lw=0, zorder=0)
        for m in MODEL_ORDER:
            ys = [t.loc[m, met] * 100 if m in t.index else np.nan for t in tables]
            mk = model_marker(m)
            mk["markersize"] = 4.5
            ax.plot(x, ys, color=model_color(m), ls=model_linestyle(m), lw=1.1, **{k: v for k, v in mk.items() if k != "color"})
        ax.set_xticks(x, [lab for _, lab, _ in PROTOCOLS], fontsize=6.5)
        ax.set_xlim(-0.4, len(PROTOCOLS) - 0.6)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    for ax in axes:
        ax.text((first_expl + len(PROTOCOLS) - 1) / 2, 1.0, "exploratory", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6, color=INK_2)
    fig.text(0.995, 0.0, "Main v1 = test_main_v1 (independent); Reviewed = human-reviewed 6-class test; "
             "Op. v3–v5 = operational min-8 subsets with model-assisted label audits",
             ha="right", va="top", fontsize=6.0, color=INK_2)
    fig.legend(handles=model_legend_handles(), loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.06),
               columnspacing=1.0, handletextpad=0.2)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig13_test_protocols")


def fig_yolo26m_versions():
    cols, labels = [], []
    for folder, lab in YOLO26M_VERSIONS:
        t = eval_summary(NEMO / folder).set_index("model")
        if "yolo26m" not in t.index:
            continue
        r = t.loc["yolo26m"]
        cols.append([r[f"{c}_AP50"] for c in CLASSES] + [r["mAP@0.5"], r["mAP@0.5:0.95"], r["MEAN_5"]])
        labels.append(lab.split(" ", 1)[0])
    rows = [CLASS_LABEL[c] for c in CLASSES] + ["mAP@0.5", "mAP@0.5:0.95", "MEAN_5"]
    m = np.array(cols).T  # metrics x versions
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL * 0.8, 2.9), gridspec_kw={"width_ratios": [4, 3]})
    heatmap(axes[0], m, rows, labels, fig=fig, vmin=40, vmax=100)
    axes[0].set_title("YOLO26m score (%)")
    heatmap(axes[1], m[:, 1:] - m[:, :1], rows, labels[1:], fmt="{:+.1f}", diverging=True, fig=fig)
    axes[1].set_yticklabels([])
    axes[1].set_title("Change vs v1 (pp)")
    for ax in axes:
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=0, ha="center")
    fig.text(0.995, 0.0, "Operational min-8 subsets. " + "; ".join(l for _, l in YOLO26M_VERSIONS),
             ha="right", va="top", fontsize=6.0, color=INK_2, wrap=True)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig14_audit_yolo26m_per_class")


def main():
    apply_style()
    fig_protocols()
    fig_yolo26m_versions()


if __name__ == "__main__":
    main()
