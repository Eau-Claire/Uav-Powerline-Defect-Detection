"""Fig 15 - Curation-module ablation of YOLO26m on the 1,931-image test set.
Fig 16 - Leave-one-out ablation on val_clean (numbers from the LOO research report).

Fig 15 source: results_ablation/finding_destiny_test_audit_v5/{summary.csv, */metrics.json}
  dori_m1_abf  -> Module 1 only (A flip fix, B missing adds, F tiny drop)
  dori_m2_cd   -> + Module 2 (C person drop, D synthetic purge)
  dori_m3_eg   -> + Module 3 (E quarantine, G non-grid drop)  = full curation
  destiny_m234 / m134 / m124 -> leave Module 1 / 2 / 3 out
Fig 16 source: figures/data/loo_val_clean.csv, transcribed from
  docs/RESEARCH_REPORT_LOO_CURATION_ABLATION.md tables 3.1 and 3.2.
"""
import matplotlib.pyplot as plt
import numpy as np

from common import (BENCH, BLUE, CLASSES, CLASS_LABEL, DOUBLE_COL, INK, INK_2, LOCAL_DATA,
                    ORANGE, SURFACE, apply_style, read_csv, read_json, save)
from helpers import caption as add_caption, heatmap

SRC = BENCH / "results_ablation" / "finding_destiny_test_audit_v5"
VARIANTS = [  # key, label, group
    ("dori_m1_abf", "M1 only", "Cumulative"),
    ("dori_m2_cd", "M1+M2", "Cumulative"),
    ("dori_m3_eg", "M1+M2+M3 (full)", "Cumulative"),
    ("destiny_m234", "w/o M1", "Leave-one-out"),
    ("destiny_m134", "w/o M2", "Leave-one-out"),
    ("destiny_m124", "w/o M3", "Leave-one-out"),
]
REFERENCE = "dori_m3_eg"
GROUP_COLOR = {"Cumulative": BLUE, "Leave-one-out": ORANGE}


def load_test():
    s = read_csv(SRC / "summary.csv").set_index("ablation")
    for k, *_ in VARIANTS:
        mj = SRC / k / "metrics.json"
        if mj.exists():
            for c, v in read_json(mj).get("per_class_map50_95", {}).items():
                s.loc[k, f"{c}_ap50_95"] = v
    s["f1"] = 2 * s.precision * s.recall / (s.precision + s.recall)
    return s


def bars(ax, labels, values, colors, ref=None, fmt="{:.1f}"):
    """Lollipops measured from the reference (dashed line); a truncated axis is fine
    here because no bar length is drawn from zero."""
    y = np.arange(len(labels))[::-1]
    ref = values[0] if ref is None else ref
    span = max(values.max(), ref) - min(values.min(), ref)
    lo, hi = min(values.min(), ref) - span * 0.25 - 0.3, max(values.max(), ref) + span * 0.45 + 0.6
    ax.set_xlim(lo, hi)
    ax.axvline(ref, color=INK_2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for yi, v, c in zip(y, values, colors):
        ax.plot([ref, v], [yi, yi], color=c, lw=1.6, solid_capstyle="round", zorder=2)
        ax.plot(v, yi, "o", color=c, markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
        ax.text(max(v, ref) + (hi - lo) * 0.04, yi, fmt.format(v), va="center", fontsize=6.4, color=INK)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def fig_test():
    s = load_test()
    keys = [k for k, *_ in VARIANTS if k in s.index]
    labels = [lab for k, lab, _ in VARIANTS if k in s.index]
    colors = [GROUP_COLOR[g] for k, _, g in VARIANTS if k in s.index]
    fig = plt.figure(figsize=(DOUBLE_COL, 3.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.65)
    for i, (col, title) in enumerate((("precision", "Precision"), ("recall", "Recall"),
                                      ("map50", "mAP@0.5"), ("map50_95", "mAP@0.5:0.95"))):
        ax = fig.add_subplot(gs[0, i])
        vals = s.loc[keys, col].values * 100
        bars(ax, labels if i == 0 else [""] * len(labels), vals, colors, ref=s.loc[REFERENCE, col] * 100)
        ax.set_title(f"{title} (%)")
    ax = fig.add_subplot(gs[1, :])
    cls_cols = [f"{c}_ap50_95" for c in CLASSES]
    if all(c in s.columns for c in cls_cols):
        delta = s.loc[keys, cls_cols].values - s.loc[REFERENCE, cls_cols].values
        overall = s.loc[keys, "map50_95"].values - s.loc[REFERENCE, "map50_95"]
        m = np.column_stack([delta, overall])
        heatmap(ax, m, labels, [CLASS_LABEL[c] for c in CLASSES] + ["mAP@0.5:0.95"], fmt="{:+.1f}",
                diverging=True, fig=fig)
        ax.set_title("Per-class AP@0.5:0.95 change vs full curation (pp)")
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=0, ha="center")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], marker="o", color=c, label=g, markeredgecolor=SURFACE)
                        for g, c in GROUP_COLOR.items()], loc="upper left",
               ncol=2, bbox_to_anchor=(0.06, 1.0))
    add_caption(fig, "YOLO26m, evaluated on the 1,931-image test set of the v5 audit bundle; "
                     "dashed line = full curation (M1+M2+M3)")
    save(fig, "fig15_curation_ablation_test")


def fig_val_clean():
    path = LOCAL_DATA / "loo_val_clean.csv"
    d = read_csv(path).set_index("stage")
    labels = list(d["label"])
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.3), gridspec_kw={"width_ratios": [1, 2.2]})
    base = d.iloc[0]
    colors = [BLUE] + [ORANGE] * (len(d) - 1)
    bars(axes[0], labels, d["map50"].values * 100, colors, ref=base["map50"] * 100)
    axes[0].set_title("mAP@0.5 (%)")
    cols = [f"{c}_ap50" for c in CLASSES]
    delta = (d[cols].values - base[cols].values.astype(float))[1:]
    heatmap(axes[1], delta, labels[1:], [CLASS_LABEL[c] for c in CLASSES], fmt="{:+.1f}", diverging=True, fig=fig)
    axes[1].set_title("Per-class AP@0.5 change vs baseline (pp)")
    axes[1].set_xticks(axes[1].get_xticks(), axes[1].get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout(w_pad=1.5)
    add_caption(fig, "YOLO26m on val_clean (2,271 images); values from RESEARCH_REPORT_LOO_CURATION_ABLATION.md")
    save(fig, "fig16_loo_ablation_val_clean")


def main():
    apply_style()
    fig_test()
    fig_val_clean()


if __name__ == "__main__":
    main()
