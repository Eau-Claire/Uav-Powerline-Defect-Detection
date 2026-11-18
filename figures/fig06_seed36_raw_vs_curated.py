"""Fig 6 - Raw vs Curated replication with seed 36 (YOLO only, 1,931 test images).

Sources: results_seed/finding_seed36_test_audit_v5/{summary.csv, */metrics.json}.
Evaluated on the 1,931-image test set of the v5 audit bundle; do not mix these
numbers with the val_clean results of Fig 4 in one table.

Only runs that still have a metrics.json are used: commit ceeb9fc removed the
curated seed-36 metrics as wrong while summary.csv kept the stale rows. Until the
curated runs are re-evaluated the figure is skipped.
"""
import matplotlib.pyplot as plt
import pandas as pd

from common import (BENCH, CLASSES, CLASS_LABEL, DOUBLE_COL, MODEL_LABEL, MODEL_ORDER, apply_style,
                    norm_model, read_csv, read_json, save)
from helpers import dumbbell, heatmap

SRC = BENCH / "results_seed" / "finding_seed36_test_audit_v5"


def load():
    s = read_csv(SRC / "summary.csv")
    s["model"] = s["model"].map(norm_model)
    s = s[[(SRC / r / "metrics.json").exists() for r in s["run"]]].copy()
    for i, r in s.iterrows():
        m = read_json(SRC / r["run"] / "metrics.json")
        for c, v in m.get("per_class_map50_95", {}).items():
            s.loc[i, f"{c}_ap50_95"] = v
    raw = s[s.dataset == "raw"].set_index("model")
    cur = s[s.dataset == "curated"].set_index("model")
    order = [m for m in MODEL_ORDER if m in raw.index and m in cur.index]
    return raw.loc[order], cur.loc[order]


def main():
    apply_style()
    raw, cur = load()
    if raw.empty or cur.empty:
        print("  skip fig06: seed-36 needs both raw and curated metrics.json "
              f"(raw={len(raw)}, curated={len(cur)})")
        return
    labels = [MODEL_LABEL[m] for m in raw.index]
    fig = plt.figure(figsize=(DOUBLE_COL, 2.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.6])
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    dumbbell(ax1, labels, raw["map50"].values, cur["map50"].values)
    ax1.set_title("mAP@0.5 (%)")
    dumbbell(ax2, labels, raw["map50_95"].values, cur["map50_95"].values)
    ax2.set_title("mAP@0.5:0.95 (%)")
    ax2.set_yticklabels([])
    ax1.legend(loc="lower left", bbox_to_anchor=(0, 1.13), ncol=2, handletextpad=0.2)
    ax3 = fig.add_subplot(gs[2])
    cols = [f"{c}_ap50_95" for c in CLASSES]
    heatmap(ax3, cur[cols].values - raw[cols].values, labels, [CLASS_LABEL[c] for c in CLASSES],
            fmt="{:+.1f}", diverging=True, fig=fig)
    ax3.set_title("Δ AP@0.5:0.95 (pp), Curated − Raw")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig06_seed36_raw_vs_curated")


if __name__ == "__main__":
    main()
