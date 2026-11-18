"""Fig 18-20 - Early (v6p5) training ablations on the 2,272-image validation set.

fig18_v6p5_synthetic_data   YOLO11n: v6.2E baseline vs +500 synthetic images, per class
                            (artifacts/v6p5_train_local/20261008T181605Z_6b0eac5f/*report.json)
fig19_v6p5_recipe_curves    Validation curves of the training-recipe runs
                            (artifacts/v6p5_train_local/*/runs/*/results.csv + args.yaml)
fig20_v6p5_5cls_cleaning    YOLO26n 5-class: original vs cleaned labels, per class
                            (artifacts/v6p5_train_local/20261010T133406Z_y26n_5cls_plateau_wd001/y26n_5cls_eval.csv)

These runs predate the controlled benchmark: datasets and schedules differ between
runs and several runs stopped early, so they are exploratory evidence only.
"""
import matplotlib.pyplot as plt
import numpy as np

from common import (AQUA, BLUE, CLASSES, CLASS_LABEL, DOUBLE_COL, INK, INK_2, ORANGE, ROOT, SURFACE,
                    apply_style, read_csv, read_json, save)
from helpers import caption as add_caption, dumbbell

V6P5 = ROOT / "artifacts" / "v6p5_train_local"

# run folder, legend label, colour, linestyle, planned epochs
RECIPES = [
    ("20261008T181605Z_6b0eac5f/runs/yolo11n_v6p5_data_ablation_640",
     "YOLO11n · SGD (auto) · +500 synthetic", BLUE, (0, (4, 2))),
    ("20261009T024019Z_roboflow/runs/yolo11n_v6p5_roboflow_ablation_640",
     "YOLO11n · MuSGD · +Roboflow", BLUE, "-"),
    ("20261009T131324Z_adamw/runs/yolo11n_v6p5_roboflow_adamw_cosine_640",
     "YOLO11n · AdamW cosine · +Roboflow", ORANGE, "-"),
    ("20261010T060816Z_plateau_wd001/runs/yolo11n_v6p5_plateau_wd001",
     "YOLO11n · AdamW plateau wd=0.01 · +Roboflow", ORANGE, (0, (4, 2))),
    ("20261010T064937Z_y26n_plateau_wd001/runs/yolo26n_v6p5_plateau_wd001",
     "YOLO26n · AdamW plateau wd=0.01 · +Roboflow", AQUA, "-"),
]


def _planned_epochs(run):
    for line in open(run / "args.yaml"):
        if line.startswith("epochs:"):
            return int(line.split(":")[1])
    return None


def fig_synthetic():
    path = V6P5 / "20261008T181605Z_6b0eac5f" / "v6p5_yolo11n_data_ablation_report.json"
    rep = read_json(path)
    ref = rep["reference_v6p2e_yolo11n"]["per_class"]
    new = {p["class"]: p for p in rep["per_class"]}
    labels = [CLASS_LABEL[c] for c in CLASSES] + ["All classes"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.4), sharey=True)
    for ax, key, title in ((axes[0], "AP50", "AP@0.5 (%)"), (axes[1], "AP50_95", "AP@0.5:0.95 (%)")):
        before = [ref[c][key] for c in CLASSES] + [rep["reference_v6p2e_yolo11n"]["mAP50" if key == "AP50" else "mAP50_95"]]
        after = [new[c][key] for c in CLASSES] + [rep["mAP50" if key == "AP50" else "mAP50_95"]]
        dumbbell(ax, labels, before, after, before_label="v6.2E baseline",
                 after_label=f"+{rep['added_synthetic_train_images']} synthetic")
        ax.set_title(title)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.12), ncol=2, handletextpad=0.2)
    gate = rep.get("promotion_gate", {})
    fig.tight_layout(w_pad=1.0)
    add_caption(fig, f"YOLO11n, 50 epochs, val {rep['val_images']:,} images; "
                     f"train {rep['base_train_images']:,} → {rep['final_train_images']:,} images. "
                     f"Promotion gate passed: {gate.get('candidate_pass')}")
    save(fig, "fig18_v6p5_synthetic_data")


def fig_recipes():
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.2), sharex=True)
    for folder, label, color, ls in RECIPES:
        run = V6P5 / folder
        if not (run / "results.csv").exists():
            print(f"  skip run {folder}: no results.csv")
            continue
        d = read_csv(run / "results.csv")
        d.columns = [c.strip() for c in d.columns]
        planned = _planned_epochs(run)
        done = int(d["epoch"].max())
        for ax, col in ((axes[0], "metrics/mAP50(B)"), (axes[1], "metrics/mAP50-95(B)")):
            ax.plot(d["epoch"], d[col] * 100, color=color, ls=ls, lw=1.3,
                    label=label + ("" if planned is None or done >= planned else f" (stopped {done}/{planned})"))
            if planned and done < planned:
                ax.plot(done, d[col].iloc[-1] * 100, marker="x", color=color, markersize=6, mew=1.5)
            b = d[col].idxmax()
            ax.plot(d["epoch"][b], d[col][b] * 100, marker="*", color=color, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=5)
    axes[0].set_title("Validation mAP@0.5 (%)")
    axes[1].set_title("Validation mAP@0.5:0.95 (%)")
    for ax in axes:
        ax.set_xlabel("Epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=1.2, rect=(0, 0.2, 1, 1))
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=6.3,
               handlelength=2.6, columnspacing=1.2)
    add_caption(fig, "★ best epoch; × run stopped before its planned epochs. Val set: 2,272 images. "
                     "Exploratory runs with different datasets and schedules.")
    save(fig, "fig19_v6p5_recipe_curves")


def fig_5cls(split="val"):
    path = V6P5 / "20261010T133406Z_y26n_5cls_plateau_wd001" / "y26n_5cls_eval.csv"
    d = read_csv(path)
    d = d[d["val"] == split]
    old = d[d.model == "y26n_5cls_old"].set_index("class")
    new = d[d.model == "y26n_5clsclean"].set_index("class")
    classes = [c for c in CLASSES if c in old.index and c in new.index]
    rows = classes + (["ALL"] if "ALL" in old.index else [])
    labels = [CLASS_LABEL.get(c, "All classes") for c in rows]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.2), sharey=True)
    for ax, col, title in ((axes[0], "AP50", "AP@0.5 (%)"), (axes[1], "AP50_95", "AP@0.5:0.95 (%)")):
        dumbbell(ax, labels, old.loc[rows, col].values, new.loc[rows, col].values,
                 before_label="Original 5-class labels", after_label="Cleaned 5-class labels", after_color=AQUA)
        ax.set_title(title)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.12), ncol=2, handletextpad=0.2)
    fig.tight_layout(w_pad=1.0)
    add_caption(fig, f"YOLO26n, AdamW plateau, evaluated on {split}. The original-label run stopped at epoch 9, the cleaned run at 66 "
                     "(planned 100), so part of the gap is training length.")
    save(fig, f"fig20_v6p5_5cls_cleaning_{split}")


def main():
    apply_style()
    fig_synthetic()
    fig_recipes()
    fig_5cls("val")
    fig_5cls("val_clean")


if __name__ == "__main__":
    main()
