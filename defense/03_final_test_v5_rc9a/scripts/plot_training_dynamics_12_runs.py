#!/usr/bin/env python3
"""Plot one four-panel training-dynamics figure for all 12 Raw/Curated runs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO / "camber" / "artifacts" / "benchmark"
OUT_DIR = (
    REPO
    / "benchmakring_transmission_line_defect_model"
    / "results"
    / "finding_nemo"
    / "operational_min8_v5"
    / "figures"
)
OUTPUT_STEM = OUT_DIR / "figure_training_dynamics_12_runs"

RUNS = {
    "YOLO11n": ("raw/yolo11n/results.csv", "curated_latest/yolo11n/results.csv", "yolo"),
    "YOLO11m": ("raw/yolo11m/results.csv", "curated_latest/yolo11m/results.csv", "yolo"),
    "YOLO26n": ("raw/yolo26n/results.csv", "curated_latest/yolo26n/results.csv", "yolo"),
    "YOLO26m": ("raw/yolo26m/results.csv", "curated_latest/yolo26m/results.csv", "yolo"),
    "RT-DETR-R50": ("raw/rtdetr50/result.csv", "curated_latest/r50vd/result.csv", "rtdetr"),
    "RT-DETR-R101": ("raw/rtdetr101/result.csv", "curated_latest/r101vd/result.csv", "rtdetr"),
}

COLORS = {
    "YOLO11n": "#2878c7",
    "YOLO11m": "#15579b",
    "YOLO26n": "#ef7c3a",
    "YOLO26m": "#c84d18",
    "RT-DETR-R50": "#25a77a",
    "RT-DETR-R101": "#087b59",
}
LINESTYLES = {"Raw": (0, (5, 3)), "Curated": "-"}
METRICS = {
    "map50": "metrics/mAP50(B)",
    "map5095": "metrics/mAP50-95(B)",
    "val_box": "val/box_loss",
}


def read_log(relative_path: str) -> pd.DataFrame:
    frame = pd.read_csv(LOG_ROOT / relative_path)
    frame.columns = [column.strip() for column in frame.columns]
    required = ["epoch", *METRICS.values()]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing columns {missing} in {relative_path}")
    return frame[required].dropna().sort_values("epoch")


def style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(color="#dedede", linewidth=0.85)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(1, 50)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.labelsize": 10.5})
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2))
    ax_map50, ax_map5095, ax_yolo_loss, ax_rt_loss = axes.flat

    legend_handles = []
    legend_labels = []
    for model, (raw_path, curated_path, family) in RUNS.items():
        for regime, relative_path in (("Raw", raw_path), ("Curated", curated_path)):
            frame = read_log(relative_path)
            epoch = frame["epoch"]
            label = f"{model} · {regime}"
            common = {
                "color": COLORS[model],
                "linestyle": LINESTYLES[regime],
                "linewidth": 2.05 if regime == "Curated" else 1.8,
                "alpha": 0.97 if regime == "Curated" else 0.82,
            }

            line = ax_map50.plot(epoch, frame[METRICS["map50"]] * 100, label=label, **common)[0]
            ax_map5095.plot(epoch, frame[METRICS["map5095"]] * 100, **common)
            target_loss_axis = ax_yolo_loss if family == "yolo" else ax_rt_loss
            target_loss_axis.plot(epoch, frame[METRICS["val_box"]], **common)

            best = frame[METRICS["map50"]].idxmax()
            ax_map50.scatter(
                frame.loc[best, "epoch"],
                frame.loc[best, METRICS["map50"]] * 100,
                marker="*",
                s=72,
                color=COLORS[model],
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
            )
            legend_handles.append(line)
            legend_labels.append(label)

    style_axis(ax_map50, "(a) Validation mAP@0.5", "mAP@0.5 (%)")
    style_axis(ax_map5095, "(b) Validation mAP@0.5:0.95", "mAP@0.5:0.95 (%)")
    style_axis(ax_yolo_loss, "(c) YOLO validation box loss", "Validation box loss")
    style_axis(ax_rt_loss, "(d) RT-DETR validation box loss", "Validation box loss")
    ax_map50.set_ylim(10, 89)
    ax_map5095.set_ylim(5, 50)

    fig.suptitle(
        "Training dynamics across 12 Raw and Curated detector runs",
        fontsize=18,
        fontweight="bold",
        y=0.992,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
        fontsize=9.2,
        columnspacing=1.5,
        handlelength=3.0,
    )
    fig.text(
        0.5,
        0.012,
        "Solid = Curated; dashed = Raw. Stars mark maximum observed validation mAP50. "
        "Loss panels are separated by model family because YOLO and RT-DETR objectives are not directly comparable.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.115, 0.98, 0.95), h_pad=1.8, w_pad=1.6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
