#!/usr/bin/env python3
"""Plot matched Raw/Curated validation-mAP50 trajectories for six detectors."""

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
OUTPUT_STEM = OUT_DIR / "figure_validation_map50_raw_curated"
SUMMARY_CSV = OUT_DIR.parent / "report" / "validation_map50_maxima.csv"

RUNS = {
    "YOLO11n": ("raw/yolo11n/results.csv", "curated_latest/yolo11n/results.csv"),
    "YOLO11m": ("raw/yolo11m/results.csv", "curated_latest/yolo11m/results.csv"),
    "YOLO26n": ("raw/yolo26n/results.csv", "curated_latest/yolo26n/results.csv"),
    "YOLO26m": ("raw/yolo26m/results.csv", "curated_latest/yolo26m/results.csv"),
    "RT-DETR-R50": ("raw/rtdetr50/result.csv", "curated_latest/r50vd/result.csv"),
    "RT-DETR-R101": ("raw/rtdetr101/result.csv", "curated_latest/r101vd/result.csv"),
}

METRIC = "metrics/mAP50(B)"
COLORS = {"Raw": "#7b8794", "Curated": "#1676c2"}


def read_log(relative_path: str) -> pd.DataFrame:
    path = LOG_ROOT / relative_path
    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    if "epoch" not in frame or METRIC not in frame:
        raise ValueError(f"Required columns missing from {path}")
    return frame[["epoch", METRIC]].dropna().sort_values("epoch")


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
    })
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 10), sharex=True, sharey=True)
    maxima = []

    for ax, (model, (raw_path, curated_path)) in zip(axes.flat, RUNS.items()):
        for regime, relative_path in (("Raw", raw_path), ("Curated", curated_path)):
            frame = read_log(relative_path)
            epochs = frame["epoch"].to_numpy()
            scores = frame[METRIC].to_numpy() * 100
            best_position = scores.argmax()
            best_epoch = int(epochs[best_position])
            best_score = float(scores[best_position])

            ax.plot(
                epochs,
                scores,
                color=COLORS[regime],
                linewidth=2.1,
                label=regime,
                alpha=0.96,
            )
            ax.scatter(
                [best_epoch],
                [best_score],
                marker="*",
                s=135,
                color=COLORS[regime],
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
            ax.annotate(
                f"e{best_epoch}: {best_score:.1f}",
                (best_epoch, best_score),
                xytext=(-4, 8 if regime == "Curated" else -14),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color=COLORS[regime],
            )
            maxima.append(
                {
                    "Model": model,
                    "Train": regime,
                    "Max validation mAP50 epoch": best_epoch,
                    "Max validation mAP50": best_score / 100,
                    "Source log": str(LOG_ROOT / relative_path),
                }
            )

        ax.set_title(model, loc="left")
        ax.grid(color="#dedede", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(1, 50)
        ax.set_ylim(50, 88)

    for ax in axes[:, 0]:
        ax.set_ylabel("Validation mAP50 (%)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.suptitle(
        "Raw versus Curated validation trajectories",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Stars mark the maximum observed validation mAP50 row in each available log; "
        "they do not by themselves verify the checkpoint evaluated on V5.",
        ha="center",
        fontsize=9.2,
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.045, 0.98, 0.91), h_pad=1.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    pd.DataFrame(maxima).to_csv(SUMMARY_CSV, index=False)
    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".pdf"))
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
