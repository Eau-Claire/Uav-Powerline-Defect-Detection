#!/usr/bin/env python3
"""Build report-ready RC tables and figures from the current benchmark artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmakring_transmission_line_defect_model"
RESULT_ROOT = BENCH / "results" / "finding_nemo" / "operational_min8_v5"
OUT = RESULT_ROOT / "report"
FIGURE_OUT = RESULT_ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGURE_OUT.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "insulator",
    "insulator_broken",
    "insulator_flashover",
    "bird_nest",
    "foreign_object",
    "broken_strand",
]
DISPLAY = ["Insulator", "Broken\ninsulator", "Flashover", "Bird nest", "Foreign\nobject", "Broken\nstrand"]

# Baseline Raw counts are the frozen team-benchmark counts from Table RC-5.
RAW_IMG = [2993, 5125, 1081, 5469, 1372, 1929]
RAW_BOX = [8665, 6236, 1249, 6136, 1491, 2030]
RAW_TOTAL = (18334, 25807)


def dataframe_markdown(frame: pd.DataFrame, float_format: str) -> str:
    """Render a compact Markdown table without pandas' optional `tabulate` dependency."""
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(":---" if frame[column].dtype == object else "---:" for column in frame.columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(format(float(value), float_format))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def yolo_distribution(label_dir: Path) -> tuple[list[int], list[int], int, int]:
    files = sorted(label_dir.glob("*.txt"))
    img_counts: Counter[int] = Counter()
    box_counts: Counter[int] = Counter()
    for path in files:
        seen: set[int] = set()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            if cls not in range(len(CLASSES)):
                raise ValueError(f"Invalid class {cls} in {path}")
            box_counts[cls] += 1
            seen.add(cls)
        img_counts.update(seen)
    return (
        [img_counts[i] for i in range(len(CLASSES))],
        [box_counts[i] for i in range(len(CLASSES))],
        len(files),
        sum(box_counts.values()),
    )


curated_root = BENCH / "data" / "finding_nemo_human_reviewed_6cls" / "labels"
v5_root = BENCH / "data" / "test_operational_min8_v5_full_consensus_assisted" / "labels" / "test"
if len(list(v5_root.glob("*.txt"))) != 1871:
    raise FileNotFoundError(
        "Operational V5 labels are not restored (expected 1,871 files). "
        "Restore finding_nemo_test_audit_v5_bundle.tar before rebuilding report assets; "
        "the committed report remains valid and must not be overwritten with empty counts."
    )
cur_img, cur_box, cur_total_img, cur_total_box = yolo_distribution(curated_root / "train")
val_img, val_box, val_total_img, val_total_box = yolo_distribution(curated_root / "val")
test_img, test_box, test_total_img, test_total_box = yolo_distribution(v5_root)

rows = []
for i, name in enumerate(CLASSES):
    rows.append({
        "Class": name,
        "Raw train Img": RAW_IMG[i], "Raw train Box": RAW_BOX[i],
        "Curated train Img": cur_img[i], "Curated train Box": cur_box[i],
        "Val Img": val_img[i], "Val Box": val_box[i],
        "Test V5 Img": test_img[i], "Test V5 Box": test_box[i],
    })
rows.append({
    "Class": "TOTAL unique images",
    "Raw train Img": RAW_TOTAL[0], "Raw train Box": RAW_TOTAL[1],
    "Curated train Img": cur_total_img, "Curated train Box": cur_total_box,
    "Val Img": val_total_img, "Val Box": val_total_box,
    "Test V5 Img": test_total_img, "Test V5 Box": test_total_box,
})
distribution = pd.DataFrame(rows)
distribution.to_csv(OUT / "table_rc5_dataset_distribution.csv", index=False)

md_rows = []
for _, r in distribution.iterrows():
    md_rows.append(
        f"| **{r['Class']}** | {r['Raw train Img']:,} ({r['Raw train Box']:,}) | "
        f"{r['Curated train Img']:,} ({r['Curated train Box']:,}) | "
        f"{r['Val Img']:,} ({r['Val Box']:,}) | {r['Test V5 Img']:,} ({r['Test V5 Box']:,}) |"
    )
table_md = "\n".join([
    "**Table RC-5.** Image and box counts for the matched Raw-versus-Curated benchmark. Img denotes images containing at least one instance of the class; Box denotes annotated instances. Test V5 is the model-assisted operational min-8 audit set.",
    "",
    "| **Class** | **Raw train Img (Box)** | **Curated train Img (Box)** | **Val Img (Box)** | **Test V5 Img (Box)** |",
    "|---|---:|---:|---:|---:|",
    *md_rows,
])
(OUT / "table_rc5_dataset_distribution.md").write_text(table_md + "\n", encoding="utf-8")

# Distribution figure: per-class counts, excluding the non-additive unique-image total.
x = np.arange(len(CLASSES))
series = {
    "Raw train": (RAW_IMG, RAW_BOX),
    "Curated train": (cur_img, cur_box),
    "Validation": (val_img, val_box),
    "Test V5": (test_img, test_box),
}
colors = ["#7A7A7A", "#1677B8", "#E69F00", "#009E73"]
fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.2), sharex=True)
width = 0.19
for j, ((label, values), color) in enumerate(zip(series.items(), colors)):
    axes[0].bar(x + (j - 1.5) * width, values[0], width, label=label, color=color)
    axes[1].bar(x + (j - 1.5) * width, values[1], width, label=label, color=color)
for ax, ylabel in zip(axes, ("Images containing class", "Annotated boxes")):
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(ncol=4, frameon=False, loc="upper right")
axes[1].set_xticks(x, DISPLAY)
fig.suptitle("Class distribution across the Raw, Curated, validation, and operational V5 splits", fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURE_OUT / "figure_rc_dataset_distribution.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURE_OUT / "figure_rc_dataset_distribution.pdf", bbox_inches="tight")
plt.close(fig)

# Matched V5 metrics and Curated-minus-Raw deltas.
metrics_path = RESULT_ROOT / "benchmark_summary.csv"
metrics = pd.read_csv(metrics_path)
model_order = ["yolo11n", "yolo11m", "yolo26n", "yolo26m", "rtdetr50", "rtdetr101"]
labels = ["YOLO11n", "YOLO11m", "YOLO26n", "YOLO26m", "RT-DETR-R50", "RT-DETR-R101"]
metrics["Train"] = pd.Categorical(metrics["Train"], categories=["Raw", "Curated"], ordered=True)
metrics["Model"] = pd.Categorical(metrics["Model"], categories=model_order, ordered=True)
metrics = metrics.sort_values(["Train", "Model"]).reset_index(drop=True)
metrics.to_csv(OUT / "table_rc_v5_raw_curated_metrics.csv", index=False)

raw = metrics[metrics["Train"] == "Raw"].set_index("Model").reindex(model_order)
cur = metrics[metrics["Train"] == "Curated"].set_index("Model").reindex(model_order)
deltas = pd.DataFrame({
    "Model": labels,
    "Delta F1 (pp)": (cur["F1"].to_numpy() - raw["F1"].to_numpy()) * 100,
    "Delta mAP50 (pp)": (cur["mAP50"].to_numpy() - raw["mAP50"].to_numpy()) * 100,
    "Delta mAP50:95 (pp)": (cur["mAP50:95"].to_numpy() - raw["mAP50:95"].to_numpy()) * 100,
    "Delta MEAN_5 (pp)": (cur["MEAN_5"].to_numpy() - raw["MEAN_5"].to_numpy()) * 100,
})
deltas.to_csv(OUT / "table_rc_v5_curated_minus_raw_deltas.csv", index=False, float_format="%.2f")
report_metrics = metrics.copy()
report_metrics["Model"] = report_metrics["Model"].astype(str).map(dict(zip(model_order, labels)))

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=False)
for ax, metric, title in zip(axes, ("F1", "mAP50", "mAP50:95"), ("F1", "mAP@0.5", "mAP@0.5:0.95")):
    pos = np.arange(len(model_order))
    ax.bar(pos - .18, raw[metric].to_numpy() * 100, .36, label="Raw", color="#7A7A7A")
    ax.bar(pos + .18, cur[metric].to_numpy() * 100, .36, label="Curated", color="#1677B8")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(pos, labels, rotation=35, ha="right")
    ax.set_ylabel("Score (%)")
    ax.grid(axis="y", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False)
fig.suptitle("Matched Raw-versus-Curated results on operational min-8 audit V5", fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURE_OUT / "figure_rc_v5_raw_vs_curated.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURE_OUT / "figure_rc_v5_raw_vs_curated.pdf", bbox_inches="tight")
plt.close(fig)

section = f"""## 5.2.2 Matched results on operational min-8 audit V5

All twelve fixed checkpoints (Raw and Curated training across six architectures) were evaluated with the same evaluator on `test_operational_min8_v5_full_consensus_assisted`. The set contains {test_total_img:,} images and {test_total_box:,} boxes after the fixed minimum-side exclusion and V5 consensus-assisted annotation updates. Because evaluated model families contributed proposals to V5, these results are an operational exploratory comparison rather than a replacement for the independent `test_main_v1` benchmark.

**Table RC-9a.** Matched Raw-versus-Curated detection results on operational min-8 audit V5.

{dataframe_markdown(report_metrics, '.4f')}

**Table RC-9b.** Curated-minus-Raw differences on operational min-8 audit V5, in percentage points.

{dataframe_markdown(deltas, '+.2f')}

Curated YOLO26m achieves the highest mAP50 (0.8284), mAP50:95 (0.4780), and MEAN_5 (0.7999). Curated RT-DETR-R101 has the highest F1 (0.8191), while Raw RT-DETR-R101 has high precision but substantially lower recall at the reported operating point. The matched results therefore support YOLO26m as the strongest overall V5 detector under threshold-independent AP metrics, while the V5 provenance remains explicit.

**Figure RC-1.** Matched Raw-versus-Curated F1, mAP50, and mAP50:95 on operational min-8 audit V5. This is a model-assisted exploratory protocol, not the primary independent test.
"""
(OUT / "section_5_2_v5_ready_to_paste.md").write_text(section, encoding="utf-8")

print(OUT)
print(distribution.to_string(index=False))
print(deltas.to_string(index=False))
