"""Shared paths, palette, model metadata, loaders and save helpers for paper figures.

Every figure script imports from here so that colours, model order and fonts stay
identical across the paper. Run any script from anywhere; paths resolve from this file.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmakring_transmission_line_defect_model"
DATA = BENCH / "data"
NEMO = BENCH / "results_ablation" / "finding_nemo"
OUT = Path(__file__).resolve().parent / "out"
LOCAL_DATA = Path(__file__).resolve().parent / "data"

# --------------------------------------------------------------------------- palette
# Reference categorical palette (dataviz skill), first three slots: validated
# all-pairs for CVD + normal vision. Aqua is < 3:1 on white, so every chart that
# uses it also carries direct labels or marker-shape encoding.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e0"
RAW_GRAY = "#a3a29c"  # "before" state in before/after charts
SURFACE = "#ffffff"
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#f0efec", "#2a78d6"  # red <-> gray <-> blue

FAMILY_COLOR = {"YOLO11": BLUE, "YOLO26": ORANGE, "RT-DETR": AQUA}

# key, label, family, size ("small" = n / R50, "large" = m / R101)
MODELS = [
    ("yolo11n", "YOLO11n", "YOLO11", "small"),
    ("yolo11m", "YOLO11m", "YOLO11", "large"),
    ("yolo26n", "YOLO26n", "YOLO26", "small"),
    ("yolo26m", "YOLO26m", "YOLO26", "large"),
    ("rtdetr50", "RT-DETR-R50", "RT-DETR", "small"),
    ("rtdetr101", "RT-DETR-R101", "RT-DETR", "large"),
]
MODEL_ORDER = [m[0] for m in MODELS]
MODEL_LABEL = {m[0]: m[1] for m in MODELS}
MODEL_FAMILY = {m[0]: m[2] for m in MODELS}
MODEL_SIZE = {m[0]: m[3] for m in MODELS}
ALIASES = {"rtdetr100": "rtdetr101", "rtdetr_r50": "rtdetr50", "rtdetr_r101": "rtdetr101"}

CLASSES = ["insulator", "insulator_broken", "insulator_flashover",
           "bird_nest", "foreign_object", "broken_strand"]
CLASS_LABEL = {
    "insulator": "Insulator",
    "insulator_broken": "Broken insulator",
    "insulator_flashover": "Flashover",
    "bird_nest": "Bird nest",
    "foreign_object": "Foreign object",
    "broken_strand": "Broken strand",
}


def norm_model(key: str) -> str:
    key = key.strip().lower()
    for prefix in ("raw_", "curated_", "nemo_"):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return ALIASES.get(key, key)


def model_color(key: str) -> str:
    return FAMILY_COLOR[MODEL_FAMILY[norm_model(key)]]


def model_marker(key: str) -> dict:
    """Large variants are filled; small variants are hollow (secondary encoding)."""
    k = norm_model(key)
    c = model_color(k)
    shape = {"YOLO11": "o", "YOLO26": "s", "RT-DETR": "D"}[MODEL_FAMILY[k]]
    if MODEL_SIZE[k] == "large":
        return dict(marker=shape, color=c, markerfacecolor=c, markeredgecolor=SURFACE,
                    markeredgewidth=0.8, markersize=6.5)
    return dict(marker=shape, color=c, markerfacecolor=SURFACE, markeredgecolor=c,
                markeredgewidth=1.4, markersize=6)


def model_linestyle(key: str) -> str:
    return "-" if MODEL_SIZE[norm_model(key)] == "large" else (0, (4, 2))


# --------------------------------------------------------------------------- style
SINGLE_COL = 3.5   # inches (IEEE / Elsevier single column)
DOUBLE_COL = 7.16  # inches (two-column width)


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 8,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": INK_3,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "pdf.fonttype": 42,  # embed TrueType so journals accept the PDF
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


# figure-number -> output sub-folder
GROUPS = {
    "01_dataset": (1, 2, 3, 21),
    "02_raw_vs_curated": (4, 5, 6),
    "03_benchmark": (7, 8, 9, 12),
    "04_analysis": (10, 11, 13, 14, 17),
    "05_ablation": (15, 16, 18, 19, 20),
}


def out_dir_for(name: str) -> Path:
    if name.startswith("paper_"):
        return OUT / "07_paper"
    if name.startswith("diag"):
        return OUT / "06_method"
    try:
        num = int(name[3:5])
    except ValueError:
        return OUT / "misc"
    for group, nums in GROUPS.items():
        if num in nums:
            return OUT / group
    return OUT / "misc"


def save(fig, name: str) -> list[Path]:
    """Write <name>.pdf (vector, for LaTeX) and <name>.png (300 dpi, for Word/slides)
    into figures/out/<group>/."""
    folder = out_dir_for(name)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = folder / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03)
        paths.append(p)
    plt.close(fig)
    print(f"  wrote {folder.name}/{name}.pdf / .png")
    return paths


def model_legend_handles(keys=None):
    from matplotlib.lines import Line2D
    keys = keys or MODEL_ORDER
    return [Line2D([0], [0], linestyle="none", label=MODEL_LABEL[k], **model_marker(k)) for k in keys]


def panel_title(ax, title: str, subtitle: str | None = None) -> None:
    """Bold panel title with an optional regular-weight subtitle line under it."""
    ax.set_title(title, pad=14 if subtitle else 4)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=6.5, color=INK_2,
                ha="left", va="bottom")


def pct(x, _pos=None) -> str:
    return f"{x * 100:.0f}"


# --------------------------------------------------------------------------- loaders
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def read_json(path: Path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def eval_summary(folder: Path) -> pd.DataFrame:
    """benchmark_summary.csv + per_class_summary.csv of a selected-checkpoint evaluation."""
    s = read_csv(folder / "benchmark_summary.csv")
    key_col = "Model" if "Model" in s.columns else "Run"
    s["model"] = s[key_col].map(norm_model)
    pc_path = folder / "per_class_summary.csv"
    if pc_path.exists():
        pc = read_csv(pc_path)
        pc["model"] = pc["Model"].map(norm_model)
        s = s.merge(pc.drop(columns=[c for c in ("Model", "Description") if c in pc]), on="model")
    s = s[s["model"].isin(MODEL_ORDER)].copy()
    s["order"] = s["model"].map(MODEL_ORDER.index)
    return s.sort_values("order").reset_index(drop=True)


def benchmark_json(path: Path) -> pd.DataFrame:
    """Flatten a benchmark_test_results.json (raw/curated runs) into one row per run."""
    rows = []
    for run_key, r in read_json(path).items():
        acc, hw = r["accuracy"], r.get("hardware", {})
        row = {
            "run": run_key,
            "regime": run_key.split("_", 1)[0].lower(),
            "model": norm_model(run_key),
            "precision": acc["precision"], "recall": acc["recall"], "f1": acc["f1_score"],
            "map50": acc["mAP50"], "map50_95": acc["mAP50_95"],
            "mean5": acc.get("MEAN_5_defect_ap50"),
            "latency_ms": hw.get("latency_ms"), "params_m": hw.get("parameters_m"),
            "gflops": hw.get("gflops"), "vram_mb": hw.get("inference_peak_vram_mb"),
            "dataset_yaml": r.get("evaluation_dataset_yaml"),
        }
        for c, v in acc.get("per_class", {}).items():
            row[f"{c}_ap50"], row[f"{c}_ap50_95"] = v["ap50"], v["ap50_95"]
        rows.append(row)
    return pd.DataFrame(rows)


def yolo_label_rows(label_dir: Path):
    """Yield (class_id, xc, yc, w, h) for every YOLO label line under label_dir."""
    for p in sorted(label_dir.glob("*.txt")):
        with open(p) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    yield p.stem, int(float(parts[0])), *map(float, parts[1:5])


def read_simple_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))
