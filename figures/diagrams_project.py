"""Project-level diagrams and the dataset-version chart.

diag06_project_pipeline   Whole project: sources -> datasets -> curation & review -> training
                          -> evaluation -> deployment
diag07_cross_label_review How finding_nemo_human_reviewed_6cls is produced from curated data
fig21_dataset_versions    Boxes per class: Raw vs Curated vs Human-reviewed, per split

Every image/box count is read from the label files at run time, so the figures stay
correct if a dataset is rebuilt. Sources of the remaining facts:
  finding_nemo_raw/data.yaml               raw = v6p5_roboflow_ablation (train/val)
  finding_nemo_human_reviewed_6cls/DATASET_STATUS.json   reviewed splits and their sources
  tools/roboflow_relabel/*.py, HANDOFF.md  cross-label workflow
  data/curated_combined/manifest.json      curation
  deployment/pi3/README.md, deployment/docker_api/README.md   deployment targets
"""
import collections
import os

import matplotlib.pyplot as plt
import numpy as np

from common import (BLUE, CLASSES, CLASS_LABEL, DATA, DOUBLE_COL, INK, INK_2, ORANGE, RAW_GRAY,
                    SURFACE, apply_style, panel_title, read_json, save)
from diagrams import anchor, arrow, box, canvas, elbow, lane, legend

RAW = DATA / "finding_nemo_raw"
CUR = DATA / "curated_combined"
REV = DATA / "finding_nemo_human_reviewed_6cls"
TEST = DATA / "test_main_v1"


def count(label_dir):
    """(images, boxes, per-class Counter) for one YOLO label folder."""
    c, n = collections.Counter(), 0
    for f in os.listdir(label_dir):
        if not f.endswith(".txt"):
            continue
        n += 1
        with open(label_dir / f) as fh:
            for line in fh:
                if line.strip():
                    c[int(float(line.split()[0]))] += 1
    return n, sum(c.values()), c


def counts():
    return {
        "raw_train": count(RAW / "labels" / "train"),
        "raw_val": count(RAW / "labels" / "val"),
        "cur_train": count(CUR / "labels" / "train"),
        "cur_val_clean": count(CUR / "labels" / "val_clean"),
        "rev_train": count(REV / "labels" / "train"),
        "rev_val": count(REV / "labels" / "val"),
        "rev_test": count(REV / "labels" / "test"),
        "test_main_v1": count(TEST / "labels" / "test"),
    }


def fmt(c):
    return f"{c[0]:,} img · {c[1]:,} boxes"


# --------------------------------------------------------------------------- diag06
def vlane(ax, y, h, label, x=0.06, w=7.04):
    """Horizontal band with its label written vertically in a left gutter."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor="none", edgecolor="#e2e1db", linewidth=0.8, zorder=0))
    ax.text(x + 0.13, y - h / 2, label.upper(), rotation=90, fontsize=5.6, color="#8a8984",
            fontweight="bold", ha="center", va="center")


def diag_project(k):
    fig, ax = canvas(7.16, 6.95)
    legend(ax, 0.34, 6.9)
    L = 0.34

    vlane(ax, 6.72, 1.0, "1 · Sources")
    src = box(ax, L, 6.52, 2.4, 0.72, "Six public sources (benchmark)", [
        "ZHENG1600 · IDID · CPLID", "VPMBGI · InsuFault-Net · FOTL / EFOD"], role="data")
    ext = box(ax, 2.88, 6.52, 2.3, 0.72, "V6.5 external stage", [
        "Open Images V7 composites (+500 syn.)", "Roboflow transmission (+351 train)"], role="data")
    box(ax, 5.32, 6.52, 1.7, 0.72, "Studied, not merged", [
        "MPCD (source unavailable)", "Power Equipment (~86% dup.)"], role="note", dashed=True, body_size=5.7)

    vlane(ax, 5.6, 1.12, "2 · Datasets")
    v62 = box(ax, L, 5.38, 1.85, 0.75, "V6.2E parent (frozen)", [
        "v6p2e_23562a45b343", "taxonomy · QA · dedup · split"], role="data")
    raw = box(ax, 2.33, 5.38, 2.5, 0.75, "finding_nemo_raw  (Raw)", [
        f"train {fmt(k['raw_train'])}", f"val {fmt(k['raw_val'])}"], role="data")
    tst = box(ax, 4.97, 5.38, 2.05, 0.75, "test_main_v1", [
        "V6.2E TEST 2,012 → dedup −50", f"→ −31 person-FO → {k['test_main_v1'][0]:,} img"], role="eval")
    arrow(ax, (1.2, src["y"] - src["h"]), (1.2, v62["y"]))
    arrow(ax, (3.6, ext["y"] - ext["h"]), (3.6, raw["y"]))
    arrow(ax, anchor(v62, "r"), anchor(raw, "l"))
    elbow(ax, [(1.85, v62["y"]), (1.85, 5.48), (6.0, 5.48), (6.0, tst["y"])])

    vlane(ax, 4.36, 1.3, "3 · Curation & review")
    gy, gh, g = 4.12, 0.98, 0.12
    xs, ws = [L], [1.0, 1.22, 1.22, 1.02, 1.74]
    for wi in ws[:-1]:
        xs.append(xs[-1] + wi + g)
    mod = box(ax, xs[0], gy, ws[0], gh, "Curation", ["Modules I–IV", "A–H rules", "train only"], role="process")
    cur = box(ax, xs[1], gy, ws[1], gh, "curated_combined", [
        f"train {k['cur_train'][0]:,} img", f"  {k['cur_train'][1]:,} boxes", f"val_clean {k['cur_val_clean'][0]:,}"],
        role="data")
    crs = box(ax, xs[2], gy, ws[2], gh, "Cross-label audit", [
        "curated YOLO26m", "+ Roboflow workflow", "→ proposals only"], role="process")
    rvw = box(ax, xs[3], gy, ws[3], gh, "Human review", ["accept / reject", "each proposal", "train · val · test"],
              role="process")
    rev = box(ax, xs[4], gy, ws[4], gh, "Human-reviewed 6-class", [
        f"train {k['rev_train'][0]:,} · {k['rev_train'][1]:,} bx", f"val {k['rev_val'][0]:,} · {k['rev_val'][1]:,} bx",
        f"test {k['rev_test'][0]:,} · {k['rev_test'][1]:,} bx"], role="data", body_size=5.8)
    for p, q in ((mod, cur), (cur, crs), (crs, rvw), (rvw, rev)):
        arrow(ax, anchor(p, "r"), anchor(q, "l"))
    elbow(ax, [(3.0, raw["y"] - raw["h"]), (3.0, 4.42), (mod["x"] + 0.5, 4.42), (mod["x"] + 0.5, gy)])
    elbow(ax, [(6.0, tst["y"] - tst["h"]), (6.0, 4.42), (crs["x"] + 0.8, 4.42), (crs["x"] + 0.8, gy)])

    vlane(ax, 2.96, 1.0, "4 · Training")
    ax.text(7.0, 2.9, "same recipe: 640 px · 50 epochs · AdamW cosine · AMP", fontsize=5.7, color=INK_2,
            ha="right", va="top")
    ty, th = 2.72, 0.66
    bench = box(ax, L, ty, 2.15, th, "2×6 Raw vs Curated", [
        "6 detectors × {finding_nemo_raw,", "curated_combined} · seeds 42, 36"], role="model", body_size=5.8)
    nemo = box(ax, 2.62, ty, 2.15, th, "Finding Nemo line", [
        "6 detectors on the", "human-reviewed set · seed 42"], role="model", body_size=5.8)
    abl = box(ax, 4.9, ty, 2.12, th, "Ablations (YOLO26m)", [
        "curation modules: cumulative", "and leave-one-out"], role="model", body_size=5.8)
    elbow(ax, [(cur["x"] + 0.6, gy - gh), (cur["x"] + 0.6, 3.01), (1.4, 3.01), (1.4, ty)])
    elbow(ax, [(rev["x"] + 0.9, gy - gh), (rev["x"] + 0.9, 3.01), (3.7, 3.01), (3.7, ty)])

    vlane(ax, 1.86, 0.86, "5 · Evaluation")
    ev = box(ax, L, 1.64, 6.68, 0.56, "Select the checkpoint on validation, then evaluate once per frozen test", [
        "test_main_v1 (independent, primary) · human-reviewed test · operational min-8 · "
        "audits v2–v5 (exploratory)  →  P · R · F1 · mAP · MEAN_5 · latency"], role="eval", body_size=5.8)
    for b in (bench, nemo, abl):
        arrow(ax, (b["x"] + b["w"] / 2, ty - th), (b["x"] + b["w"] / 2, ev["y"]))

    vlane(ax, 0.92, 0.86, "6 · Deploy")
    pi = box(ax, L, 0.7, 3.27, 0.56, "Edge · Raspberry Pi 3", [
        "curated YOLO11n → ONNX → NCNN, 640 px · alerts (NCNN on Pi pending)"], role="model", body_size=5.7)
    api = box(ax, 3.75, 0.7, 3.27, 0.56, "Server · Docker API / Modal", [
        "POST /detect · /health · /ready · baseline v5 checkpoint (7 classes)"], role="model", body_size=5.7)
    arrow(ax, (2.0, ev["y"] - ev["h"]), (2.0, pi["y"]))
    arrow(ax, (5.4, ev["y"] - ev["h"]), (5.4, api["y"]))
    save(fig, "diag06_project_pipeline")


# --------------------------------------------------------------------------- diag07
def diag_cross_label(k):
    status = read_json(REV / "DATASET_STATUS.json")
    fig, ax = canvas(7.16, 2.85)
    inp = box(ax, 0.1, 2.75, 1.25, 1.0, "Inputs", ["curated train", "val_clean", "test_main_v1", "(labels untouched)"],
              role="data")
    yolo = box(ax, 1.55, 2.75, 1.35, 0.46, "Teacher", ["curated YOLO26m"], role="model")
    rf = box(ax, 1.55, 2.17, 1.35, 0.46, "Roboflow", ["local inference workflow"], role="model")
    prop = box(ax, 3.1, 2.75, 1.3, 1.0, "Proposals", ["yolo.jsonl", "proposals.jsonl", "changes / conflicts", "(unreviewed)"],
               role="process")
    rev = box(ax, 4.6, 2.75, 1.15, 1.0, "Human review", ["browser tool (train)", "Streamlit (val, test)",
                                                          "accept / reject"], role="process", body_size=5.8)
    app = box(ax, 5.95, 2.75, 1.12, 1.0, "Apply", ["original labels", "+ accepted", "proposals only"], role="process")
    for a, b in ((inp, yolo), (inp, rf)):
        arrow(ax, anchor(a, "r"), anchor(b, "l"))
    arrow(ax, anchor(yolo, "r"), (prop["x"], 2.52))
    arrow(ax, anchor(rf, "r"), (prop["x"], 1.94))
    arrow(ax, anchor(prop, "r"), anchor(rev, "l"))
    arrow(ax, anchor(rev, "r"), anchor(app, "l"))

    out = box(ax, 1.55, 1.42, 5.52, 0.68, "finding_nemo_human_reviewed_6cls", [
        "   ·   ".join(f"{sp}: {status['splits'][sp]['images']:,} img, {status['splits'][sp]['boxes']:,} boxes "
                       f"(from {status['splits'][sp]['source_split']})" for sp in ("train", "val", "test")),
        f"boxes before → after review: train {k['cur_train'][1]:,} → {k['rev_train'][1]:,} · "
        f"val_clean {k['cur_val_clean'][1]:,} → {k['rev_val'][1]:,} · test {k['test_main_v1'][1]:,} → {k['rev_test'][1]:,}",
        "assembled with relative paths; DATASET_STATUS.json stores the SHA-256 of each split's review status"],
        role="data", body_size=5.5)
    elbow(ax, [(app["x"] + app["w"] / 2, app["y"] - app["h"]), (app["x"] + app["w"] / 2, 1.55),
               (app["x"] + app["w"] / 2, out["y"])])
    ax.text(0.1, 0.5, "Rule: model output is never written as ground truth unless a human accepted it. "
                       "The reviewed test is therefore reported separately from the independent test_main_v1.",
            fontsize=5.8, color=INK_2, style="italic")
    save(fig, "diag07_cross_label_review")


# --------------------------------------------------------------------------- fig21
def fig_versions(k):
    panels = [
        ("Train", [("Raw", "raw_train", RAW_GRAY), ("Curated", "cur_train", BLUE),
                   ("Reviewed", "rev_train", ORANGE)]),
        ("Validation", [("Raw val", "raw_val", RAW_GRAY), ("val_clean", "cur_val_clean", BLUE),
                        ("Reviewed", "rev_val", ORANGE)]),
        ("Test", [("test_main_v1", "test_main_v1", RAW_GRAY), ("Reviewed", "rev_test", ORANGE)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.5), sharey=True)
    y = np.arange(len(CLASSES))[::-1]
    for ax, (title, series) in zip(axes, panels):
        n = len(series)
        hgt = 0.8 / n
        top = 0
        for j, (lab, key, col) in enumerate(series):
            vals = np.array([k[key][2][i] for i in range(len(CLASSES))])
            top = max(top, vals.max())
            ax.barh(y + 0.4 - hgt * (j + 0.5), vals, height=hgt * 0.92, color=col, edgecolor=SURFACE,
                    linewidth=0.6, label=f"{lab} — {k[key][0]:,} img / {k[key][1]:,} bx")
        ax.set_xlim(0, top * 1.12)
        panel_title(ax, title)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_xlabel("Bounding boxes", fontsize=7)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.24), fontsize=5.8, handlelength=1, frameon=False)
    axes[0].set_yticks(y, [CLASS_LABEL[c] for c in CLASSES])
    fig.tight_layout(w_pad=1.6)
    fig.text(0.995, 0.0, "Raw = finding_nemo_raw · Curated = curated_combined · "
             "Reviewed = finding_nemo_human_reviewed_6cls", ha="right", va="top", fontsize=6.0, color=INK_2)
    save(fig, "fig21_dataset_versions")


def main():
    apply_style()
    k = counts()
    plt.rcParams["axes.grid"] = True
    fig_versions(k)
    plt.rcParams["axes.grid"] = False
    diag_project(k)
    diag_cross_label(k)


if __name__ == "__main__":
    main()
