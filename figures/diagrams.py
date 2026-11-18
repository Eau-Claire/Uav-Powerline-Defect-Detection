"""Method diagrams (vector PDF + PNG) -> figures/out/06_method/.

diag01_pipeline           End-to-end data-centric methodology
diag02_curation_modules   Curation Modules I-IV and their rules (train split only)
diag03_ablation_design    Cumulative and leave-one-out ablation design
diag04_test_protocol      Construction of the test sets and the audit lineage
diag05_audit_loop         Targeted label-audit loop on the test set

Facts come from the repository, not from memory:
  rules/thresholds  benchmakring_.../scripts/build_curation_module_ablation.py,
                    build_curation_loo_ablation.py
  test sets         benchmakring_.../CLEAN_TEST_PROTOCOL.md, data/*/freeze.json
  counts            data/curated_combined/manifest.json, test_main_v5 freeze.json
Where docs/RESEARCH_REPORT_LOO_CURATION_ABLATION.md disagrees with the build scripts
(it calls G "non-grid drop" and Module 4 "canonical harmonisation") the scripts win.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import (AQUA, BLUE, DATA, INK, INK_2, INK_3, ORANGE, RAW_GRAY, SURFACE, apply_style,
                    read_json, save)

BORDER = "#d6d5cf"  # box outline; paper figures override this for monochrome print

# role -> (accent colour, fill)
ROLE = {
    "data": (RAW_GRAY, "#f4f3f0"),
    "process": (BLUE, "#eef4fc"),
    "model": (ORANGE, "#fdf0ea"),
    "eval": (AQUA, "#e9f7f1"),
    "note": (INK_3, SURFACE),
}


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), role="process", title_size=7.2, body_size=6.2, dashed=False):
    """Rounded box with a coloured accent bar on the left. (x, y) = top-left corner."""
    accent, fill = ROLE[role]
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=fill, edgecolor=accent if dashed else BORDER,
                                linewidth=0.9 if dashed else 0.6, linestyle=(0, (3, 2)) if dashed else "-",
                                zorder=2))
    ax.add_patch(FancyBboxPatch((x, y - h), 0.05, h, boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor=accent, edgecolor="none", zorder=3))
    ax.text(x + 0.12, y - 0.09, title, fontsize=title_size, fontweight="bold", color=INK,
            va="top", ha="left", zorder=4)
    if lines:
        ax.text(x + 0.12, y - 0.09 - title_size / 72 * 1.45, "\n".join(lines), fontsize=body_size,
                color=INK_2, va="top", ha="left", linespacing=1.35, zorder=4)
    return dict(x=x, y=y, w=w, h=h)


def anchor(b, side):
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    return {"l": (x, y - h / 2), "r": (x + w, y - h / 2), "t": (x + w / 2, y), "b": (x + w / 2, y - h)}[side]


def arrow(ax, p, q, label=None, color=INK_3, rad=0.0, dashed=False, label_offset=(0, 0.07)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, color=color, lw=0.9,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=(0, (3, 2)) if dashed else "-",
                                 shrinkA=1, shrinkB=1, zorder=1))
    if label:
        mx, my = (p[0] + q[0]) / 2 + label_offset[0], (p[1] + q[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=5.8, color=INK_2, ha="center", va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", facecolor=SURFACE, edgecolor="none"))


def elbow(ax, points, color=INK_3):
    """Orthogonal connector through the given points; arrowhead on the last segment."""
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=0.9, solid_joinstyle="round", zorder=1)
    arrow(ax, points[-2], points[-1], color=color)


def lane(ax, x, y, w, h, label):
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor="none", edgecolor="#e2e1db", linewidth=0.8, zorder=0))
    ax.text(x + 0.08, y - 0.06, label.upper(), fontsize=5.6, color=INK_3, va="top", ha="left",
            fontweight="bold")


def legend(ax, x, y, roles=("data", "process", "model", "eval")):
    names = {"data": "Data", "process": "Processing", "model": "Training", "eval": "Evaluation"}
    for i, r in enumerate(roles):
        ax.add_patch(FancyBboxPatch((x + i * 1.05, y - 0.1), 0.14, 0.1, boxstyle="round,pad=0,rounding_size=0.02",
                                    facecolor=ROLE[r][0], edgecolor="none"))
        ax.text(x + i * 1.05 + 0.2, y - 0.05, names[r], fontsize=6.2, color=INK_2, va="center")


# --------------------------------------------------------------------------- diag01
def diag_pipeline():
    fig, ax = canvas(7.16, 4.0)
    legend(ax, 0.15, 3.95)
    top = 3.62
    lane(ax, 0.08, top, 1.55, 3.5, "1 · Data")
    src = box(ax, 0.16, 3.38, 1.39, 1.12, "Public sources", [
        "ZHENG1600 · IDID · CPLID", "VPMBGI · InsuFault", "FOTL / EFOD_Drone",
        "V6.5: Open Images", "composites, Roboflow"], role="data")
    canon = box(ax, 0.16, 1.98, 1.39, 1.5, "V6.2E canonical set", [
        "6 classes", "taxonomy → bbox → QA", "→ dedup → split → freeze", "",
        "train 18,334 · val 2,272", "test 2,012 rows"], role="data")
    arrow(ax, anchor(src, "b"), anchor(canon, "t"))

    lane(ax, 1.71, top, 1.6, 3.5, "2 · Curation (train only)")
    raw = box(ax, 1.79, 3.38, 1.44, 0.5, "Raw train", ["18,334 images · Raw arm"], role="data")
    cur = box(ax, 1.79, 2.68, 1.44, 1.0, "Curation Modules", [
        "I    A flip · B missing · F tiny", "II   C person · D synthetic",
        "III  E quarantine · G review", "IV  H SAM tightening"], role="process", body_size=5.9)
    curd = box(ax, 1.79, 1.48, 1.44, 0.5, "Curated train", ["17,903 images · Curated arm"], role="data")
    arrow(ax, anchor(raw, "b"), anchor(cur, "t"))
    arrow(ax, anchor(cur, "b"), anchor(curd, "t"))
    arrow(ax, (1.55, 2.95), anchor(raw, "l"), rad=-0.35)
    ax.text(2.51, 0.62, "val and test are\nnever curated", fontsize=5.9, color=INK_2, ha="center",
            va="center", style="italic")

    lane(ax, 3.39, top, 1.6, 3.5, "3 · Training")
    det = box(ax, 3.47, 3.38, 1.44, 2.55, "Six detectors × 2 arms", [
        "YOLO11n · YOLO11m", "YOLO26n · YOLO26m", "RT-DETR-R50vd", "RT-DETR-R101vd", "",
        "same recipe in both arms:", "640 px · 50 epochs · AMP", "AdamW · cosine LR", "seed 42",
        "(seed 36 replication)"], role="model")
    arrow(ax, anchor(raw, "r"), (3.47, 3.13))
    arrow(ax, anchor(curd, "r"), (3.47, 1.23))

    lane(ax, 5.07, top, 2.0, 3.5, "4 · Selection & evaluation")
    sel = box(ax, 5.15, 3.38, 1.84, 0.5, "Select checkpoint", ["on validation only"], role="process")
    ev = box(ax, 5.15, 2.68, 1.84, 0.95, "Evaluate once (frozen)", [
        "test_main_v1 · 1,931 (primary)", "operational min-8 · 1,871", "no tuning on test"], role="eval")
    met = box(ax, 5.15, 1.55, 1.84, 1.0, "Report", [
        "P · R · F1 · mAP@0.5", "mAP@0.5:0.95 · MEAN_5", "latency · FPS · VRAM"], role="eval")
    arrow(ax, (4.91, 3.13), anchor(sel, "l"))
    arrow(ax, anchor(sel, "b"), anchor(ev, "t"))
    arrow(ax, anchor(ev, "b"), anchor(met, "t"))
    save(fig, "diag01_pipeline")


# --------------------------------------------------------------------------- diag02
def diag_curation():
    stats = read_json(DATA / "curated_combined" / "manifest.json")["stats"]["train"]
    fig, ax = canvas(7.16, 2.75)
    top, h = 2.65, 1.0
    raw = box(ax, 0.1, top, 0.92, h, "Raw train", ["labels", "", "18,334", "images"], role="data")
    w, gap, x0 = 1.12, 0.13, 1.15
    mods = [
        ("Module I", ["Annotation geometry", f"A undo h-flip ({stats['A_hflip']})", "B add human-",
                      "   reviewed boxes", "F drop GT < 4 px"]),
        ("Module II", ["Semantic filtering", f"C drop person-as-FO", f"   boxes ({stats['C_person']})",
                       f"D purge synthetic", f"   FO images ({stats['D_syn_fo']})"]),
        ("Module III", ["Review & quarantine", "E quarantine", f"   images ({stats['E_quarantine']})",
                        "G other reviewed", f"   box drops ({stats['G_review']})"]),
        ("Module IV", ["Box tightening", "H SAM-tighten", "   insulator & FO,", "   IoU 0.50–0.85,",
                       f"   never grows ({stats['H_tighten']})"]),
    ]
    mb = []
    prev = raw
    for i, (name, lines) in enumerate(mods):
        b = box(ax, x0 + i * (w + gap), top, w, h, name, lines, role="process", body_size=5.7)
        arrow(ax, anchor(prev, "r"), anchor(b, "l"))
        mb.append(b)
        prev = b
    out = box(ax, x0 + 4 * (w + gap), top, 0.93, h, "Curated", ["train", "", "17,903", "images"], role="data")
    arrow(ax, anchor(prev, "r"), anchor(out, "l"))

    iy, ih = 1.25, 0.5
    judge = box(ax, x0, iy, 1.5, ih, "YOLO11n judge audit", ["flips · missing · tiny · unseen GT"],
                role="note", dashed=True, body_size=5.7)
    review = box(ax, x0 + 1.75, iy, 1.5, ih, "Human review", ["drop / add / quarantine decisions"],
                 role="note", dashed=True, body_size=5.7)
    sam = box(ax, x0 + 3 * (w + gap), iy, 1.3, ih, "SAM proposals", ["insulator, foreign object"],
              role="note", dashed=True, body_size=5.7)
    def up(src, dst, label):
        p = (src["x"] + src["w"] / 2, src["y"])
        q = (dst["x"] + dst["w"] / 2, dst["y"] - dst["h"])
        arrow(ax, p, q, dashed=True)
        ax.text((p[0] + q[0]) / 2 + 0.05, (p[1] + q[1]) / 2, label, fontsize=5.6, color=INK_2,
                va="center", ha="left")
    up(judge, mb[0], "A, F (D uses unseen-GT flags)")
    up(review, mb[2], "B, C, E, G")
    up(sam, mb[3], "H")
    notes = ("Train split only: val and val_clean are byte-identical links to the raw data.  "
             "Model output is evidence, never ground truth (B adds only human-accepted boxes).\n"
             "Judge thresholds: missing conf ≥ 0.60, IoU ≤ 0.30; person conf ≥ 0.60.  "
             "Numbers in brackets: train images affected (data/curated_combined/manifest.json).")
    ax.text(0.1, 0.5, notes, fontsize=5.8, color=INK_2, va="top", linespacing=1.6)
    save(fig, "diag02_curation_modules")


# --------------------------------------------------------------------------- diag03
def diag_ablation():
    fig, ax = canvas(7.16, 2.75)
    ax.text(0.12, 2.68, "A · Cumulative stages (add one module at a time)", fontsize=7.2, fontweight="bold", va="top")
    stages = [("m0_raw", "no curation", False), ("m1_a_b_f", "Module I", True), ("m2_c_d", "I + II", True),
              ("m3_e_g", "I + II + III", True), ("m4_h", "I + II + III + IV", False)]
    prev = None
    for i, (k, lab, reported) in enumerate(stages):
        b = box(ax, 0.12 + i * 1.4, 2.4, 1.24, 0.55, k, [lab] + ([] if reported else ["built, not reported"]),
                role="process" if reported else "note", dashed=not reported, body_size=5.7)
        if prev:
            arrow(ax, anchor(prev, "r"), anchor(b, "l"))
        prev = b
    ax.text(0.12, 1.7, "B · Leave-one-out from the full set I + II + III + IV", fontsize=7.2, fontweight="bold", va="top")
    loo = [("m_234", "without Module I", "A, B, F"), ("m_134", "without Module II", "C, D"),
           ("m_124", "without Module III", "E, G")]
    for i, (k, lab, rules) in enumerate(loo):
        box(ax, 0.12 + i * 1.62, 1.42, 1.48, 0.55, k, [f"{lab} (drops {rules})"], role="process", body_size=5.7)
    box(ax, 5.05, 1.7, 2.0, 0.83, "Held fixed in every run", [
        "YOLO26m · 640 px · 50 epochs", "AdamW · cosine · AMP · seed 42", "batch 8 → nbs 64",
        "identical val / val_clean files"], role="model", body_size=5.7)
    box(ax, 0.12, 0.62, 6.93, 0.48, "Evaluated on", [
        "val_clean, 2,271 images (fig16, LOO report)   ·   1,931-image test of the v5 audit bundle "
        "(fig15: m1, m2, m3 and the three LOO runs)"], role="eval", body_size=5.9)
    save(fig, "diag03_ablation_design")


# --------------------------------------------------------------------------- diag04
def diag_test_protocol():
    v5 = read_json(DATA / "test_main_v5_full_consensus_assisted" / "freeze.json")
    op1 = read_json(DATA / "test_operational_min8_v1" / "freeze.json")
    fig, ax = canvas(7.16, 3.3)
    a = box(ax, 0.12, 3.2, 1.55, 0.5, "V6.2E TEST", ["2,012 rows as released"], role="data")
    b = box(ax, 0.12, 2.5, 1.55, 0.5, "Deduplicate", ["SHA-256 · −50 duplicate rows"], role="process")
    c = box(ax, 0.12, 1.8, 1.55, 0.62, "Remove person-as-FO", ["31 images checked by eye,", "whole image removed"], role="process")
    m = box(ax, 0.12, 0.98, 1.55, 0.5, "test_main_v1", ["1,931 images · frozen"], role="eval")
    for p, q in ((a, b), (b, c), (c, m)):
        arrow(ax, anchor(p, "b"), anchor(q, "t"))

    lane(ax, 1.95, 3.25, 5.12, 1.12, "Independent · headline results")
    op = box(ax, 2.05, 2.98, 2.3, 0.78, "Operational min-8 v1", [
        f"drop an image if any GT box has a short side", f"< {op1['minimum_side_pixels']:.0f} px at 640 · "
        f"{op1['source_images']:,} → {op1['kept_images']:,} images"], role="eval", body_size=5.8)
    ax.text(4.6, 2.86, "test_main_v1 = primary benchmark\noperational min-8 v1 = secondary\n"
                       "no label edits · no sampling · no tuning", fontsize=5.9, color=INK_2, va="top", linespacing=1.5)

    lane(ax, 1.95, 1.98, 5.12, 1.55, "Exploratory · model-assisted · never replaces test_main_v1")
    vers = [("v2", ["model-assisted", "box proposals"]), ("v3", ["model-aligned", "broken-strand boxes"]),
            ("v4", ["consensus-assisted", "(partial)"]),
            ("v5", [f"+{v5['added_insulator_boxes']} insulator boxes", f"{v5['minimum_consensus_models']}/4 YOLO agree",
                    f"mean conf ≥ {v5['minimum_mean_confidence']}"])]
    prev = None
    for i, (k, lab) in enumerate(vers):
        bb = box(ax, 2.05 + i * 1.25, 1.68, 1.12, 0.78, f"test_main_{k}", lab, role="process", body_size=5.7)
        if prev:
            arrow(ax, anchor(prev, "r"), anchor(bb, "l"))
        prev = bb
        if i == 0:
            first = bb
    ax.text(2.05, 0.75, "Each version also gets an operational min-8 subset; v3, v4 and v5 subsets are evaluated "
                        "for all six models (fig13).", fontsize=5.8, color=INK_2, style="italic", va="top")
    mx, my = anchor(m, "r")
    elbow(ax, [(mx, my), (1.81, my), (1.81, anchor(first, "l")[1]), anchor(first, "l")])
    elbow(ax, [(1.81, anchor(first, "l")[1]), (1.81, anchor(op, "l")[1]), anchor(op, "l")])
    save(fig, "diag04_test_protocol")


# --------------------------------------------------------------------------- diag05
def diag_audit_loop():
    fig, ax = canvas(7.16, 1.75)
    steps = [
        ("Predict", ["selected checkpoints", "on test_main_v1", "NMS IoU 0.5"], "model"),
        ("Log errors", ["match at IoU 0.5", "TP · FP · FN", "confusion · overlaps"], "process"),
        ("Select", ["insulator and", "broken_strand first,", "small / thin objects"], "process"),
        ("Human review", ["Streamlit tool:", "GT + predictions", "accept / reject"], "data"),
        ("Correction log", ["new test version", "manifest + hashes", "v1 never overwritten"], "data"),
        ("Re-evaluate", ["six models, once", "reported apart from", "test_main_v1"], "eval"),
    ]
    w, gap = 1.05, 0.12
    bs = []
    for i, (t, lines, role) in enumerate(steps):
        b = box(ax, 0.08 + i * (w + gap), 1.65, w, 0.72, t, lines, role=role, body_size=5.8)
        if bs:
            arrow(ax, anchor(bs[-1], "r"), anchor(b, "l"))
        bs.append(b)
    arrow(ax, anchor(bs[-1], "b"), anchor(bs[1], "b"), rad=-0.12, dashed=True)
    ax.text(3.6, 0.42, "next audit round (v2 → v5)", fontsize=5.9, color=INK_2, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.12", facecolor=SURFACE, edgecolor="none"))
    ax.text(0.1, 0.12, "Proposals come from the evaluated model families, so corrected sets can favour them; they stay "
                       "exploratory until an independent human pass.", fontsize=5.8, color=INK_2, style="italic")
    save(fig, "diag05_audit_loop")


def main():
    apply_style()
    plt.rcParams["axes.grid"] = False
    diag_pipeline()
    diag_curation()
    diag_ablation()
    diag_test_protocol()
    diag_audit_loop()


if __name__ == "__main__":
    main()
