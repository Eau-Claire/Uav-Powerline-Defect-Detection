"""Fig 1 - Box count per class in each split (curated train / val / val_clean / test_main_v1).

Sources: data/curated_combined/manifest.json (stats) and data/test_main_v1/distribution.csv.
"""
import matplotlib.pyplot as plt

from common import (BLUE, CLASSES, CLASS_LABEL, DATA, DOUBLE_COL, INK, apply_style, panel_title,
                    read_csv, read_json, save)


def load():
    stats = read_json(DATA / "curated_combined" / "manifest.json")["stats"]
    splits = {}
    for split, title in (("train", "Train (curated)"), ("val", "Validation"), ("val_clean", "Val-clean")):
        s = stats[split]
        splits[title] = ({c: s.get(f"box:{c}", 0) for c in CLASSES}, s["images"])
    dist = read_csv(DATA / "test_main_v1" / "distribution.csv").set_index("class")
    manifest = read_csv(DATA / "test_main_v1" / "manifest.csv")
    splits["Test (test_main_v1)"] = ({c: int(dist.loc[c, "boxes"]) for c in CLASSES}, len(manifest))
    return splits


def main():
    apply_style()
    splits = load()
    fig, axes = plt.subplots(1, len(splits), figsize=(DOUBLE_COL, 2.1), sharey=True)
    y = list(range(len(CLASSES)))[::-1]
    for ax, (title, (counts, n_img)) in zip(axes, splits.items()):
        vals = [counts[c] for c in CLASSES]
        total = sum(vals)
        ax.barh(y, vals, height=0.62, color=BLUE, edgecolor="white", linewidth=1)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.03, yi, f"{v:,}", va="center", fontsize=6.5, color=INK)
        ax.set_xlim(0, max(vals) * 1.32)
        panel_title(ax, title, f"{n_img:,} images · {total:,} boxes")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    axes[0].set_yticks(y, [CLASS_LABEL[c] for c in CLASSES])
    fig.supxlabel("Bounding boxes", fontsize=8, y=0.01)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig01_dataset_distribution")


if __name__ == "__main__":
    main()
