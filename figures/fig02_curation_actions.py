"""Fig 2 - Number of images touched by each curation rule (train vs val_clean).

Source: data/curated_combined/manifest.json -> stats.{train,val_clean}.<rule>.
"""
import matplotlib.pyplot as plt

from common import BLUE, DATA, INK, SINGLE_COL, apply_style, read_json, save

RULES = [
    ("A_hflip", "A · Flip-inverted box fix"),
    ("C_person", "C · Person sample drop"),
    ("D_syn_fo", "D · Synthetic foreign-object purge"),
    ("E_quarantine", "E · Optical quarantine"),
    ("G_review", "G · Manual review fix"),
    ("H_tighten", "H · SAM box tightening"),
]


def main():
    apply_style()
    stats = read_json(DATA / "curated_combined" / "manifest.json")["stats"]
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL * 1.75, 1.9), sharey=True,
                             gridspec_kw={"width_ratios": [3, 1.3]})
    y = list(range(len(RULES)))[::-1]
    for ax, split, title in ((axes[0], "train", "Train"), (axes[1], "val_clean", "Val-clean")):
        vals = [stats[split].get(k, 0) for k, _ in RULES]
        ax.barh(y, vals, height=0.62, color=BLUE, edgecolor="white", linewidth=1)
        top = max(vals) or 1
        for yi, v in zip(y, vals):
            ax.text(v + top * 0.03, yi, "–" if v == 0 else f"{v:,}", va="center", fontsize=6.5, color=INK)
        ax.set_xlim(0, top * 1.25)
        ax.set_title(f"{title} ({stats[split]['images']:,} images)", fontsize=7.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    axes[0].set_yticks(y, [lab for _, lab in RULES])
    fig.supxlabel("Images affected", fontsize=8, y=0.02)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig02_curation_actions")


if __name__ == "__main__":
    main()
