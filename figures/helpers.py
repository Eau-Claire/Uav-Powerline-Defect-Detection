"""Reusable chart primitives (dumbbell, annotated heatmap, model dot plot)."""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from common import (DIV_MID, DIV_NEG, DIV_POS, GRID, INK, INK_2, MODEL_LABEL, RAW_GRAY,
                    SEQ_BLUE, SURFACE, BLUE, model_marker)

SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
DIV_CMAP = LinearSegmentedColormap.from_list("div_rb", [DIV_NEG, DIV_MID, DIV_POS])


def dumbbell(ax, labels, before, after, *, before_label="Raw", after_label="Curated",
             after_color=BLUE, delta_fmt="{:+.1f}", scale=100.0, xlabel=None):
    """One row per label: gray 'before' dot -> coloured 'after' dot, delta printed right."""
    y = np.arange(len(labels))[::-1]
    for yi, b, a in zip(y, before, after):
        ax.plot([b * scale, a * scale], [yi, yi], color=GRID, lw=2.4, solid_capstyle="round", zorder=1)
    ax.scatter(np.asarray(before) * scale, y, s=34, color=RAW_GRAY, edgecolor=SURFACE,
               linewidth=0.8, zorder=3, label=before_label)
    ax.scatter(np.asarray(after) * scale, y, s=38, color=after_color, edgecolor=SURFACE,
               linewidth=0.8, zorder=4, label=after_label)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    lo = min(min(before), min(after)) * scale
    hi = max(max(before), max(after)) * scale
    pad = (hi - lo) * 0.12 + 0.5
    ax.set_xlim(lo - pad, hi + pad * 2.6)
    for yi, b, a in zip(y, before, after):
        ax.text(hi + pad * 1.2, yi, delta_fmt.format((a - b) * scale), va="center", ha="left",
                fontsize=7, color=INK if a >= b else "#c4302f")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", length=0)


def heatmap(ax, matrix, row_labels, col_labels, *, fmt="{:.1f}", scale=100.0, diverging=False,
            vmin=None, vmax=None, cbar_label=None, fig=None):
    """Annotated heatmap. Sequential blue by default, red-gray-blue when diverging."""
    m = np.asarray(matrix, dtype=float) * scale
    if diverging:
        lim = np.nanmax(np.abs(m)) if vmax is None else vmax
        lim = max(lim, 1e-9)
        im = ax.imshow(m, cmap=DIV_CMAP, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    else:
        im = ax.imshow(m, cmap=SEQ_CMAP, vmin=vmin if vmin is not None else np.nanmin(m),
                       vmax=vmax if vmax is not None else np.nanmax(m), aspect="auto")
    norm = im.norm
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            if np.isnan(v):
                continue
            if diverging and abs(v) < 0.05:
                v = 0.0  # avoid "-0.0"
            t = norm(v)
            dark = (abs(t - 0.5) > 0.32) if diverging else t > 0.55
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.8,
                    color=SURFACE if dark else INK)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, m.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, m.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    if fig is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6.5, length=2, color=INK_2)
        if cb_label := cbar_label:
            cb.set_label(cb_label, fontsize=7, color=INK_2)
    return im


def model_dotplot(ax, df, metrics, metric_labels, *, scale=100.0, xlabel="Score (%)"):
    """Rows = metrics; inside each row one marker per model, vertically offset."""
    models = list(df["model"])
    n = len(models)
    offsets = np.linspace(0.30, -0.30, n) if n > 1 else [0]
    y0 = np.arange(len(metrics))[::-1]
    for yi, met in zip(y0, metrics):
        ax.axhspan(yi - 0.42, yi + 0.42, color="#f6f5f2", zorder=0, lw=0)
        for off, (_, r) in zip(offsets, df.iterrows()):
            ax.plot(r[met] * scale, yi + off, linestyle="none", zorder=3, **model_marker(r["model"]))
        best = df[met].idxmax()
        ax.text(df.loc[best, met] * scale + 0.6, yi + offsets[list(df.index).index(best)],
                f"{df.loc[best, met] * scale:.1f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y0, metric_labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.6, len(metrics) - 0.4)


def model_label(key):
    return MODEL_LABEL.get(key, key)


def place_labels(ax, xs, ys, texts, *, dx=6, min_gap=9, fontsize=6.5, color=INK):
    """Label points to the right; nudge labels vertically (in points) so none overlap."""
    fig = ax.figure
    fig.canvas.draw()
    to_disp = ax.transData.transform
    pts = [(to_disp((x, y)), x, y, t) for x, y, t in zip(xs, ys, texts)]
    pts.sort(key=lambda p: -p[0][1])
    placed = []  # (disp_x, disp_y_of_label)
    scale = fig.dpi / 72.0
    for (px, py), x, y, t in pts:
        ly = py
        for _ in range(40):
            clash = [q for q in placed if abs(q[0] - px) < 90 * scale and abs(q[1] - ly) < min_gap * scale]
            if not clash:
                break
            ly = min(q[1] for q in clash) - min_gap * scale
        placed.append((px, ly))
        off = (ly - py) / scale
        ax.annotate(t, (x, y), xytext=(dx, off), textcoords="offset points", va="center",
                    fontsize=fontsize, color=color,
                    arrowprops=None if abs(off) < 4 else dict(arrowstyle="-", color=GRID, lw=0.6,
                                                               shrinkA=0, shrinkB=3))


def caption(fig, text):
    """Provenance line under the figure (kept out of the axes so it never hits tick labels)."""
    fig.text(0.995, 0.0, text, ha="right", va="top", fontsize=6.3, color=INK_2)
