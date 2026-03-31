#!/usr/bin/env python3
"""Clean scientific plot of the correction-gating axis.

Just the data: KDE distributions, axis markers, semantic labels.
No cards, callouts, or editorial elements — those go in the HTML figure.

Run:
  uv run python -m src.analysis.plot_mechanism_axis
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde

# ── Muted palette ──
P = {
    "entrench": "#C44E52",
    "entrench_fill": "#F0CCC9",
    "correct": "#4C72B0",
    "correct_fill": "#CCDAEC",
    "i1c": "#55A868",
    "i1c_fill": "#DCE9DF",
    "i1d": "#C4A94D",
    "bg_left": "#FAF0EE",
    "bg_right": "#EDF1F8",
    "axis": "#4A443D",
    "muted": "#8A8279",
    "text": "#2F2924",
    "bg": "#FBF8F3",
}

AXIS_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "cg", [P["entrench"], "#B49AAF", P["correct"]])

plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.facecolor": P["bg"],
})


def main():
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    data = np.load("figures/mechanism_projections.npz")
    entrenching = data["i1a_entrenching"]
    correcting = data["i1a_correcting"]
    i1c_all = data["i1c_all"]

    all_i1a = np.concatenate([entrenching, correcting])
    i1a_mean = float(np.mean(all_i1a))
    i1c_mean = float(np.mean(i1c_all))
    i1d_proj = 2.4

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(13, 5))

    xlim = (-22, 16)
    ax.set_xlim(xlim)

    # Semantic background — very subtle
    mid = (i1a_mean + i1c_mean) / 2
    ax.axvspan(xlim[0], mid, color=P["bg_left"], alpha=0.55, zorder=0)
    ax.axvspan(mid, xlim[1], color=P["bg_right"], alpha=0.55, zorder=0)

    # ── KDEs — desaturated fills, crisp outlines ──
    x = np.linspace(xlim[0], xlim[1], 500)
    kde_e = gaussian_kde(entrenching, bw_method=0.33)
    kde_c = gaussian_kde(correcting, bw_method=0.30)
    kde_i = gaussian_kde(i1c_all, bw_method=0.32)

    ye, yc, yi = kde_e(x), kde_c(x), kde_i(x)
    s = 0.90 / max(ye.max(), yc.max(), yi.max())
    ye *= s; yc *= s; yi *= s

    ax.fill_between(x, 0, ye, color=P["entrench_fill"], alpha=0.70, zorder=1)
    ax.plot(x, ye, color=P["entrench"], lw=1.6, alpha=0.85, zorder=2)

    ax.fill_between(x, 0, yc, color=P["correct_fill"], alpha=0.65, zorder=1)
    ax.plot(x, yc, color=P["correct"], lw=1.6, alpha=0.85, zorder=2)

    ax.fill_between(x, 0, yi, color=P["i1c_fill"], alpha=0.35, zorder=1)
    ax.plot(x, yi, color=P["i1c"], lw=1.3, ls=(0, (4, 2)), alpha=0.7, zorder=2)

    # ── Rug: colored scatter along axis ──
    rng = np.random.default_rng(7)
    jitter = rng.normal(scale=0.007, size=len(all_i1a))
    norm = mcolors.Normalize(vmin=xlim[0], vmax=xlim[1])
    ax.scatter(all_i1a, -0.022 + jitter, c=AXIS_CMAP(norm(all_i1a)),
               s=10, alpha=0.40, edgecolors="none", zorder=3)

    # ── Axis line ──
    ax.axhline(0, color=P["axis"], lw=1.0, zorder=4)

    # ── Tick marks — lighter ──
    for t in np.arange(-20, 16, 5):
        if xlim[0] < t < xlim[1]:
            ax.plot([t, t], [-0.008, 0.008], color=P["muted"], lw=0.7, zorder=5)
            ax.text(t, -0.045, str(int(t)), fontsize=7.5, color=P["muted"],
                    ha="center", va="top")

    # ── Instruction diamonds — offset labels to avoid overlap ──
    # i1c and i1d are close (0.48 and 2.4), so stagger vertically
    label_specs = [
        ("i1a", i1a_mean, P["entrench"], 0.82, "center"),
        ("i1c", i1c_mean, P["i1c"], 0.82, "center"),
        ("i1d", i1d_proj, P["i1d"], 0.68, "center"),  # lower to avoid i1c
    ]
    for name, proj, color, label_y, ha in label_specs:
        ax.plot(proj, 0, "D", markersize=12, color=color,
                markeredgecolor="white", markeredgewidth=1.8, zorder=10)
        ax.plot([proj, proj], [0.015, label_y - 0.05], color=color, lw=0.6,
                ls=":", alpha=0.35, zorder=5)
        ax.text(proj, label_y, name, fontsize=13, fontweight="bold",
                color=color, ha=ha, va="bottom", zorder=10)

    # ── Semantic arrows — compact ──
    arr_y = -0.09
    ax.annotate("", xy=(xlim[0]+1, arr_y), xytext=(-10, arr_y),
                arrowprops=dict(arrowstyle="->", color=P["entrench"], lw=1.8,
                                mutation_scale=15))
    ax.text(xlim[0]+1.5, arr_y - 0.018, "Prior-preserving",
            fontsize=10.5, fontweight="bold", color=P["entrench"], va="top")

    ax.annotate("", xy=(xlim[1]-1, arr_y), xytext=(8, arr_y),
                arrowprops=dict(arrowstyle="->", color=P["correct"], lw=1.8,
                                mutation_scale=15))
    ax.text(xlim[1]-1.5, arr_y - 0.018, "Evidence-integrating",
            fontsize=10.5, fontweight="bold", color=P["correct"], ha="right", va="top")

    # ── X label ──
    ax.text(np.mean(xlim), -0.145, "Projection onto correction-gating direction",
            fontsize=9, color=P["muted"], ha="center", fontstyle="italic")

    # ── Legend — compact, top-left ──
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=P["entrench"], lw=1.6,
               label=f"Entrenching (n={len(entrenching)})"),
        Line2D([0], [0], color=P["correct"], lw=1.6,
               label=f"Correcting (n={len(correcting)})"),
        Line2D([0], [0], color=P["i1c"], lw=1.3, ls="--",
               label=f"i1c distribution (n={len(i1c_all)})"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=8.5,
                    frameon=True, facecolor="white", edgecolor="#DDD",
                    framealpha=0.9, borderpad=0.6, handlelength=1.8)
    leg.get_frame().set_linewidth(0.5)

    # ── Clean up ──
    ax.set_ylim(-0.17, 0.97)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    for fmt in ["png", "pdf", "svg"]:
        path = output_dir / f"mechanism_axis.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=300, facecolor=P["bg"])
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
