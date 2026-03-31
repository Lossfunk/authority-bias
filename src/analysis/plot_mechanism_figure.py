#!/usr/bin/env python3
"""Mechanistic study figure for NeurIPS paper.

Creates a 3-panel figure summarizing the correction-gating mechanism:
  Panel A: The correction-gating axis — i1a/i1d/i1c projections + within-i1a distribution
  Panel B: Patching results — real vs controls
  Panel C: Evidence specificity — C1 vs W1 vs N0

Style: Clean, minimal, inspired by the Assistant Axis paper (Lu et al., 2026).
Palette: Okabe-Ito colorblind-friendly.

Run:
  uv run python -m src.analysis.plot_mechanism_figure
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Okabe-Ito palette ──
OI = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "gray": "#999999",
    "black": "#000000",
}

plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
})


def panel_a(ax):
    """Panel A: The correction-gating axis.

    Shows i1a/i1d/i1c instruction projections and within-i1a
    entrenching vs correcting distributions on a single 1D axis.
    Inspired by Assistant Axis Figure 3 histogram style.
    """
    # ── Real data from activation projections ──
    data = np.load("figures/mechanism_projections.npz")
    entrenching = data["i1a_entrenching"]   # n=151, mean=-9.48
    correcting = data["i1a_correcting"]     # n=298, mean=-1.55
    i1c_all = data["i1c_all"]               # n=449, mean=0.48

    # Instruction-level mean projections (endorsement position, layer 23)
    i1a_mean = np.mean(np.concatenate([entrenching, correcting]))
    i1c_mean = np.mean(i1c_all)
    i1d_proj = +2.4  # from findings.md

    # ── Plot ──
    bins = np.linspace(-18, 16, 50)

    # Histograms for within-i1a distributions
    ax.hist(entrenching, bins=bins, alpha=0.45, color=OI["vermillion"],
            density=True, label="Entrenching (within i1a)")
    ax.hist(correcting, bins=bins, alpha=0.45, color=OI["bluish_green"],
            density=True, label="Correcting (within i1a)")

    # Also show i1c distribution lightly
    ax.hist(i1c_all, bins=bins, alpha=0.25, color=OI["sky_blue"],
            density=True, label="All items (i1c)")

    # Instruction projections as vertical lines with labels
    ymax = ax.get_ylim()[1]
    for name, proj, color, ha, va_offset in [
        ("i1a", i1a_mean, OI["vermillion"], "right", 0.95),
        ("i1d", i1d_proj, OI["orange"], "center", 0.85),
        ("i1c", i1c_mean, OI["bluish_green"], "left", 0.95),
    ]:
        ax.axvline(proj, color=color, linestyle="--", linewidth=2.0, alpha=0.9)
        ax.text(
            proj, ymax * va_offset, f"  {name}\n  ({proj:+.1f})",
            color=color, fontsize=9, fontweight="bold",
            ha=ha, va="top",
        )

    # Axis labels and arrows
    ax.set_xlabel("Projection onto correction-gating direction", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("A. Representational geometry", fontsize=12, fontweight="bold", loc="left")

    # Semantic arrows at bottom
    arrow_y = -ymax * 0.18
    ax.annotate(
        "", xy=(-16, arrow_y), xytext=(-5, arrow_y),
        arrowprops=dict(arrowstyle="<-", color=OI["vermillion"], lw=1.5),
        annotation_clip=False,
    )
    ax.text(-16, arrow_y - ymax * 0.06, "Prior-preserving",
            color=OI["vermillion"], fontsize=9, fontstyle="italic",
            ha="left", va="top")

    ax.annotate(
        "", xy=(14, arrow_y), xytext=(5, arrow_y),
        arrowprops=dict(arrowstyle="<-", color=OI["bluish_green"], lw=1.5),
        annotation_clip=False,
    )
    ax.text(14, arrow_y - ymax * 0.06, "Evidence-integrating",
            color=OI["bluish_green"], fontsize=9, fontstyle="italic",
            ha="right", va="top")

    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(-18, 16)


def panel_b(ax):
    """Panel B: Patching results — real vector vs controls.

    Bar chart showing margin shift for each condition.
    """
    conditions = [
        "Real vector\n(endorsement, L23)",
        "Random\nvector",
        "Opposite\ndirection",
        "Wrong position\n(instruction)",
    ]
    shifts = [+1.93, -0.50, -1.33, -0.28]
    colors = [OI["bluish_green"], OI["gray"], OI["vermillion"], OI["gray"]]
    flip_rates = ["8.6%", "0%", "0%", "0%"]
    pct_positive = ["87%", "5%", "1%", "24%"]

    bars = ax.barh(range(len(conditions)), shifts, color=colors, height=0.6,
                   edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions, fontsize=9)
    ax.set_xlabel("Logit margin shift (toward correct)", fontsize=11)
    ax.set_title("B. Causal patching (alpha = 4)", fontsize=12, fontweight="bold", loc="left")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")

    # Annotate with flip rates
    for i, (shift, flip, pct) in enumerate(zip(shifts, flip_rates, pct_positive)):
        x_pos = shift + 0.08 if shift >= 0 else shift - 0.08
        ha = "left" if shift >= 0 else "right"
        ax.text(x_pos, i, f"{shift:+.2f}", fontsize=9, fontweight="bold",
                va="center", ha=ha, color=colors[i])

    # Add "flips" annotation for real vector
    ax.text(2.1, 0, "13/151 flipped\n0 harmed", fontsize=8,
            va="center", ha="left", color=OI["bluish_green"],
            fontstyle="italic")

    ax.set_xlim(-2.0, 3.2)
    ax.invert_yaxis()


def panel_c(ax):
    """Panel C: Evidence specificity — effect depends on endorsement presence.

    Shows the 30x difference between C1, W1, and N0 conditions.
    """
    conditions = ["No endorsement\n(N0)", "Wrong endorsement\n(W1)", "Correct endorsement\n(C1)"]
    shifts = [-0.07, +0.30, +1.93]
    colors = [OI["gray"], OI["orange"], OI["bluish_green"]]

    bars = ax.bar(range(len(conditions)), shifts, color=colors, width=0.6,
                  edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Logit margin shift", fontsize=11)
    ax.set_title("C. Evidence specificity", fontsize=12, fontweight="bold", loc="left")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

    # Annotate values
    for i, (shift, color) in enumerate(zip(shifts, colors)):
        y_pos = shift + 0.06 if shift >= 0 else shift - 0.06
        va = "bottom" if shift >= 0 else "top"
        ax.text(i, y_pos, f"{shift:+.2f}", fontsize=10, fontweight="bold",
                va=va, ha="center", color=color)

    # Add the "30x" annotation with bracket
    ax.plot([0, 0, 2, 2], [0.15, 1.05, 1.05, 2.05], color=OI["blue"],
            lw=1.5, alpha=0.7)
    ax.text(1.0, 1.15, "30x", fontsize=15, fontweight="bold",
            color=OI["blue"], ha="center", va="bottom")

    # Subtitle
    ax.text(1.0, -0.55, "The vector requires evidence to operate on",
            fontsize=9, fontstyle="italic", color=OI["gray"],
            ha="center", va="top")

    ax.set_ylim(-0.7, 2.3)


def main():
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(18, 5.5))

    # Three panels: A (wider), B, C
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 0.9], wspace=0.35)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)

    # Suptitle
    fig.suptitle(
        "The correction-gating mechanism: a single direction gates evidence integration",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for fmt in ["png", "pdf"]:
        path = output_dir / f"mechanism_figure.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
