"""
PIQA forward-patch transfer across models.

Self-contained, white background. Per-panel top-left legend (matching
original placement). Sequential coral gradient encodes layer depth across
all 5 model panels:
  light = earliest of the three sampled layers
  dark  = latest

Run:
    python fig_piqa_alpha_small_multiples_v2.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "neurips" / "v2" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Style ------------------------------------------------------

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Inter", "Inter Regular", "Inter Medium",
        "Styrene B", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans",
    ],
    "axes.unicode_minus": True,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Ink ramp
INK_PRIMARY   = "#1F1C19"
INK_SECONDARY = "#3F3A35"
INK_TITLE     = "#6B6660"   # soft-gray panel titles

SPINE_COLOR = "#1F1C19"     # near-black, thin
GRID_COLOR  = "#ECE8E1"     # very faint horizontal hairlines
RULE_COLOR  = "#C7C2BA"     # subtle dotted x=0 rule

# Sequential coral gradient: shallow → deep layer (relative within each model)
LAYER_PALETTE = ("#E89880", "#B85130", "#5C1F0E")
LAYER_MARKERS = ("o", "s", "D")


# ---------- Data (PIQA matched flip rate, %) --------------------------

ALPHAS = (0.0, 0.3, 0.5, 0.7, 1.0)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    layers: Tuple[int, int, int]
    series: Dict[int, Tuple[float, ...]]


SPECS: List[ModelSpec] = [
    ModelSpec(
        name="Qwen",
        layers=(2, 5, 10),
        series={
            2:  (0.0, 2.0,  9.8, 16.7, 19.6),
            5:  (0.0, 4.9,  8.8, 20.6, 20.6),
            10: (0.0, 4.9, 14.7, 18.6, 17.6),
        },
    ),
    ModelSpec(
        name="GPT-OSS",
        layers=(16, 18, 20),
        series={
            16: (0.0, 8.9, 6.9, 11.9, 18.8),
            18: (0.0, 7.9, 8.9, 10.9,  9.9),
            20: (0.0, 3.0, 7.9,  8.9, 13.9),
        },
    ),
    ModelSpec(
        name="Gemma",
        layers=(20, 22, 24),
        series={
            20: (0.0, 0.7, 1.4, 1.4, 1.4),
            22: (0.0, 0.7, 0.7, 0.7, 1.4),
            24: (0.0, 0.0, 0.0, 0.7, 0.7),
        },
    ),
    ModelSpec(
        name="OLMo-2",
        layers=(10, 16, 22),
        series={
            10: (0.0, 0.0,  6.2, 2.8, 2.8),
            16: (0.0, 0.0, 10.4, 2.1, 2.8),
            22: (0.0, 0.7,  6.2, 2.8, 4.2),
        },
    ),
    ModelSpec(
        name="OLMo-3.1",
        layers=(15, 18, 22),
        series={
            15: (0.0, 0.0, 4.2, 7.0, 25.4),
            18: (0.0, 0.0, 4.2, 5.6,  5.6),
            22: (0.0, 1.4, 4.2, 4.2,  2.8),
        },
    ),
]


# ---------- Helpers ----------------------------------------------------

def style_axes(ax: plt.Axes) -> None:
    """Editorial spines: top/right hidden, left/bottom thin near-black."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE_COLOR)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(
        axis="both", which="both",
        color=SPINE_COLOR, length=3.2, width=0.8,
        labelcolor=INK_SECONDARY, labelsize=9,
    )


# ---------- Figure -----------------------------------------------------

def main() -> None:
    n = len(SPECS)
    fig, axes = plt.subplots(
        1, n,
        figsize=(15.6, 3.55),
        sharey=True,
        gridspec_kw=dict(left=0.050, right=0.992,
                         bottom=0.18, top=0.86,
                         wspace=0.18),
    )
    fig.patch.set_facecolor("white")

    y_top = 28.0   # OLMo-3.1's L15 endpoint sits at 25.4

    for idx, (ax, spec) in enumerate(zip(axes, SPECS)):
        ax.set_facecolor("white")
        style_axes(ax)

        # Faint horizontal hairlines only — helps eye traverse panels
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

        # Subtle dotted x=0 rule
        ax.axvline(0.0, color=RULE_COLOR, linestyle=(0, (1.5, 2.5)),
                   linewidth=0.7, zorder=1)

        # Lines: shallow→deep layer maps to light→dark coral
        for li, layer in enumerate(spec.layers):
            xs = np.array(ALPHAS, dtype=float)
            ys = np.array(spec.series[layer], dtype=float)
            color = LAYER_PALETTE[li]
            ax.plot(
                xs, ys,
                color=color,
                linewidth=1.6,
                marker=LAYER_MARKERS[li],
                markersize=4.6,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                solid_capstyle="round",
                solid_joinstyle="round",
                label=f"L{layer}",
                zorder=4 + li,   # deepest layer line drawn on top
            )

        # Limits, ticks
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(0, y_top)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 5, 10, 15, 20, 25])

        # Per-panel legend, top-left, frameless — matches original placement
        ax.legend(
            loc="upper left",
            frameon=False,
            fontsize=9,
            handlelength=2.0,
            handletextpad=0.55,
            borderaxespad=0.4,
            labelcolor=INK_PRIMARY,
        )

        # Soft-gray, top-left panel title
        ax.set_title(
            spec.name,
            fontsize=14.5,
            fontweight="medium",
            color=INK_TITLE,
            pad=8,
            loc="left",
        )

        ax.set_xlabel("Patch strength (α)", fontsize=10.5,
                      color=INK_SECONDARY, labelpad=6)

    axes[0].set_ylabel("Matched PIQA flip rate (%)",
                       fontsize=10.5, color=INK_SECONDARY, labelpad=6)

    # Save
    stem = "piqa_alpha_small_multiples_v2"
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=240,
                facecolor="white", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf",
                facecolor="white", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.svg",
                facecolor="white", bbox_inches="tight")
    print(f"Saved {OUT_DIR / stem}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
