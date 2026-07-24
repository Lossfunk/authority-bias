"""
FactualQA forward-patch transfer across models.

Sister figure to fig_piqa_alpha_small_multiples_v2.py. Identical style
choices — sequential coral gradient (light = earliest of the three sampled
layers, dark = latest), per-panel top-left legend, white background, soft
gray titles, hairline horizontal grid.

Two data-level differences from the PIQA figure:
  1. OLMo-3.1 was sampled at α = 0.75 instead of α = 0.7. Each ModelSpec
     therefore carries its own alpha grid.
  2. Gemma's three sampled layers here are L15/L18/L20 (the FactualQA set),
     not L20/L22/L24 like in PIQA.

Run:
    python fig_forward_patch_small_multiples_v2.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

INK_PRIMARY   = "#1F1C19"
INK_SECONDARY = "#3F3A35"
INK_TITLE     = "#6B6660"

SPINE_COLOR = "#1F1C19"
GRID_COLOR  = "#ECE8E1"
RULE_COLOR  = "#C7C2BA"

# Sequential coral gradient: shallow → deep layer (same as PIQA figure)
LAYER_PALETTE = ("#E89880", "#B85130", "#5C1F0E")
LAYER_MARKERS = ("o", "s", "D")


# ---------- Data (FactualQA matched flip rate, %) ---------------------

DEFAULT_ALPHAS = (0.0, 0.3, 0.5, 0.7, 1.0)
OLMO31_ALPHAS  = (0.0, 0.3, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    layers: Tuple[int, int, int]
    series: Dict[int, Tuple[float, ...]]
    alphas: Tuple[float, ...] = DEFAULT_ALPHAS


SPECS: List[ModelSpec] = [
    ModelSpec(
        name="Qwen",
        layers=(2, 5, 10),
        series={
            2:  (0.0, 1.9, 10.1, 11.8, 16.1),
            5:  (0.0, 1.9,  6.6, 14.1, 12.6),
            10: (0.0, 2.8, 11.1, 14.1, 12.2),
        },
    ),
    ModelSpec(
        name="GPT-OSS",
        layers=(16, 18, 20),
        series={
            16: (0.0, 6.0, 8.9, 17.9, 32.7),
            18: (0.0, 6.0, 8.9,  8.9, 11.3),
            20: (0.0, 4.2, 3.0,  6.0,  4.2),
        },
    ),
    ModelSpec(
        name="Gemma",
        layers=(15, 18, 20),
        series={
            15: (0.0, 0.7, 1.1, 1.7, 1.3),
            18: (0.0, 0.1, 0.5, 0.1, 0.5),
            20: (0.0, 6.6, 6.6, 7.1, 7.9),
        },
    ),
    ModelSpec(
        name="OLMo-2",
        layers=(10, 16, 22),
        series={
            10: (0.0,  4.8, 16.7, 16.7, 16.7),
            16: (0.0,  3.6, 19.0, 15.5, 15.5),
            22: (0.0, 10.7, 16.7, 15.5, 15.5),
        },
    ),
    ModelSpec(
        name="OLMo-3.1",
        layers=(15, 18, 22),
        series={
            15: (0.0, 32.0, 38.1, 48.5, 58.8),
            18: (0.0, 23.7, 33.0, 36.1, 39.2),
            22: (0.0, 22.7, 25.8, 27.8, 29.9),
        },
        alphas=OLMO31_ALPHAS,
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

    y_top = 65.0   # OLMo-3.1's L15 endpoint sits at 58.8

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
            xs = np.array(spec.alphas, dtype=float)
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
        ax.set_yticks([0, 10, 20, 30, 40, 50, 60])

        # Per-panel legend, top-left, frameless
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

    axes[0].set_ylabel("Matched flip rate (%)",
                       fontsize=10.5, color=INK_SECONDARY, labelpad=6)

    stem = "forward_patch_alpha_small_multiples_v2"
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=240,
                facecolor="white", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf",
                facecolor="white", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.svg",
                facecolor="white", bbox_inches="tight")
    print(f"Saved {OUT_DIR / stem}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
