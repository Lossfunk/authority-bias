#!/usr/bin/env python3
"""Option A: VA recovery and overlap summary figure.

Run:
  uv run python -m src.analysis.plot_option_a_va_recovery_overlap
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VA_SUMMARY_PATH = Path("results/authority/mechanism/lexical_va_plane/summary.json")
ASSISTANT_AXIS_OVERLAP_PATH = Path("external/assistant-axis/hardened/overlap_summary.json")
OUTPUT_DIR = Path("figures/neurips/option-a")

COLORS = {
    "valence": "#4C72B0",
    "arousal": "#DD8452",
    "plane": "#7A68A6",
    "assistant": "#55A868",
    "grid": "#E7E7E7",
    "axis": "#BDBDBD",
}


plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def _style_axis(ax: plt.Axes, *, zero_y: bool = False) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("black")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.45)
    if zero_y:
        ax.axhline(0.0, color=COLORS["axis"], linewidth=0.8, zorder=1)
    ax.tick_params(width=0.7, length=2.7, pad=1.5)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    va_summary = json.loads(VA_SUMMARY_PATH.read_text())
    overlap_summary = json.loads(ASSISTANT_AXIS_OVERLAP_PATH.read_text())

    layers = [int(layer) for layer in va_summary["target_layers"]]
    layer_stats = va_summary["summary_by_layer"]
    valence_corr = [layer_stats[str(layer)]["valence_corr"] for layer in layers]
    arousal_corr = [layer_stats[str(layer)]["arousal_corr"] for layer in layers]
    plane_fraction = [100 * layer_stats[str(layer)]["compliance_overlap"]["plane_norm_fraction"] for layer in layers]
    cosine_valence = [100 * abs(layer_stats[str(layer)]["compliance_overlap"]["cosine_valence"]) for layer in layers]
    cosine_arousal = [100 * abs(layer_stats[str(layer)]["compliance_overlap"]["cosine_arousal"]) for layer in layers]

    directions = ["shared_within_label", "authority_given_wrong", "endorsement_presence", "content"]
    axis_cos = [
        100 * overlap_summary["directions"][direction]["cos_with_assistant_axis"]
        for direction in directions
    ]
    va_frac = [
        100 * overlap_summary["directions"][direction]["subspace_norm_fraction"]
        for direction in directions
    ]

    direction_labels = ["Shared", "Auth-wrong", "Endorsement", "Content"]

    fig = plt.figure(figsize=(8.3, 2.65))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.15], wspace=0.42)

    # Panel A: recovery by layer
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(layers, valence_corr, marker="o", color=COLORS["valence"], linewidth=1.2, markersize=3.2, label="Valence")
    ax1.plot(layers, arousal_corr, marker="o", color=COLORS["arousal"], linewidth=1.2, markersize=3.2, label="Arousal")
    ax1.set_title("A. Lexical VA recovery", loc="left", pad=4)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Correlation (r)")
    ax1.set_xticks(layers)
    ax1.set_ylim(0.14, 0.31)
    _style_axis(ax1)
    ax1.legend(loc="lower left", frameon=False)

    # Panel B: overlap by layer
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(layers))
    width = 0.22
    ax2.bar(x - width, plane_fraction, width=width, color=COLORS["plane"], alpha=0.9, label="VA plane")
    ax2.bar(x, cosine_valence, width=width, color=COLORS["valence"], alpha=0.9, label="|cos V|")
    ax2.bar(x + width, cosine_arousal, width=width, color=COLORS["arousal"], alpha=0.9, label="|cos A|")
    ax2.set_title("B. Compliance overlap", loc="left", pad=4)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Magnitude (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.set_ylim(0, 7)
    _style_axis(ax2)
    ax2.legend(loc="upper right", frameon=False)

    # Panel C: direction-wise comparison
    ax3 = fig.add_subplot(gs[0, 2])
    markers = ["o", "s", "^", "D"]
    for x_val, y_val, label, marker in zip(axis_cos, va_frac, direction_labels, markers):
        ax3.scatter(x_val, y_val, s=72, color=COLORS["plane"], marker=marker,
                    edgecolor="white", linewidth=0.5, zorder=3, label=label)
    ax3.set_title("C. Direction deconfounds (L18)", loc="left", pad=4)
    ax3.set_xlabel("Assistant-axis cosine (%)")
    ax3.set_ylabel("VA-plane fraction (%)")
    ax3.set_xlim(-3, 9)
    ax3.set_ylim(3, 10.2)
    for spine in ax3.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("black")
    ax3.grid(color=COLORS["grid"], linewidth=0.55, alpha=0.45)
    ax3.axvline(0.0, color=COLORS["axis"], linewidth=0.8, zorder=1)
    ax3.tick_params(width=0.7, length=2.7, pad=1.5)
    ax3.legend(loc="lower right", frameon=False, fontsize=6.6, handletextpad=0.4)

    for ext in ("png", "pdf", "svg"):
        out_path = OUTPUT_DIR / f"va_recovery_overlap.{ext}"
        fig.savefig(out_path)
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
