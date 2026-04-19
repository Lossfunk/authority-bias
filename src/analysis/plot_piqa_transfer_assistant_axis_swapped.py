#!/usr/bin/env python3
"""Plot PIQA transfer results with swapped axes.

Run:
  uv run python -m src.analysis.plot_piqa_transfer_assistant_axis_swapped
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


RESULTS_PATH = Path("neurips-results/mechanism/piqa_forward_patch_200_freegen/piqa_summary.json")
OUTPUT_DIR = Path("figures/neurips")

COLORS = {
    "wrong": "#C44E52",
    "correct": "#4C72B0",
    "axis": "#BDBDBD",
    "grid": "#E7E7E7",
}


plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def _load_rows() -> list[dict]:
    rows = json.loads(RESULTS_PATH.read_text())
    return [row for row in rows if row["condition_code"] == "N0_note"]


def _group_by_layer(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["target_layers"][0]), []).append(row)
    for layer_rows in grouped.values():
        layer_rows.sort(key=lambda row: float(row["alpha"]))
    return grouped


def _style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("black")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.45)
    ax.axvline(0.0, color=COLORS["axis"], linewidth=0.8, zorder=1)
    ax.axhline(0.0, color=COLORS["axis"], linestyle="--", linewidth=0.8, zorder=1)
    ax.tick_params(width=0.7, length=2.7, pad=1.5)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = _load_rows()
    grouped = _group_by_layer(rows)
    layers = [16, 18, 20]

    fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.1), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.80, bottom=0.24, top=0.84, wspace=0.14)

    legend_handles = None
    for ax, layer in zip(axes, layers):
        layer_rows = grouped[layer]
        alpha = np.array([float(row["alpha"]) for row in layer_rows], dtype=float)
        wrong_delta = np.array([100.0 * float(row["wrong_rate_parsed"]) for row in layer_rows], dtype=float)
        correct_delta = np.array([100.0 * float(row["accuracy_parsed"]) for row in layer_rows], dtype=float)

        wrong_delta = wrong_delta - wrong_delta[0]
        correct_delta = correct_delta - correct_delta[0]

        alpha_dense = np.linspace(float(alpha.min()), float(alpha.max()), 300)
        wrong_dense = PchipInterpolator(alpha, wrong_delta)(alpha_dense)
        correct_dense = PchipInterpolator(alpha, correct_delta)(alpha_dense)

        ax.plot(wrong_dense, alpha_dense, color=COLORS["wrong"], linewidth=1.05, alpha=0.92, zorder=2)
        ax.plot(correct_dense, alpha_dense, color=COLORS["correct"], linewidth=1.05, alpha=0.92, zorder=2)

        wrong_pts = ax.plot(
            wrong_delta, alpha, color=COLORS["wrong"], marker="o", markersize=2.9, linewidth=0.0,
            markerfacecolor=COLORS["wrong"], markeredgecolor="white", markeredgewidth=0.3,
            zorder=3, label="Δ wrong answer"
        )[0]
        correct_pts = ax.plot(
            correct_delta, alpha, color=COLORS["correct"], marker="o", markersize=2.9, linewidth=0.0,
            markerfacecolor=COLORS["correct"], markeredgecolor="white", markeredgewidth=0.3,
            zorder=3, label="Δ correct answer"
        )[0]
        if legend_handles is None:
            legend_handles = [wrong_pts, correct_pts]

        ax.set_title(f"L{layer}", pad=4)
        ax.set_xlim(-12, 12)
        ax.set_xticks([-12, -6, 0, 6, 12])
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        _style_axis(ax)

    axes[0].set_ylabel("Patch alpha")
    fig.supxlabel("Δ responses (pp)", y=0.08, fontsize=8.5)

    fig.legend(
        legend_handles,
        ["Δ wrong answer", "Δ correct answer"],
        loc="center left",
        bbox_to_anchor=(0.81, 0.56),
        frameon=True,
        edgecolor="#DDDDDD",
        facecolor="white",
        framealpha=0.95,
        borderpad=0.35,
        handlelength=1.1,
    )

    for ext in ("png", "pdf", "svg"):
        out_path = OUTPUT_DIR / f"piqa_transfer_assistant_axis_swapped.{ext}"
        fig.savefig(out_path)
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
