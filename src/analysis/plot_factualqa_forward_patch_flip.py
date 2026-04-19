#!/usr/bin/env python3
"""Plot FactualQA forward-patch flip rates among baseline-correct items.

Run:
  uv run python -m src.analysis.plot_factualqa_forward_patch_flip
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


ROWS_PATH = Path("neurips-results/mechanism/forward_patch_test_256/steering_rows.jsonl")
OUTPUT_DIR = Path("figures/neurips")

COLORS = {
    "flip": "#C44E52",
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
    return [json.loads(line) for line in ROWS_PATH.read_text().splitlines() if line.strip()]


def _compute_flip_rates(rows: list[dict]) -> dict[int, list[tuple[float, float]]]:
    grouped: dict[int, dict[float, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["condition_code"] != "N0_note":
            continue
        layer = int(row["target_layers"][0])
        alpha = float(row["alpha"])
        grouped[layer][alpha][row["uid"]] = row

    out: dict[int, list[tuple[float, float]]] = {}
    for layer, by_alpha in grouped.items():
        base_rows = by_alpha[0.0]
        baseline_correct = {uid for uid, row in base_rows.items() if row.get("is_correct") is True}
        layer_points: list[tuple[float, float]] = []
        for alpha in sorted(by_alpha):
            rows_at_alpha = by_alpha[alpha]
            flips = sum(1 for uid in baseline_correct if rows_at_alpha[uid].get("chose_wrong") is True)
            flip_rate = 100.0 * flips / len(baseline_correct)
            layer_points.append((alpha, flip_rate))
        out[layer] = layer_points
    return out


def _style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("black")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.45)
    ax.axvline(0.0, color=COLORS["axis"], linestyle="--", linewidth=0.8, zorder=1)
    ax.tick_params(width=0.7, length=2.7, pad=1.5)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rates = _compute_flip_rates(_load_rows())
    layers = [16, 18, 20]

    fig, axes = plt.subplots(1, 3, figsize=(6.35, 1.95), sharey=True)
    fig.subplots_adjust(left=0.095, right=0.86, bottom=0.29, top=0.84, wspace=0.12)

    legend_handle = None
    for ax, layer in zip(axes, layers):
        points = rates[layer]
        x = np.array([alpha for alpha, _ in points], dtype=float)
        y = np.array([rate for _, rate in points], dtype=float)

        x_dense = np.linspace(float(x.min()), float(x.max()), 300)
        y_dense = PchipInterpolator(x, y)(x_dense)

        ax.plot(x_dense, y_dense, color=COLORS["flip"], linewidth=1.1, alpha=0.92, zorder=2)
        legend_handle = ax.plot(
            x,
            y,
            color=COLORS["flip"],
            marker="o",
            markersize=2.9,
            linewidth=0.0,
            markerfacecolor=COLORS["flip"],
            markeredgecolor="white",
            markeredgewidth=0.3,
            zorder=3,
            label="Flip rate",
        )[0]

        ax.set_title(f"L{layer}", pad=4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylim(0, 36)
        ax.set_yticks([0, 9, 18, 27, 36])
        _style_axis(ax)

    axes[0].set_ylabel("Right-to-wrong flips (%)")
    fig.supxlabel("Patch alpha", y=0.10, fontsize=8.5)

    fig.legend(
        [legend_handle],
        ["Flip rate"],
        loc="center left",
        bbox_to_anchor=(0.865, 0.56),
        frameon=True,
        edgecolor="#DDDDDD",
        facecolor="white",
        framealpha=0.95,
        borderpad=0.35,
        handlelength=1.1,
    )

    for ext in ("png", "pdf", "svg"):
        out_path = OUTPUT_DIR / f"factualqa_forward_patch_flip.{ext}"
        fig.savefig(out_path)
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
