#!/usr/bin/env python3
"""Refined Assistant-Axis-style FactualQA flip-rate figure.

Run:
  uv run python -m src.analysis.plot_factualqa_forward_patch_flip_refined
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


ROWS_PATH = Path("results/authority/mechanism/forward_patch_test_256/steering_rows.jsonl")
OUTPUT_DIR = Path("figures/neurips")

COLORS = {
    "flip": "#C44E52",
    "flip_fill": "#EBC2C4",
    "axis": "#BDBDBD",
    "grid": "#E7E7E7",
    "text": "#222222",
    "muted": "#777777",
}


plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.4,
        "axes.labelsize": 8.3,
        "axes.titlesize": 8.6,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
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
            layer_points.append((alpha, 100.0 * flips / len(baseline_correct)))
        out[layer] = layer_points
    return out


def _style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("black")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.45)
    ax.axvline(0.0, color=COLORS["axis"], linestyle="--", linewidth=0.8, zorder=1)
    ax.tick_params(width=0.7, length=2.7, pad=1.5)


def _annotate_peak(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    idx = int(np.argmax(y))
    x_peak = float(x[idx])
    y_peak = float(y[idx])
    x_max = float(np.max(x))
    offset_x = -16 if x_peak > 0.85 * x_max else 6
    offset_y = -2 if y_peak > 0.88 * ax.get_ylim()[1] else 7
    ax.annotate(
        f"{y_peak:.1f}%",
        xy=(x_peak, y_peak),
        xytext=(offset_x, offset_y),
        textcoords="offset points",
        color=COLORS["flip"],
        fontsize=7.1,
        ha="right" if offset_x < 0 else "left",
        va="top" if offset_y < 0 else "bottom",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.8),
        annotation_clip=False,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rates = _compute_flip_rates(_load_rows())
    layers = [16, 18, 20]

    fig, axes = plt.subplots(1, 3, figsize=(5.95, 1.9), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.30, top=0.84, wspace=0.14)

    for ax, layer in zip(axes, layers):
        points = rates[layer]
        x = np.array([alpha for alpha, _ in points], dtype=float)
        y = np.array([rate for _, rate in points], dtype=float)

        x_dense = np.linspace(float(x.min()), float(x.max()), 300)
        y_dense = PchipInterpolator(x, y)(x_dense)

        ax.fill_between(x_dense, 0, y_dense, color=COLORS["flip_fill"], alpha=0.18, zorder=1)
        ax.plot(x_dense, y_dense, color=COLORS["flip"], linewidth=1.15, alpha=0.96, zorder=2)
        ax.plot(
            x,
            y,
            color=COLORS["flip"],
            marker="o",
            markersize=3.0,
            linewidth=0.0,
            markerfacecolor=COLORS["flip"],
            markeredgecolor="white",
            markeredgewidth=0.3,
            zorder=3,
        )
        _annotate_peak(ax, x, y)

        ax.set_title(f"L{layer}", pad=4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylim(0, 36)
        ax.set_yticks([0, 9, 18, 27, 36])
        _style_axis(ax)

    axes[0].set_ylabel("Right-to-wrong flips (%)")
    fig.supxlabel("Patch alpha", y=0.10, fontsize=8.3)

    for ext in ("png", "pdf", "svg"):
        out_path = OUTPUT_DIR / f"factualqa_forward_patch_flip_refined.{ext}"
        fig.savefig(out_path)
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
