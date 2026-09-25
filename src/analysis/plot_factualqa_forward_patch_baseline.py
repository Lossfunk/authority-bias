#!/usr/bin/env python3
"""Plot FactualQA forward-patch baseline shift curves.

Run:
  uv run python -m src.analysis.plot_factualqa_forward_patch_baseline
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


SUMMARY_PATH = Path("results/authority/mechanism/forward_patch_test_256/steering_summary.json")
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
    rows = json.loads(SUMMARY_PATH.read_text())
    return [row for row in rows if row["condition"] == "N0_note"]


def _extract_layer(config_id: str) -> int:
    for part in config_id.split("_"):
        if part.startswith("L"):
            return int(part[1:])
    raise ValueError(f"Could not extract layer from {config_id}")


def _group_by_layer(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_extract_layer(row["config_id"]), []).append(row)
    for layer_rows in grouped.values():
        layer_rows.sort(key=lambda row: float(row["alpha"]))
    return grouped


def _style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("black")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.45)
    ax.axvline(0.0, color=COLORS["axis"], linestyle="--", linewidth=0.8, zorder=1)
    ax.tick_params(width=0.7, length=2.7, pad=1.5)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = _load_rows()
    grouped = _group_by_layer(rows)
    layers = [16, 18, 20]

    fig, axes = plt.subplots(1, 3, figsize=(6.7, 1.95), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.80, bottom=0.29, top=0.84, wspace=0.12)

    legend_handles = None
    for ax, layer in zip(axes, layers):
        layer_rows = grouped[layer]
        x = np.array([float(row["alpha"]) for row in layer_rows], dtype=float)
        wrong_y = np.array([100.0 * float(row["wrong_rate"]) for row in layer_rows], dtype=float)
        correct_y = np.array([100.0 * float(row["accuracy"]) for row in layer_rows], dtype=float)

        x_dense = np.linspace(float(x.min()), float(x.max()), 300)
        wrong_dense = PchipInterpolator(x, wrong_y)(x_dense)
        correct_dense = PchipInterpolator(x, correct_y)(x_dense)

        ax.plot(x_dense, wrong_dense, color=COLORS["wrong"], linewidth=1.05, alpha=0.92, zorder=2)
        ax.plot(x_dense, correct_dense, color=COLORS["correct"], linewidth=1.05, alpha=0.92, zorder=2)

        wrong_pts = ax.plot(
            x,
            wrong_y,
            color=COLORS["wrong"],
            marker="o",
            markersize=2.9,
            linewidth=0.0,
            markerfacecolor=COLORS["wrong"],
            markeredgecolor="white",
            markeredgewidth=0.3,
            zorder=3,
            label="Wrong answer",
        )[0]
        correct_pts = ax.plot(
            x,
            correct_y,
            color=COLORS["correct"],
            marker="o",
            markersize=2.9,
            linewidth=0.0,
            markerfacecolor=COLORS["correct"],
            markeredgecolor="white",
            markeredgewidth=0.3,
            zorder=3,
            label="Correct answer",
        )[0]
        if legend_handles is None:
            legend_handles = [wrong_pts, correct_pts]

        ax.set_title(f"L{layer}", pad=4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylim(15, 85)
        ax.set_yticks([20, 35, 50, 65, 80])
        _style_axis(ax)

    axes[0].set_ylabel("Responses (%)")
    fig.supxlabel("Patch alpha", y=0.10, fontsize=8.5)

    fig.legend(
        legend_handles,
        ["Wrong answer", "Correct answer"],
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
        out_path = OUTPUT_DIR / f"factualqa_forward_patch_baseline.{ext}"
        fig.savefig(out_path)
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
