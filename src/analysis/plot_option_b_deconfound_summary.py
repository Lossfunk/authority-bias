#!/usr/bin/env python3
"""Option B: deconfound summary figure.

Run:
  uv run python -m src.analysis.plot_option_b_deconfound_summary
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


VA_PLANE_PATH = Path("results/authority/mechanism/lexical_va_plane/va_plane.pt")
ASSISTANT_AXIS_OVERLAP_PATH = Path("external/assistant-axis/hardened/overlap_summary.json")
COMPLIANCE_DIRECTIONS_PATH = Path("results/authority/mechanism/gpt_oss_compliance_analysis/directions.pt")
OUTPUT_DIR = Path("figures/neurips/option-b")

COLORS = {
    "assistant": "#55A868",
    "valence": "#4C72B0",
    "arousal": "#DD8452",
    "plane": "#7A68A6",
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
        "legend.fontsize": 7.3,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def _style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("black")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.45)
    ax.axvline(0.0, color=COLORS["axis"], linewidth=0.8, zorder=1)
    ax.tick_params(width=0.7, length=2.7, pad=1.5)


def _annotate_barh(ax: plt.Axes, values: list[float], ys: np.ndarray, color: str) -> None:
    for value, y in zip(values, ys):
        inside = abs(value) > 1.4
        x_pos = value / 2 if inside else value + (0.32 if value >= 0 else -0.32)
        ax.annotate(
            f"{value:+.1f}",
            xy=(x_pos, y),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center" if inside else ("left" if value >= 0 else "right"),
            va="center",
            fontsize=6.8,
            color="white" if inside else "#2b2b2b",
            bbox=None if inside else dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.9),
        )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    axis_summary = json.loads(ASSISTANT_AXIS_OVERLAP_PATH.read_text())
    va_payload = torch.load(VA_PLANE_PATH, map_location="cpu", weights_only=False)
    directions_payload = torch.load(COMPLIANCE_DIRECTIONS_PATH, map_location="cpu", weights_only=False)

    layer = int(axis_summary["layer"])
    basis = va_payload["basis_by_layer"][layer].numpy()
    valence_vec = va_payload["valence_by_layer"][layer].numpy()
    arousal_vec = va_payload["arousal_by_layer"][layer].numpy()

    directions = [
        "shared_within_label",
        "authority_given_wrong",
        "authority_given_correct",
        "endorsement_presence",
        "content",
    ]
    labels = ["Shared", "Auth-wrong", "Auth-correct", "Endorsement", "Content"]

    assistant_cos = []
    va_frac = []
    valence_cos = []
    arousal_cos = []

    for direction in directions:
        vec = directions_payload[direction].numpy()
        assistant_cos.append(100 * axis_summary["directions"][direction]["cos_with_assistant_axis"])
        coords = np.asarray([float(np.dot(vec, row)) for row in basis], dtype=np.float64)
        va_frac.append(100 * float(np.linalg.norm(coords) / np.linalg.norm(vec)))
        valence_cos.append(100 * _cosine(vec, valence_vec))
        arousal_cos.append(100 * _cosine(vec, arousal_vec))

    y = np.arange(len(labels))

    fig = plt.figure(figsize=(8.25, 3.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.2], wspace=0.42)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(y, assistant_cos, color=COLORS["assistant"], alpha=0.85)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_title("A. Assistant-axis overlap", loc="left", pad=4)
    ax1.set_xlabel("Cosine (%)")
    ax1.set_xlim(-3, 9)
    _style_axis(ax1)
    _annotate_barh(ax1, assistant_cos, y, COLORS["assistant"])

    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.26
    ax2.barh(y, va_frac, height=0.46, color=COLORS["plane"], alpha=0.32, label="VA plane")
    ax2.scatter(valence_cos, y - width / 2, color=COLORS["valence"], s=28, label="Valence", zorder=3)
    ax2.scatter(arousal_cos, y + width / 2, color=COLORS["arousal"], s=28, label="Arousal", zorder=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_title("B. Cheap VA overlap", loc="left", pad=4)
    ax2.set_xlabel("Overlap / cosine (%)")
    ax2.set_xlim(-7, 9)
    _style_axis(ax2)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, handletextpad=0.5, columnspacing=1.0)
    for value, y_pos, label in zip(va_frac, y, labels):
        ax2.annotate(
            f"{value:.1f}",
            xy=(value / 2, y_pos),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=6.8,
            color="#5b4d8a",
        )

    for ext in ("png", "pdf", "svg"):
        out_path = OUTPUT_DIR / f"deconfound_summary.{ext}"
        fig.savefig(out_path)
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
