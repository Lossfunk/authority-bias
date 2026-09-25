#!/usr/bin/env python3
"""Option C: combined summary figure with FactualQA, PIQA, and deconfound panel.

Run:
  uv run python -m src.analysis.plot_option_c_combined_summary
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator


FWD_ROWS_PATH = Path("results/authority/mechanism/forward_patch_test_256/steering_rows.jsonl")
PIQA_SUMMARY_PATH = Path("results/authority/mechanism/piqa_forward_patch_200_freegen/piqa_summary.json")
VA_PLANE_PATH = Path("results/authority/mechanism/lexical_va_plane/va_plane.pt")
ASSISTANT_AXIS_OVERLAP_PATH = Path("external/assistant-axis/hardened/overlap_summary.json")
COMPLIANCE_DIRECTIONS_PATH = Path("results/authority/mechanism/gpt_oss_compliance_analysis/directions.pt")
OUTPUT_DIR = Path("figures/neurips/option-c")

COLORS = {
    "flip": "#C44E52",
    "flip_fill": "#EBC2C4",
    "wrong": "#C44E52",
    "correct": "#4C72B0",
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
        "font.size": 7.5,
        "axes.labelsize": 8.4,
        "axes.titlesize": 8.8,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
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
    ax.axvline(0.0, color=COLORS["axis"], linestyle="--", linewidth=0.8, zorder=1)
    ax.tick_params(width=0.7, length=2.7, pad=1.5)


def _load_factualqa_flip() -> dict[int, list[tuple[float, float]]]:
    rows = [json.loads(line) for line in FWD_ROWS_PATH.read_text().splitlines() if line.strip()]
    grouped: dict[int, dict[float, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["condition_code"] != "N0_note":
            continue
        grouped[int(row["target_layers"][0])][float(row["alpha"])][row["uid"]] = row

    out: dict[int, list[tuple[float, float]]] = {}
    for layer, by_alpha in grouped.items():
        base_rows = by_alpha[0.0]
        baseline_correct = {uid for uid, row in base_rows.items() if row.get("is_correct") is True}
        out[layer] = []
        for alpha in sorted(by_alpha):
            flips = sum(1 for uid in baseline_correct if by_alpha[alpha][uid].get("chose_wrong") is True)
            out[layer].append((alpha, 100.0 * flips / len(baseline_correct)))
    return out


def _load_piqa_delta() -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rows = json.loads(PIQA_SUMMARY_PATH.read_text())
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row["condition_code"] == "N0_note":
            grouped[int(row["target_layers"][0])].append(row)
    out = {}
    for layer, layer_rows in grouped.items():
        layer_rows.sort(key=lambda row: float(row["alpha"]))
        x = np.array([float(row["alpha"]) for row in layer_rows], dtype=float)
        wrong = np.array([100.0 * float(row["wrong_rate_parsed"]) for row in layer_rows], dtype=float)
        correct = np.array([100.0 * float(row["accuracy_parsed"]) for row in layer_rows], dtype=float)
        out[layer] = (x, wrong - wrong[0], correct - correct[0])
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _load_deconfound_panel() -> tuple[list[str], list[float], list[float]]:
    axis_summary = json.loads(ASSISTANT_AXIS_OVERLAP_PATH.read_text())
    va_payload = torch.load(VA_PLANE_PATH, map_location="cpu", weights_only=False)
    directions_payload = torch.load(COMPLIANCE_DIRECTIONS_PATH, map_location="cpu", weights_only=False)

    layer = int(axis_summary["layer"])
    basis = va_payload["basis_by_layer"][layer].numpy()
    directions = ["shared_within_label", "authority_given_wrong", "endorsement_presence", "content"]
    labels = ["Shared", "Auth-wrong", "Endorsement", "Content"]

    axis_cos = []
    va_frac = []
    for direction in directions:
        vec = directions_payload[direction].numpy()
        coords = np.asarray([float(np.dot(vec, row)) for row in basis], dtype=np.float64)
        axis_cos.append(100 * axis_summary["directions"][direction]["cos_with_assistant_axis"])
        va_frac.append(100 * float(np.linalg.norm(coords) / np.linalg.norm(vec)))
    return labels, axis_cos, va_frac


def _plot_flip_panel(ax: plt.Axes, layer: int, points: list[tuple[float, float]]) -> None:
    x = np.array([alpha for alpha, _ in points], dtype=float)
    y = np.array([rate for _, rate in points], dtype=float)
    x_dense = np.linspace(float(x.min()), float(x.max()), 300)
    y_dense = PchipInterpolator(x, y)(x_dense)
    ax.fill_between(x_dense, 0, y_dense, color=COLORS["flip_fill"], alpha=0.18, zorder=1)
    ax.plot(x_dense, y_dense, color=COLORS["flip"], linewidth=1.1, alpha=0.96, zorder=2)
    ax.plot(x, y, color=COLORS["flip"], marker="o", markersize=2.9, linewidth=0.0,
            markerfacecolor=COLORS["flip"], markeredgecolor="white", markeredgewidth=0.3, zorder=3)
    ax.set_title(f"L{layer}", pad=4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0, 36)
    ax.set_yticks([0, 9, 18, 27, 36])
    _style_axis(ax)


def _plot_piqa_panel(ax: plt.Axes, layer: int, x: np.ndarray, wrong: np.ndarray, correct: np.ndarray) -> None:
    x_dense = np.linspace(float(x.min()), float(x.max()), 300)
    wrong_dense = PchipInterpolator(x, wrong)(x_dense)
    correct_dense = PchipInterpolator(x, correct)(x_dense)
    ax.plot(x_dense, wrong_dense, color=COLORS["wrong"], linewidth=1.05, alpha=0.92, zorder=2)
    ax.plot(x_dense, correct_dense, color=COLORS["correct"], linewidth=1.05, alpha=0.92, zorder=2)
    ax.plot(x, wrong, color=COLORS["wrong"], marker="o", markersize=2.8, linewidth=0.0,
            markerfacecolor=COLORS["wrong"], markeredgecolor="white", markeredgewidth=0.3, zorder=3)
    ax.plot(x, correct, color=COLORS["correct"], marker="o", markersize=2.8, linewidth=0.0,
            markerfacecolor=COLORS["correct"], markeredgecolor="white", markeredgewidth=0.3, zorder=3)
    ax.set_title(f"L{layer}", pad=4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(-12, 12)
    ax.set_yticks([-12, -6, 0, 6, 12])
    _style_axis(ax, zero_y=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    factual = _load_factualqa_flip()
    piqa = _load_piqa_delta()
    labels, axis_cos, va_frac = _load_deconfound_panel()

    fig = plt.figure(figsize=(8.4, 5.9))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.95], hspace=0.78, wspace=0.28)

    # Row 1
    top_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, layer in zip(top_axes, [16, 18, 20]):
        _plot_flip_panel(ax, layer, factual[layer])
    top_axes[0].set_ylabel("Right-to-wrong flips (%)")
    for ax in top_axes:
        ax.set_xlabel("")
    top_axes[1].set_xlabel("Patch alpha", labelpad=4)
    fig.text(0.08, 0.965, "A. FactualQA forward patch", fontsize=9.2, weight="bold")

    # Row 2
    mid_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for ax, layer in zip(mid_axes, [16, 18, 20]):
        x, wrong, correct = piqa[layer]
        _plot_piqa_panel(ax, layer, x, wrong, correct)
    mid_axes[0].set_ylabel("Δ responses (pp)")
    for ax in mid_axes:
        ax.set_xlabel("")
    mid_axes[1].set_xlabel("Patch alpha", labelpad=4)
    mid_axes[0].legend(
        handles=[
            Line2D([0], [0], color=COLORS["wrong"], marker="o", linewidth=1.0, markersize=3, label="Wrong answer"),
            Line2D([0], [0], color=COLORS["correct"], marker="o", linewidth=1.0, markersize=3, label="Correct answer"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=6.8,
    )
    fig.text(0.08, 0.635, "B. PIQA transfer", fontsize=9.2, weight="bold")

    # Row 3 summary
    ax3 = fig.add_subplot(gs[2, :])
    x = np.arange(len(labels))
    width = 0.35
    ax3.bar(x - width / 2, axis_cos, width=width, color=COLORS["assistant"], alpha=0.85, label="Assistant-axis cosine")
    ax3.bar(x + width / 2, va_frac, width=width, color=COLORS["plane"], alpha=0.85, label="VA-plane fraction")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("Overlap (%)")
    ax3.set_title("C. Deconfounds (L18)", loc="left", pad=4)
    ax3.set_ylim(-2, 10.5)
    _style_axis(ax3, zero_y=True)
    ax3.legend(loc="upper left", ncol=2, frameon=False)
    ax3.set_xlabel("")

    for ext in ("png", "pdf", "svg"):
        out_path = OUTPUT_DIR / f"combined_summary.{ext}"
        fig.savefig(out_path)
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
