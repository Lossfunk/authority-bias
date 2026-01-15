#!/usr/bin/env python3
"""Publication-quality figures for sycophancy paper.

Creates A*-tier alignment paper figures with:
- Single clear takeaway per figure
- Minimal visual clutter
- Direct annotations
- Okabe-Ito color palette
- Generous whitespace

Run:
  uv run python -m src.analysis.plot_publication_figures
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# Okabe-Ito colorblind-friendly palette
OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "gray": "#999999",
}

# Matplotlib style settings for publication quality
plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
})


def read_jsonl(path: Path) -> List[dict]:
    """Read JSONL file."""
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_phase1_simple(
    phase1_path: Path,
    output_dir: Path,
    formats: List[str] = ["png", "pdf"]
) -> List[Path]:
    """
    Phase 1: Simplified dumbbell plot showing neutral → user_asserts_wrong shift.

    Single takeaway: "Models shift toward user's wrong belief"
    """
    records = read_jsonl(phase1_path)

    # Extract data
    rows = []
    for r in records:
        fc = r["forced_choice"]
        rows.append({
            "dataset": r.get("dataset", ""),
            "question": r.get("question", ""),
            "neutral": float(fc["neutral"]),
            "assert_wrong": float(fc["user_asserts_wrong"]),
            "delta": float(r["sycophancy_score"]),
        })

    # Sort by delta (largest effect first)
    rows.sort(key=lambda x: x["delta"], reverse=True)

    # Create labels (shortened)
    labels = []
    for r in rows:
        q = " ".join(r["question"].split())
        if len(q) > 55:
            q = q[:52] + "..."
        labels.append(q)

    neutral = np.array([r["neutral"] for r in rows])
    assert_wrong = np.array([r["assert_wrong"] for r in rows])
    delta = np.array([r["delta"] for r in rows])

    n = len(rows)
    y_positions = np.arange(n)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw connecting lines (gray, subtle)
    for i in range(n):
        ax.plot(
            [neutral[i], assert_wrong[i]],
            [y_positions[i], y_positions[i]],
            color=OKABE_ITO["gray"],
            alpha=0.6,
            linewidth=2,
            zorder=1
        )

    # Plot neutral points (gray)
    ax.scatter(
        neutral,
        y_positions,
        s=80,
        color=OKABE_ITO["gray"],
        alpha=0.8,
        zorder=2,
        label="Neutral baseline"
    )

    # Plot user_asserts_wrong points (orange)
    ax.scatter(
        assert_wrong,
        y_positions,
        s=100,
        color=OKABE_ITO["orange"],
        alpha=0.9,
        zorder=3,
        label="User asserts wrong"
    )

    # Styling
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("P(wrong | forced choice)", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.invert_yaxis()

    # Minimal grid
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)

    # Title with clear takeaway
    ax.set_title(
        "Models shift toward user's stated belief",
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    # Legend (upper right to avoid overlap)
    ax.legend(loc="upper right", fontsize=10)

    # Add mean shift annotation (lower left for better spacing)
    mean_shift = float(delta.mean())
    ax.text(
        0.02,
        0.02,
        f"Mean shift: +{mean_shift:.3f}\n({100 * np.mean(delta > 0):.0f}% positive)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=OKABE_ITO["orange"],
            alpha=0.15,
            edgecolor=OKABE_ITO["orange"],
            linewidth=1.5
        )
    )

    plt.tight_layout()

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for fmt in formats:
        out_path = output_dir / f"phase1_publication.{fmt}"
        fig.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=300)
        out_paths.append(out_path)

    plt.close(fig)
    return out_paths


def plot_phase2_simple(
    base_path: Path,
    instruct_path: Path,
    output_dir: Path,
    formats: List[str] = ["png", "pdf"]
) -> List[Path]:
    """
    Phase 2: Two-panel figure comparing base and instruct models.

    Panel A: Overlaid histograms showing both models shift toward user belief
    Panel B: Scatter plot showing correlation between base and instruct sycophancy
    """
    base_records = read_jsonl(base_path)
    instruct_records = read_jsonl(instruct_path)

    # Match records by UID
    base_by_uid = {r["uid"]: r for r in base_records}
    inst_by_uid = {r["uid"]: r for r in instruct_records}
    common_uids = sorted(set(base_by_uid.keys()) & set(inst_by_uid.keys()))

    # Extract sycophancy scores
    base_syc = np.array([float(base_by_uid[uid]["sycophancy_score"]) for uid in common_uids])
    inst_syc = np.array([float(inst_by_uid[uid]["sycophancy_score"]) for uid in common_uids])

    # Extract neutral baseline confidence for instruct
    base_neutral = np.array([float(base_by_uid[uid]["forced_choice"]["neutral"]) for uid in common_uids])
    inst_neutral = np.array([float(inst_by_uid[uid]["forced_choice"]["neutral"]) for uid in common_uids])

    # Create two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ============ Panel A: Histogram of sycophancy scores ============
    bins = np.linspace(-0.8, 1.0, 50)

    ax1.hist(
        base_syc,
        bins=bins,
        density=True,
        color=OKABE_ITO["blue"],
        alpha=0.4,
        edgecolor="white",
        linewidth=0.5,
        label="Base model"
    )

    ax1.hist(
        inst_syc,
        bins=bins,
        density=True,
        color=OKABE_ITO["vermillion"],
        alpha=0.4,
        edgecolor="white",
        linewidth=0.5,
        label="Instruct model"
    )

    # Add vertical lines at means
    base_mean = float(base_syc.mean())
    inst_mean = float(inst_syc.mean())

    ax1.axvline(base_mean, color=OKABE_ITO["blue"], linewidth=2.5, linestyle="--", alpha=0.8)
    ax1.axvline(inst_mean, color=OKABE_ITO["vermillion"], linewidth=2.5, linestyle="--", alpha=0.8)
    ax1.axvline(0, color="black", linewidth=1.2, alpha=0.4)

    ax1.set_xlabel("Sycophancy score", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Both models shift toward user belief (~90% positive)",
        fontsize=13,
        fontweight="bold",
        pad=12
    )

    ax1.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax1.legend(loc="upper left", fontsize=10)

    # Add statistics annotation
    pos_base = 100 * np.mean(base_syc > 0)
    pos_inst = 100 * np.mean(inst_syc > 0)

    ax1.text(
        0.98,
        0.98,
        f"Base: μ={base_mean:.3f} ({pos_base:.0f}% positive)\n"
        f"Instruct: μ={inst_mean:.3f} ({pos_inst:.0f}% positive)",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            alpha=0.9,
            edgecolor=OKABE_ITO["gray"],
            linewidth=1.2
        )
    )

    # ============ Panel B: Scatter plot ============
    ax2.scatter(
        base_syc,
        inst_syc,
        s=12,
        color=OKABE_ITO["gray"],
        alpha=0.4
    )

    # Add diagonal reference line
    lims = [
        np.min([ax2.get_xlim(), ax2.get_ylim()]),
        np.max([ax2.get_xlim(), ax2.get_ylim()]),
    ]
    ax2.plot(lims, lims, 'k--', alpha=0.4, linewidth=1.2, zorder=1)

    # Add mean point
    ax2.scatter(
        [base_mean],
        [inst_mean],
        s=200,
        color=OKABE_ITO["vermillion"],
        marker="X",
        edgecolor="white",
        linewidth=1.5,
        zorder=5,
        label="Mean"
    )

    ax2.set_xlabel("Base sycophancy score", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Instruct sycophancy score", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Instruct shows similar pattern to base",
        fontsize=13,
        fontweight="bold",
        pad=12
    )

    ax2.grid(alpha=0.25, linewidth=0.8)
    ax2.legend(loc="upper left", fontsize=10)

    # Compute correlation
    corr = np.corrcoef(base_syc, inst_syc)[0, 1]

    ax2.text(
        0.98,
        0.02,
        f"Correlation: r={corr:.3f}\nn={len(common_uids)}",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            alpha=0.9,
            edgecolor=OKABE_ITO["gray"],
            linewidth=1.2
        )
    )

    plt.tight_layout()

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for fmt in formats:
        out_path = output_dir / f"phase2_publication.{fmt}"
        fig.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=300)
        out_paths.append(out_path)

    plt.close(fig)
    return out_paths


def main():
    """Generate all publication figures."""
    # Paths
    phase1_path = Path("new-phase-results/phase1_sycophancy.jsonl")
    phase2_dir = Path("new-phase-results/phase2")
    output_dir = Path("new-phase-results/figures")

    # Find phase2 files
    base_path = phase2_dir / "meta-llama__Llama-3.1-8B.jsonl"
    instruct_path = phase2_dir / "meta-llama__Llama-3.1-8B-Instruct.jsonl"

    # Generate figures
    print("Generating publication-quality figures...")

    print("\n1. Phase 1 (simplified dumbbell)...")
    out1 = plot_phase1_simple(phase1_path, output_dir)
    for p in out1:
        print(f"   ✓ {p}")

    print("\n2. Phase 2 (histogram + scatter)...")
    out2 = plot_phase2_simple(base_path, instruct_path, output_dir)
    for p in out2:
        print(f"   ✓ {p}")

    print("\n✓ All figures generated successfully!")


if __name__ == "__main__":
    main()
