#!/usr/bin/env python3
"""Exp12-K: Repeated wrong-endorsement pressure figures.

Generates three publication-quality figures comparing Llama-3.1-8B-Instruct
and Qwen3-4B-Instruct under repeated wrong-endorsement pressure (K=1..20).

Run:
  uv run python -m src.exp12.plot_repeated_endorsement
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Okabe-Ito palette -- consistent with project (exp7-10, plot_phase_sycophancy)
# ---------------------------------------------------------------------------
COLORS = {
    "sky_blue": "#56B4E9",
    "orange": "#E69F00",
    "bluish_green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "yellow": "#F0E442",
    "gray": "#999999",
    "black": "#000000",
    "white": "#FFFFFF",
}

# Publication rcParams -- aligned with paper_figures/theme.py
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
        "lines.linewidth": 2.0,
    }
)

SAVE_DPI = 600  # 600 DPI for raster per guidelines

# ---------------------------------------------------------------------------
# Model styling -- sky_blue/orange matches project's two-model palette
# ---------------------------------------------------------------------------
MODEL_STYLES = {
    "llama": {
        "color": "#3B7EA1",   # steel blue – consistent with paper figures
        "marker": "o",
        "label": "Llama-3.1-8B-Instruct",
        "short": "Llama",
    },
    "qwen": {
        "color": "#C4648A",   # deep rose – consistent with paper figures
        "marker": "s",
        "label": "Qwen3-4B-Instruct",
        "short": "Qwen",
    },
}

TAGS = ["expert", "note"]
TAG_LABELS = {"expert": "Expert", "note": "Note"}
K_VALUES = [1, 2, 5, 10, 20]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_summary(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _ci95(mean: float, std: float, n: int) -> Tuple[float, float]:
    """Return (lo, hi) for a 95% CI from pre-computed summary stats."""
    if n <= 0:
        return (mean, mean)
    se = std / math.sqrt(n)
    return (mean - 1.96 * se, mean + 1.96 * se)


def extract_pressure_series(
    summary: dict, tag: str, instr_key: str = "instr_0"
) -> Tuple[List[float], List[float], List[float]]:
    """Return (means, ci_lo, ci_hi) for the pressure accumulation curve.

    Index 0 = initial.immediate_wrong_shift  (K=0 baseline)
    Index 1..5 = pressure.K<n>.pressure_wrong_shift  (K=1,2,5,10,20)
    """
    metrics = summary["metrics"][tag][instr_key]

    init = metrics["initial"]["immediate_wrong_shift"]
    means = [init["mean"]]
    lo, hi = _ci95(init["mean"], init["std"], init["n"])
    ci_lo = [lo]
    ci_hi = [hi]

    for k in K_VALUES:
        m = metrics["pressure"][f"K{k}"]["pressure_wrong_shift"]
        means.append(m["mean"])
        lo, hi = _ci95(m["mean"], m["std"], m["n"])
        ci_lo.append(lo)
        ci_hi.append(hi)

    return means, ci_lo, ci_hi


def extract_washout_series(
    summary: dict, tag: str, mode: str, instr_key: str = "instr_0"
) -> Tuple[List[float], List[float], List[float]]:
    """Return (means, ci_lo, ci_hi) for washout_score_wrong across K values."""
    probes = summary["metrics"][tag][instr_key]["probes"][mode]["same"]
    means, ci_lo, ci_hi = [], [], []
    for k in K_VALUES:
        m = probes[f"K{k}"]["washout_score_wrong"]
        means.append(m["mean"])
        lo, hi = _ci95(m["mean"], m["std"], m["n"])
        ci_lo.append(lo)
        ci_hi.append(hi)
    return means, ci_lo, ci_hi


def extract_instruction_bars(
    summary: dict, tag: str, k_vals: List[int]
) -> Tuple[Dict[str, List[float]], Dict[str, List[Tuple[float, float]]]]:
    """Return {instr_key: [means]} and {instr_key: [(lo, hi)]} for selected K values."""
    means_by_instr: Dict[str, List[float]] = {}
    ci_by_instr: Dict[str, List[Tuple[float, float]]] = {}
    for instr_key in ["instr_0", "instr_1"]:
        means, cis = [], []
        for k in k_vals:
            m = summary["metrics"][tag][instr_key]["pressure"][f"K{k}"]["wrong_pressure_margin"]
            means.append(m["mean"])
            cis.append(_ci95(m["mean"], m["std"], m["n"]))
        means_by_instr[instr_key] = means
        ci_by_instr[instr_key] = cis
    return means_by_instr, ci_by_instr


def _draw_rounded_bar(
    ax,
    *,
    x_center: float,
    height: float,
    width: float,
    color: str,
    alpha: float = 0.9,
    rounding: float = 0.12,
    hatch: Optional[str] = None,
) -> FancyBboxPatch:
    """Vertical bar using FancyBboxPatch. Supports negative heights."""
    y0 = min(0.0, height)
    h = abs(height)
    x0 = x_center - width / 2
    rounding_eff = min(rounding, width * 0.45, (h * 0.45) if h > 0 else rounding)
    patch = FancyBboxPatch(
        (x0, y0),
        width,
        h,
        boxstyle=f"round,pad=0,rounding_size={rounding_eff}",
        facecolor=color,
        edgecolor="white",
        linewidth=1.2,
        alpha=alpha,
        hatch=hatch,
    )
    ax.add_patch(patch)
    return patch


def _save_fig(fig, output_dir: Path, stem: str, formats: Sequence[str]) -> List[Path]:
    """Save figure in requested formats at publication DPI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []
    for fmt in formats:
        out = output_dir / f"{stem}.{fmt}"
        dpi = SAVE_DPI if fmt == "png" else 300
        if fmt == "png":
            # Cap so neither dimension exceeds 2000px (account for tight bbox)
            w_in, h_in = fig.get_size_inches()
            max_dim = max(w_in, h_in)
            if max_dim > 0 and max_dim * dpi > 1900:
                dpi = int(1900 / max_dim)
        fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=dpi)
        out_paths.append(out)
    plt.close(fig)
    return out_paths


# ---------------------------------------------------------------------------
# Figure 1: Pressure Accumulation Curve
# ---------------------------------------------------------------------------
def plot_pressure_accumulation(
    llama: dict,
    qwen: dict,
    output_dir: Path,
    formats: Sequence[str],
) -> List[Path]:
    x_pos = np.arange(6)
    x_labels = ["0\n(initial)", "1", "2", "5", "10", "20"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True,
                             constrained_layout=True)

    for ax_i, tag in enumerate(TAGS):
        ax = axes[ax_i]

        for model_key, summary in [("llama", llama), ("qwen", qwen)]:
            style = MODEL_STYLES[model_key]
            means, ci_lo, ci_hi = extract_pressure_series(summary, tag)
            means_arr = np.array(means)
            ci_lo_arr = np.array(ci_lo)
            ci_hi_arr = np.array(ci_hi)

            ax.fill_between(
                x_pos, ci_lo_arr, ci_hi_arr,
                color=style["color"], alpha=0.15,
            )
            ax.plot(
                x_pos, means_arr,
                color=style["color"],
                marker=style["marker"],
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=2.0,
                zorder=3,
            )

            ax.annotate(
                style["short"],
                xy=(x_pos[-1], means_arr[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color=style["color"],
                va="center",
            )

            if model_key == "llama":
                y_off = 10
            else:
                y_off = -14
            ax.annotate(
                f"{means_arr[0]:.1f}",
                xy=(x_pos[0], means_arr[0]),
                xytext=(6, y_off),
                textcoords="offset points",
                fontsize=8,
                color=style["color"],
                ha="left",
                fontweight="bold",
            )

        ax.axhline(0, color=COLORS["gray"], linewidth=1.0, linestyle="--", alpha=0.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("K (number of wrong endorsements)")
        ax.set_title(TAG_LABELS[tag], fontsize=10, loc="left", pad=6)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)

        if ax_i == 0:
            ax.set_ylabel("Pressure shift (log-odds)\n(Lower = more susceptible)")

    fig.suptitle(
        "Repeated Wrong-Endorsement Pressure\n"
        "(Margin shift: wrong branch minus neutral)",
        fontsize=11, fontweight="bold",
    )

    return _save_fig(fig, output_dir, "fig1_pressure_accumulation", formats)


# ---------------------------------------------------------------------------
# Figure 2: Persistence & Washout (context probes only)
# ---------------------------------------------------------------------------
def plot_washout(
    llama: dict,
    qwen: dict,
    output_dir: Path,
    formats: Sequence[str],
) -> List[Path]:
    x_pos = np.arange(5)
    x_labels = ["1", "2", "5", "10", "20"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True,
                             constrained_layout=True)

    for ax_i, tag in enumerate(TAGS):
        ax = axes[ax_i]

        for model_key, summary in [("llama", llama), ("qwen", qwen)]:
            style = MODEL_STYLES[model_key]
            means, ci_lo, ci_hi = extract_washout_series(summary, tag, "context")
            means_arr = np.array(means)
            ci_lo_arr = np.array(ci_lo)
            ci_hi_arr = np.array(ci_hi)

            ax.fill_between(
                x_pos, ci_lo_arr, ci_hi_arr,
                color=style["color"], alpha=0.15,
            )
            ax.plot(
                x_pos, means_arr,
                color=style["color"],
                marker=style["marker"],
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=2.0,
                zorder=3,
            )

            ax.annotate(
                f"{style['short']}  {means_arr[-1]:.2f}",
                xy=(x_pos[-1], means_arr[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color=style["color"],
                va="center",
            )

        ax.axhline(1.0, color=COLORS["gray"], linewidth=1.0, linestyle="--", alpha=0.5)
        ax.axhline(0, color=COLORS["gray"], linewidth=1.0, linestyle="--", alpha=0.5)

        if ax_i == 1:
            ax.text(
                -0.3, 1.0, "Complete washout",
                fontsize=7, color=COLORS["gray"], va="bottom",
            )
            ax.text(
                -0.3, 0.0, "Full persistence",
                fontsize=7, color=COLORS["gray"], va="top",
            )

        ax.set_ylim(-0.08, 1.12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("K (number of wrong endorsements)")
        ax.set_title(TAG_LABELS[tag], fontsize=10, loc="left", pad=6)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)

        if ax_i == 0:
            ax.set_ylabel("In-context washout score\n(0 = persists, 1 = washes out)")

    axes[0].text(
        0.03, 0.97,
        "Fresh-context probes (not shown)\nalways = 1.0 by construction",
        transform=axes[0].transAxes,
        ha="left", va="top", fontsize=7,
        color=COLORS["gray"],
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white", alpha=0.85,
            edgecolor=COLORS["gray"], linewidth=0.8,
        ),
    )

    fig.suptitle(
        "In-Context Persistence After Endorsement Pressure\n"
        "(Higher = more recovery toward correct answer)",
        fontsize=11, fontweight="bold",
    )

    return _save_fig(fig, output_dir, "fig2_washout_persistence", formats)


# ---------------------------------------------------------------------------
# Figure 3: Instruction Modulation
# ---------------------------------------------------------------------------
def plot_instruction_modulation(
    llama: dict,
    qwen: dict,
    output_dir: Path,
    formats: Sequence[str],
) -> List[Path]:
    selected_k = [1, 5, 20]
    n_groups = len(selected_k)
    n_bars = 4
    bar_w = 0.18
    group_gap = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True,
                             constrained_layout=True)

    for ax_i, tag in enumerate(TAGS):
        ax = axes[ax_i]
        group_centers = np.arange(n_groups) * (n_bars * bar_w + 0.15 + group_gap)

        bar_configs = []
        for model_key, summary in [("llama", llama), ("qwen", qwen)]:
            style = MODEL_STYLES[model_key]
            means_by_instr, ci_by_instr = extract_instruction_bars(
                summary, tag, selected_k
            )
            bar_configs.append({
                "means": means_by_instr["instr_0"],
                "cis": ci_by_instr["instr_0"],
                "color": style["color"],
                "alpha": 0.9,
                "hatch": None,
                "label": style["short"],
            })
            bar_configs.append({
                "means": means_by_instr["instr_1"],
                "cis": ci_by_instr["instr_1"],
                "color": style["color"],
                "alpha": 0.45,
                "hatch": "//",
                "label": f"{style['short']} + instr.",
            })

        for b_i, cfg in enumerate(bar_configs):
            offsets = group_centers + (b_i - (n_bars - 1) / 2) * bar_w
            for g_i in range(n_groups):
                _draw_rounded_bar(
                    ax,
                    x_center=offsets[g_i],
                    height=cfg["means"][g_i],
                    width=bar_w * 0.88,
                    color=cfg["color"],
                    alpha=cfg["alpha"],
                    rounding=0.06,
                    hatch=cfg["hatch"],
                )
                # Error bar with caps
                lo, hi = cfg["cis"][g_i]
                ax.plot(
                    [offsets[g_i], offsets[g_i]], [lo, hi],
                    color=COLORS["black"], linewidth=1.0, alpha=0.6, zorder=4,
                )
                cap_w = bar_w * 0.3
                for y_val in [lo, hi]:
                    ax.plot(
                        [offsets[g_i] - cap_w / 2, offsets[g_i] + cap_w / 2],
                        [y_val, y_val],
                        color=COLORS["black"], linewidth=1.0, alpha=0.6, zorder=4,
                    )

                # Value annotation on tallest bars only (reduces clutter)
                if abs(cfg["means"][g_i]) > 2.0:
                    y_ann = cfg["means"][g_i]
                    if y_ann > 0:
                        y_anchor = hi
                        y_off = 5
                    else:
                        y_anchor = lo
                        y_off = -12
                    ax.annotate(
                        f"{y_ann:.1f}",
                        xy=(offsets[g_i], y_anchor),
                        xytext=(0, y_off),
                        textcoords="offset points",
                        ha="center", fontsize=7, fontweight="bold",
                        color=cfg["color"],
                    )

        ax.axhline(0, color=COLORS["black"], linewidth=1.0, alpha=0.4)
        ax.set_xticks(group_centers)
        ax.set_xticklabels([f"K = {k}" for k in selected_k])
        ax.set_xlabel("Repetition count")
        ax.set_title(TAG_LABELS[tag], fontsize=10, loc="left", pad=6)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)

        if ax_i == 0:
            ax.set_ylabel("Wrong pressure margin (log-odds)")
            handles = []
            for cfg in bar_configs:
                h = FancyBboxPatch(
                    (0, 0), 1, 1,
                    boxstyle="round,pad=0,rounding_size=0.1",
                    facecolor=cfg["color"], alpha=cfg["alpha"],
                    edgecolor="white", linewidth=0.8,
                )
                if cfg["hatch"]:
                    h.set_hatch(cfg["hatch"])
                handles.append(h)
            ax.legend(
                handles, [c["label"] for c in bar_configs],
                loc="upper left", fontsize=7, ncol=2,
                handlelength=1.5, handleheight=1.2, columnspacing=1.0,
            )

    fig.suptitle(
        '"Be Correct" Instruction Effect on Endorsement Pressure\n'
        "(Positive = margin favors correct answer)",
        fontsize=11, fontweight="bold",
    )

    return _save_fig(fig, output_dir, "fig3_instruction_modulation", formats)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot exp12-K repeated wrong-endorsement pressure figures."
    )
    parser.add_argument(
        "--llama-summary", type=Path,
        default=Path(
            "new-phase-results/llama/exp12_k/"
            "meta-llama__Llama-3.1-8B-Instruct_summary.json"
        ),
    )
    parser.add_argument(
        "--qwen-summary", type=Path,
        default=Path(
            "new-phase-results/qwen/exp12_k/"
            "Qwen__Qwen3-4B-Instruct-2507_summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("new-phase-results/figures/exp12_k"),
    )
    parser.add_argument(
        "--formats", type=str, nargs="+",
        default=["png", "pdf"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llama = load_summary(args.llama_summary)
    qwen = load_summary(args.qwen_summary)

    out1 = plot_pressure_accumulation(llama, qwen, args.output_dir, args.formats)
    out2 = plot_washout(llama, qwen, args.output_dir, args.formats)
    out3 = plot_instruction_modulation(llama, qwen, args.output_dir, args.formats)

    print("Wrote:")
    for p in out1 + out2 + out3:
        print(f"  {p}")


if __name__ == "__main__":
    main()
