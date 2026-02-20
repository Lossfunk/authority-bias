#!/usr/bin/env python3
"""Figure 6: Evidence-quality scaling in Exp14.

Design goals:
1. Match the paper's warm editorial style and smooth trajectories.
2. Show per-model evidence ladders on the prior-wrong slice.
3. Include a prompt-evidence panel with a concrete template example.

Run:
    uv run python -m src.paper_figures.fig_evidence_quality
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.theme import (
    PAL,
    apply_theme,
    label_panel,
    add_y_grid,
    add_zero_line,
    save_fig,
)


MODEL_ORDER = [
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B\nInstruct"),
    ("Qwen/Qwen3-4B-Instruct-2507", "Qwen 3 4B\nInstruct"),
    ("Qwen/Qwen3-4B-Thinking-2507", "Qwen 3 4B\nThinking"),
]

TAG_ORDER = [("expert", "Expert"), ("note", "Note")]

EVIDENCE_LABELS = {
    "bare": "E0",
    "reason1": "E1",
    "reason2": "E2",
    "reason_data": "E3",
}

LINE_STYLES = {
    "correct": {"color": PAL.blue, "label": "Correct endorsement shift"},
    "wrong": {"color": PAL.red, "label": "Wrong endorsement shift"},
}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _smooth_curve(x: np.ndarray, y: np.ndarray, n: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """PCHIP interpolation keeps smooth curves without overshoot."""
    try:
        from scipy.interpolate import PchipInterpolator
    except Exception:
        return x, y

    if x.size < 2:
        return x, y
    xp = np.linspace(float(x.min()), float(x.max()), n)
    yp = PchipInterpolator(x, y)(xp)
    return xp, yp


def _extract_delta_series(
    model_payload: dict,
    *,
    slice_name: str,
    tag: str,
    instruction_key: str,
    direction: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], int]:
    payload = model_payload["monotonicity"][slice_name][tag][instruction_key][direction]
    level_rows = payload["mean_by_level"]
    levels = model_payload["evidence_levels"]

    means = np.array([float(row["mean"]) for row in level_rows], dtype=float)
    lo = np.array([float(row["ci95_low"]) for row in level_rows], dtype=float)
    hi = np.array([float(row["ci95_high"]) for row in level_rows], dtype=float)

    # Delta from bare evidence level (E0) to compare growth rates.
    means_d = means - means[0]
    lo_d = lo - hi[0]
    hi_d = hi - lo[0]
    return means_d, lo_d, hi_d, levels, int(payload["n_items"])


def _add_vertical_semantic_gradient(ax: plt.Axes) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    grad = np.linspace(0, 1, 512).reshape(512, 1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "exp14_semantic_vertical",
        [PAL.negative_region, PAL.bg_warm, PAL.positive_region],
    )
    ax.imshow(
        grad,
        extent=[x0, x1, y0, y1],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        alpha=0.55,
        interpolation="bicubic",
        zorder=0,
    )


def make_figure(
    models: Sequence[dict],
    *,
    output_dir: Path,
    formats: Sequence[str],
    slice_name: str,
    instruction_key: str,
) -> None:
    fig = plt.figure(figsize=(7.2, 5.7), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16, wspace=0.34, hspace=0.35)
    gs = fig.add_gridspec(2, 3)

    axes_grid: List[List[plt.Axes]] = []
    for row_idx in range(2):
        row_axes: List[plt.Axes] = []
        for col_idx in range(3):
            row_axes.append(fig.add_subplot(gs[row_idx, col_idx]))
        axes_grid.append(row_axes)
    row_limits: Dict[str, Tuple[float, float]] = {}
    for tag_key, _tag_label in TAG_ORDER:
        mins: List[float] = []
        maxs: List[float] = []
        for model_payload in models:
            for direction in ("correct", "wrong"):
                _mean, lo, hi, _levels, _n = _extract_delta_series(
                    model_payload,
                    slice_name=slice_name,
                    tag=tag_key,
                    instruction_key=instruction_key,
                    direction=direction,
                )
                mins.append(float(np.min(lo)))
                maxs.append(float(np.max(hi)))
        y0 = min(mins + [0.0])
        y1 = max(maxs + [0.0])
        pad = 0.12 * (y1 - y0 + 1e-6)
        row_limits[tag_key] = (y0 - pad, y1 + pad)

    x = np.arange(4, dtype=float)
    for row_idx, (tag_key, tag_label) in enumerate(TAG_ORDER):
        ylim = row_limits[tag_key]
        for col_idx, model_payload in enumerate(models):
            ax = axes_grid[row_idx][col_idx]
            model_id = model_payload["model"]
            n_items = 0
            levels = model_payload["evidence_levels"]

            ax.set_xlim(-0.15, 3.15)
            ax.set_ylim(*ylim)
            _add_vertical_semantic_gradient(ax)
            add_zero_line(ax, alpha=0.75, linewidth=0.75)
            add_y_grid(ax, alpha=0.18)

            for direction in ("correct", "wrong"):
                means, lo, hi, levels, n_items = _extract_delta_series(
                    model_payload,
                    slice_name=slice_name,
                    tag=tag_key,
                    instruction_key=instruction_key,
                    direction=direction,
                )
                style = LINE_STYLES[direction]
                xs, ys = _smooth_curve(x, means, n=200)

                ax.fill_between(x, lo, hi, color=style["color"], alpha=0.10, linewidth=0, zorder=1)
                ax.plot(xs, ys, color=style["color"], linewidth=2.0, alpha=0.96, zorder=3)
                ax.scatter(
                    x,
                    means,
                    s=30,
                    color=style["color"],
                    edgecolors=PAL.bg_warm,
                    linewidths=0.8,
                    zorder=4,
                )
                ax.annotate(
                    f"{means[-1]:+.2f}",
                    xy=(x[-1], means[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=6.7,
                    color=style["color"],
                    fontweight="bold",
                )

            if row_idx == 0:
                ax.text(
                    0.98,
                    0.97,
                    f"n={n_items}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.5,
                    color=PAL.medium_gray,
                )

            if col_idx == 0:
                ax.set_ylabel(f"{tag_label}\nDelta logit shift from E0", fontsize=8)
            else:
                ax.set_ylabel("")

            if row_idx == 1:
                tick_labels = [EVIDENCE_LABELS.get(level, level) for level in levels]
                ax.set_xticks(x)
                ax.set_xticklabels(tick_labels, fontsize=7.2)
                ax.set_xlabel("Evidence level", fontsize=8)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([""] * len(x))

    label_panel(axes_grid[0][0], "A", x=-0.12, y=1.10, fontsize=12)

    # Column headers (kept below the main title to avoid overlap).
    for col_idx, model_payload in enumerate(models):
        model_id = model_payload["model"]
        short = next((label for key, label in MODEL_ORDER if key == model_id), model_id)
        bbox = axes_grid[0][col_idx].get_position()
        x_mid = 0.5 * (bbox.x0 + bbox.x1)
        fig.text(
            x_mid,
            0.875,
            short,
            ha="center",
            va="bottom",
            fontsize=8.8,
            color=PAL.dark_text,
            fontweight="bold",
            linespacing=1.0,
        )

    legend_handles = [
        mlines.Line2D(
            [0], [0],
            color=LINE_STYLES["correct"]["color"],
            marker="o",
            markersize=5,
            markeredgecolor=PAL.bg_warm,
            markeredgewidth=0.6,
            linewidth=2,
            label=LINE_STYLES["correct"]["label"],
        ),
        mlines.Line2D(
            [0], [0],
            color=LINE_STYLES["wrong"]["color"],
            marker="o",
            markersize=5,
            markeredgecolor=PAL.bg_warm,
            markeredgewidth=0.6,
            linewidth=2,
            label=LINE_STYLES["wrong"]["label"],
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=2,
        fontsize=7.3,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    fig.suptitle(
        "Exp14 evidence-quality scaling: stronger support drives larger corrective updates",
        fontsize=11,
        fontweight="bold",
        y=0.975,
        color=PAL.dark_text,
    )

    for path in save_fig(fig, "fig6_evidence_quality_ladder", output_dir, formats):
        print(f"  {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-path",
        type=Path,
        default=Path("new-phase-results/llama-3.1-8b-results/exp14/shared/analysis/combined_analysis.json"),
    )
    parser.add_argument("--slice-name", default="prior_wrong")
    parser.add_argument("--instruction-key", default="I0", choices=["I0", "I1"])
    parser.add_argument("--output-dir", type=Path, default=Path("new-phase-results/figures/paper"))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()

    apply_theme()

    combined = load_json(args.analysis_path)
    payload_models = combined.get("models", [])
    model_map = {m["model"]: m for m in payload_models}
    selected_models = [model_map[key] for key, _label in MODEL_ORDER if key in model_map]
    if not selected_models:
        raise ValueError("No expected models found in combined analysis payload.")

    make_figure(
        selected_models,
        output_dir=args.output_dir,
        formats=args.formats,
        slice_name=args.slice_name,
        instruction_key=args.instruction_key,
    )


if __name__ == "__main__":
    main()
