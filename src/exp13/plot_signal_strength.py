"""Plotting utilities for Exp13 analysis artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CERTAINTY_ORDER = ["might", "think", "sure"]
DEFAULT_SALIENCE_ORDER = ["plain", "important"]

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#999999",
}

plt.rcParams.update(
    {
        "figure.dpi": 180,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Exp13 signal-strength figures")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("new-phase-results/exp13/analysis"),
        help="Directory containing *_analysis.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("new-phase-results/figures/exp13"),
        help="Directory for output figures",
    )
    parser.add_argument("--formats", type=str, default="png,pdf")
    return parser.parse_args()


def _load_analysis_files(analysis_dir: Path) -> List[Dict[str, Any]]:
    files = sorted(analysis_dir.glob("*_analysis.json"))
    analyses: List[Dict[str, Any]] = []
    for path in files:
        if path.name == "combined_analysis.json":
            continue
        with path.open("r") as f:
            payload = json.load(f)
        analyses.append(payload)
    if not analyses:
        raise FileNotFoundError(f"No model analysis files found in {analysis_dir}")
    return analyses


def _parse_variant(variant: str) -> Tuple[str, str]:
    parts = variant.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected variant key format: {variant}")
    return parts[0], parts[1]


def _get_orders(analysis: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    certainty = []
    salience = []
    for variant in analysis["variants"]:
        c, s = _parse_variant(variant)
        if c not in certainty:
            certainty.append(c)
        if s not in salience:
            salience.append(s)
    certainty_sorted = [c for c in DEFAULT_CERTAINTY_ORDER if c in certainty] + [
        c for c in certainty if c not in DEFAULT_CERTAINTY_ORDER
    ]
    salience_sorted = [s for s in DEFAULT_SALIENCE_ORDER if s in salience] + [
        s for s in salience if s not in DEFAULT_SALIENCE_ORDER
    ]
    return certainty_sorted, salience_sorted


def _matrix_metric(
    analysis: Dict[str, Any],
    *,
    tag: str,
    metric: str,
    certainty_order: Sequence[str],
    salience_order: Sequence[str],
) -> np.ndarray:
    matrix = np.full((len(salience_order), len(certainty_order)), np.nan, dtype=float)
    for variant in analysis["variants"]:
        c, s = _parse_variant(variant)
        if c not in certainty_order or s not in salience_order:
            continue
        i = salience_order.index(s)
        j = certainty_order.index(c)
        matrix[i, j] = float(analysis["metric_summaries"][tag][variant][metric]["mean"])
    return matrix


def _matrix_wrong_prior_dr(
    analysis: Dict[str, Any],
    *,
    tag: str,
    certainty_order: Sequence[str],
    salience_order: Sequence[str],
) -> np.ndarray:
    matrix = np.full((len(salience_order), len(certainty_order)), np.nan, dtype=float)
    for variant in analysis["variants"]:
        c, s = _parse_variant(variant)
        if c not in certainty_order or s not in salience_order:
            continue
        i = salience_order.index(s)
        j = certainty_order.index(c)

        slices = analysis["wrong_prior_slices"].get(tag, {}).get(variant, {})
        top = slices.get("high_conf_top25", {})
        metrics = top.get("metrics", {})
        dr = metrics.get("dr_median")
        if isinstance(dr, list) and dr:
            matrix[i, j] = float(dr[0])
    return matrix


def _plot_heatmap_grid(
    *,
    analyses: Sequence[Dict[str, Any]],
    matrix_fn,
    title: str,
    output_stem: str,
    cmap: str,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    tags = analyses[0]["tags"]
    certainty_order, salience_order = _get_orders(analyses[0])

    all_vals: List[float] = []
    matrices: Dict[Tuple[int, int], np.ndarray] = {}
    for r, analysis in enumerate(analyses):
        for c, tag in enumerate(tags):
            matrix = matrix_fn(
                analysis,
                tag=tag,
                certainty_order=certainty_order,
                salience_order=salience_order,
            )
            matrices[(r, c)] = matrix
            finite = matrix[np.isfinite(matrix)]
            if finite.size > 0:
                all_vals.extend(finite.tolist())

    if not all_vals:
        raise RuntimeError(f"No finite values found while plotting {output_stem}")
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))

    fig, axes = plt.subplots(
        nrows=len(analyses),
        ncols=len(tags),
        figsize=(3.2 * len(tags), 2.6 * len(analyses)),
        squeeze=False,
        constrained_layout=True,
    )

    im = None
    for r, analysis in enumerate(analyses):
        model_name = analysis["model"].split("/")[-1]
        for c, tag in enumerate(tags):
            ax = axes[r][c]
            matrix = matrices[(r, c)]
            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(np.arange(len(certainty_order)))
            ax.set_xticklabels(certainty_order)
            ax.set_yticks(np.arange(len(salience_order)))
            ax.set_yticklabels(salience_order)
            if c == 0:
                ax.set_ylabel(model_name)
            ax.set_title(tag if r == 0 else "")

            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    if np.isnan(value):
                        text = "NA"
                        color = OKABE_ITO["gray"]
                    else:
                        text = f"{value:.2f}"
                        color = "black"
                    ax.text(j, i, text, ha="center", va="center", color=color, fontsize=7)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.85)
        cbar.ax.set_ylabel("Mean effect")
    fig.suptitle(title, fontsize=12, fontweight="bold")

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{output_stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    formats = [token.strip().lower() for token in args.formats.split(",") if token.strip()]
    analyses = _load_analysis_files(args.analysis_dir)

    _plot_heatmap_grid(
        analyses=analyses,
        matrix_fn=lambda analysis, tag, certainty_order, salience_order: _matrix_metric(
            analysis,
            tag=tag,
            metric="effect_wrong_I0",
            certainty_order=certainty_order,
            salience_order=salience_order,
        ),
        title="Exp13 Figure 1: Wrong-Endorsement Effect (I0)",
        output_stem="fig1_wrong_endorsement_effect_heatmap",
        cmap="YlOrRd",
        output_dir=args.output_dir,
        formats=formats,
    )

    _plot_heatmap_grid(
        analyses=analyses,
        matrix_fn=lambda analysis, tag, certainty_order, salience_order: _matrix_metric(
            analysis,
            tag=tag,
            metric="selectivity",
            certainty_order=certainty_order,
            salience_order=salience_order,
        ),
        title="Exp13 Figure 2: Selectivity (Truth-Tracking Proxy)",
        output_stem="fig2_selectivity_heatmap",
        cmap="RdBu_r",
        output_dir=args.output_dir,
        formats=formats,
    )

    _plot_heatmap_grid(
        analyses=analyses,
        matrix_fn=lambda analysis, tag, certainty_order, salience_order: _matrix_wrong_prior_dr(
            analysis,
            tag=tag,
            certainty_order=certainty_order,
            salience_order=salience_order,
        ),
        title="Exp13 Figure 3: High-Confidence-Wrong dr_median",
        output_stem="fig3_high_conf_wrong_dr_heatmap",
        cmap="RdBu_r",
        output_dir=args.output_dir,
        formats=formats,
    )

    print(f"Wrote Exp13 figures to {args.output_dir}")


if __name__ == "__main__":
    main()

