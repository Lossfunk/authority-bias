"""Plotting utilities for Exp14 evidence-quality analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


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


# Model dirs under results-root that contain exp14/analysis
_EXP14_MODEL_DIRS = (
    "llama-3.1-8b-results",
    "qwen3-4b-results",
    "qwen3-4b-thinking-results",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Exp14 evidence-quality figures")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Directory containing *_analysis.json files (if not using --results-root)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("new-phase-results"),
        help=(
            "Root of results tree; discovers *_analysis.json under "
            "{root}/{model_dir}/evidence_quality/analysis for all models",
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("new-phase-results/figures/evidence_quality"),
        help="Directory for output figures",
    )
    parser.add_argument("--formats", type=str, default="png,pdf")
    parser.add_argument(
        "--slices",
        type=str,
        default="all,prior_correct,prior_wrong",
        help="Comma-separated slices to include",
    )
    return parser.parse_args()


def _parse_csv(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _load_analysis_files(analysis_dir: Path) -> List[Dict[str, Any]]:
    files = sorted(analysis_dir.glob("*_analysis.json"))
    analyses: List[Dict[str, Any]] = []
    for path in files:
        if path.name == "combined_analysis.json":
            continue
        payload = json.loads(path.read_text())
        analyses.append(payload)
    if not analyses:
        raise FileNotFoundError(f"No analysis files found in {analysis_dir}")
    return analyses


def _discover_analyses_from_root(results_root: Path) -> List[Dict[str, Any]]:
    """Discover all *_analysis.json under {root}/{model_dir}/evidence_quality/analysis."""
    analyses: List[Dict[str, Any]] = []
    seen_models: set[str] = set()
    for model_dir in _EXP14_MODEL_DIRS:
        analysis_dir = results_root / model_dir / "evidence_quality" / "analysis"
        if not analysis_dir.is_dir():
            continue
        for path in sorted(analysis_dir.glob("*_analysis.json")):
            if path.name == "combined_analysis.json":
                continue
            payload = json.loads(path.read_text())
            model_id = payload.get("model", "")
            if model_id not in seen_models:
                seen_models.add(model_id)
                analyses.append(payload)
    if not analyses:
        raise FileNotFoundError(
            f"No *_analysis.json found under {results_root}/{{model_dir}}/evidence_quality/analysis "
            f"for model dirs: {list(_EXP14_MODEL_DIRS)}"
        )
    return analyses


def _series_from_level_payload(payload: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.array([float(row["mean"]) for row in payload["mean_by_level"]], dtype=float)
    lo = np.array([float(row["ci95_low"]) for row in payload["mean_by_level"]], dtype=float)
    hi = np.array([float(row["ci95_high"]) for row in payload["mean_by_level"]], dtype=float)
    return means, lo, hi


def _plot_monotonicity(
    analysis: Dict[str, Any],
    *,
    slices: Sequence[str],
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    model = analysis["model"]
    model_tag = model.replace("/", "__")
    tags = analysis["tags"]
    levels = analysis["evidence_levels"]
    instruction_keys = analysis["instruction_keys"]

    selected_slices = [s for s in slices if s in analysis["monotonicity"]]
    if not selected_slices:
        return

    fig, axes = plt.subplots(
        nrows=len(selected_slices),
        ncols=len(tags),
        figsize=(3.5 * len(tags), 2.7 * len(selected_slices)),
        squeeze=False,
        constrained_layout=True,
    )

    x = np.arange(len(levels), dtype=float)
    for r, slice_name in enumerate(selected_slices):
        for c, tag in enumerate(tags):
            ax = axes[r][c]
            for instr_key in instruction_keys:
                payload = analysis["monotonicity"][slice_name][tag][instr_key]

                wrong_m, wrong_lo, wrong_hi = _series_from_level_payload(payload["wrong"])
                correct_m, correct_lo, correct_hi = _series_from_level_payload(payload["correct"])

                style = "-" if instr_key.endswith("0") else "--"
                ax.plot(
                    x,
                    wrong_m,
                    linestyle=style,
                    color="#D55E00",
                    marker="o",
                    linewidth=1.4,
                    label=f"wrong ({instr_key})",
                )
                ax.fill_between(x, wrong_lo, wrong_hi, color="#D55E00", alpha=0.12)

                ax.plot(
                    x,
                    correct_m,
                    linestyle=style,
                    color="#0072B2",
                    marker="o",
                    linewidth=1.4,
                    label=f"correct ({instr_key})",
                )
                ax.fill_between(x, correct_lo, correct_hi, color="#0072B2", alpha=0.12)

            ax.set_xticks(x)
            ax.set_xticklabels(levels, rotation=25, ha="right")
            ax.axhline(0.0, color="#999999", linewidth=0.8)
            if c == 0:
                ax.set_ylabel(slice_name)
            if r == 0:
                ax.set_title(tag)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle(f"Exp14 Monotonicity: {model}", fontsize=12, fontweight="bold")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{model_tag}_monotonicity"
    for fmt in formats:
        fig.savefig(f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def _plot_asymmetry(
    analysis: Dict[str, Any],
    *,
    slices: Sequence[str],
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    model = analysis["model"]
    model_tag = model.replace("/", "__")
    tags = analysis["tags"]
    levels = analysis["evidence_levels"]
    instruction_keys = analysis["instruction_keys"]

    selected_slices = [s for s in slices if s in analysis["asymmetry"]]
    if not selected_slices:
        return

    fig, axes = plt.subplots(
        nrows=len(selected_slices),
        ncols=len(tags),
        figsize=(3.5 * len(tags), 2.7 * len(selected_slices)),
        squeeze=False,
        constrained_layout=True,
    )

    x = np.arange(len(levels), dtype=float)
    for r, slice_name in enumerate(selected_slices):
        for c, tag in enumerate(tags):
            ax = axes[r][c]
            for instr_key in instruction_keys:
                payload = analysis["asymmetry"][slice_name][tag][instr_key]
                means, lo, hi = _series_from_level_payload(payload)
                style = "-" if instr_key.endswith("0") else "--"
                color = "#009E73" if instr_key.endswith("0") else "#CC79A7"
                ax.plot(
                    x,
                    means,
                    linestyle=style,
                    color=color,
                    marker="o",
                    linewidth=1.5,
                    label=f"{instr_key}",
                )
                ax.fill_between(x, lo, hi, color=color, alpha=0.15)

            ax.set_xticks(x)
            ax.set_xticklabels(levels, rotation=25, ha="right")
            ax.axhline(0.0, color="#999999", linewidth=0.8)
            if c == 0:
                ax.set_ylabel(slice_name)
            if r == 0:
                ax.set_title(tag)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle(f"Exp14 Asymmetry (logit): {model}", fontsize=12, fontweight="bold")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{model_tag}_asymmetry"
    for fmt in formats:
        fig.savefig(f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def _plot_key_diagnostic(
    analysis: Dict[str, Any],
    *,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    model = analysis["model"]
    model_tag = model.replace("/", "__")
    tags = analysis["tags"]
    instruction_keys = analysis["instruction_keys"]

    labels: List[str] = []
    means: List[float] = []
    lows: List[float] = []
    highs: List[float] = []

    diag = analysis["key_diagnostic_prior_wrong_e3_minus_e0_correct_shift"]
    for tag in tags:
        for instr_key in instruction_keys:
            payload = diag[tag][instr_key]
            labels.append(f"{tag}\n{instr_key}")
            means.append(float(payload["mean_delta"]))
            lows.append(float(payload["ci95"][0]))
            highs.append(float(payload["ci95"][1]))

    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(1.6 + 1.0 * len(labels), 3.6), constrained_layout=True)
    ax.bar(x, means, color="#0072B2", alpha=0.85)
    ax.errorbar(x, means, yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
                fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_ylabel("E_last - E_first correct shift")
    ax.set_title(f"Exp14 Key Diagnostic (Prior-Wrong): {model}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{model_tag}_key_diagnostic"
    for fmt in formats:
        fig.savefig(f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    formats = [token.lower() for token in _parse_csv(args.formats)]
    slices = _parse_csv(args.slices)

    if args.analysis_dir is not None:
        analyses = _load_analysis_files(args.analysis_dir)
    else:
        analyses = _discover_analyses_from_root(args.results_root)
    for analysis in analyses:
        _plot_monotonicity(
            analysis,
            slices=slices,
            output_dir=args.output_dir,
            formats=formats,
        )
        _plot_asymmetry(
            analysis,
            slices=slices,
            output_dir=args.output_dir,
            formats=formats,
        )
        _plot_key_diagnostic(
            analysis,
            output_dir=args.output_dir,
            formats=formats,
        )

    print(f"Wrote Exp14 figures to {args.output_dir}")


if __name__ == "__main__":
    main()
