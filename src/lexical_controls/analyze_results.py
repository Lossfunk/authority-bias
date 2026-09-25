"""Analysis and visualization for Experiment 7 results.

Generates:
- Log-odds delta distributions (histograms)
- Model comparison plots
- Effect by dataset source (if available)
- Statistical summaries with confidence intervals
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Use Okabe-Ito colorblind-friendly palette
COLORS = {
    "instruct": "#0072B2",  # Blue
    "base": "#E69F00",      # Orange
    "default": "#009E73",   # Green
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp7 results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/lexical_controls"),
        help="Directory containing Exp7 results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/lexical_controls/figures"),
        help="Directory to save figures.",
    )
    return parser.parse_args()


def load_results(results_dir: Path) -> Tuple[Dict, List[Dict]]:
    """Load summary and per-example results."""
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found at {summary_path}")

    with summary_path.open() as f:
        summary = json.load(f)

    # Load per-example results for each model
    model_results = {}
    for model_summary in summary.get("models", []):
        model_id = model_summary["model"]
        model_tag = model_id.replace("/", "__")
        results_path = results_dir / f"{model_tag}_results.jsonl"
        if results_path.exists():
            records = []
            with results_path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            model_results[model_id] = records

    return summary, model_results


def _get_model_color(model_id: str) -> str:
    """Get color for model based on type."""
    if "instruct" in model_id.lower():
        return COLORS["instruct"]
    elif any(x in model_id.lower() for x in ["base", "-8b", "-70b"]):
        # Check if it's a base model (no instruct suffix)
        if "instruct" not in model_id.lower():
            return COLORS["base"]
    return COLORS["default"]


def _get_model_label(model_id: str) -> str:
    """Get short label for model."""
    parts = model_id.split("/")
    name = parts[-1] if parts else model_id
    return name


def _bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0

    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    return float(lower), float(upper)


def plot_delta_logit_distribution(
    summary: Dict,
    model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Plot distribution of delta_logit_wrong for each model."""
    n_models = len(model_results)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)
    axes = axes[0]

    for ax, (model_id, records) in zip(axes, model_results.items()):
        values = [r["effect"]["delta_logit_wrong"] for r in records]
        color = _get_model_color(model_id)
        label = _get_model_label(model_id)

        ax.hist(values, bins=40, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="No effect")
        ax.axvline(x=np.mean(values), color="black", linestyle="-", linewidth=1.5,
                   label=f"Mean: {np.mean(values):.3f}")

        ax.set_xlabel("Delta Log-Odds (Wrong)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{label}\n(n={len(values)})", fontsize=11)
        ax.legend(fontsize=8)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "delta_logit_distribution.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "delta_logit_distribution.pdf", bbox_inches="tight")
    plt.close()


def plot_model_comparison(
    summary: Dict,
    model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Plot model comparison with confidence intervals."""
    if not model_results:
        return

    models = list(model_results.keys())
    means = []
    ci_lower = []
    ci_upper = []
    colors = []

    for model_id in models:
        values = [r["effect"]["delta_logit_wrong"] for r in model_results[model_id]]
        m = np.mean(values)
        lo, hi = _bootstrap_ci(values)
        means.append(m)
        ci_lower.append(m - lo)
        ci_upper.append(hi - m)
        colors.append(_get_model_color(model_id))

    labels = [_get_model_label(m) for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=[ci_lower, ci_upper], capsize=5,
                  color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="No effect")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean Delta Log-Odds (Wrong)", fontsize=11)
    ax.set_title("Endorsement Effect: Base vs Instruct\n(Lexical-Fixed Design)", fontsize=12)
    ax.legend(fontsize=9)

    # Add value labels on bars
    for i, (bar, m) in enumerate(zip(bars, means)):
        ax.annotate(f"{m:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, m),
                    xytext=(0, 5 if m >= 0 else -15),
                    textcoords="offset points",
                    ha="center", va="bottom" if m >= 0 else "top",
                    fontsize=9, fontweight="bold")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "model_comparison.pdf", bbox_inches="tight")
    plt.close()


def plot_effect_by_label(
    model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Plot effect broken down by which label was correct (A vs B)."""
    if not model_results:
        return

    fig, axes = plt.subplots(1, len(model_results), figsize=(5 * len(model_results), 4), squeeze=False)
    axes = axes[0]

    for ax, (model_id, records) in zip(axes, model_results.items()):
        # Split by correct label
        values_a = [r["effect"]["delta_logit_wrong"] for r in records if r["correct_label"] == "A"]
        values_b = [r["effect"]["delta_logit_wrong"] for r in records if r["correct_label"] == "B"]

        x = [0, 1]
        means = [np.mean(values_a) if values_a else 0, np.mean(values_b) if values_b else 0]

        ci_a = _bootstrap_ci(values_a) if values_a else (0, 0)
        ci_b = _bootstrap_ci(values_b) if values_b else (0, 0)
        errors = [
            [means[0] - ci_a[0], means[1] - ci_b[0]],
            [ci_a[1] - means[0], ci_b[1] - means[1]],
        ]

        bars = ax.bar(x, means, yerr=errors, capsize=5,
                      color=[COLORS["instruct"], COLORS["base"]], alpha=0.8,
                      edgecolor="black", linewidth=0.5)

        ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Correct=A\n(n={len(values_a)})", f"Correct=B\n(n={len(values_b)})"])
        ax.set_ylabel("Mean Delta Log-Odds", fontsize=10)
        ax.set_title(f"{_get_model_label(model_id)}", fontsize=11)

    plt.suptitle("Effect by Label Assignment (Bias Check)", fontsize=12)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "effect_by_label.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "effect_by_label.pdf", bbox_inches="tight")
    plt.close()


def plot_effect_by_dataset(
    model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Plot effect broken down by dataset source."""
    if not model_results:
        return

    # Check if metadata has dataset info
    first_model = next(iter(model_results.values()))
    if not first_model or "metadata" not in first_model[0]:
        return

    # Collect all datasets
    all_datasets = set()
    for records in model_results.values():
        for r in records:
            ds = r.get("metadata", {}).get("dataset", "unknown")
            all_datasets.add(ds)

    if len(all_datasets) < 2:
        return  # Not enough datasets to compare

    datasets = sorted(all_datasets)
    n_models = len(model_results)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(datasets))
    width = 0.8 / n_models

    for i, (model_id, records) in enumerate(model_results.items()):
        means = []
        errors_low = []
        errors_high = []

        for ds in datasets:
            values = [r["effect"]["delta_logit_wrong"] for r in records
                      if r.get("metadata", {}).get("dataset") == ds]
            if values:
                m = np.mean(values)
                lo, hi = _bootstrap_ci(values)
                means.append(m)
                errors_low.append(m - lo)
                errors_high.append(hi - m)
            else:
                means.append(0)
                errors_low.append(0)
                errors_high.append(0)

        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=[errors_low, errors_high],
               label=_get_model_label(model_id), color=_get_model_color(model_id),
               alpha=0.8, capsize=3)

    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel("Mean Delta Log-Odds (Wrong)", fontsize=11)
    ax.set_title("Endorsement Effect by Dataset Source", fontsize=12)
    ax.legend()

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "effect_by_dataset.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "effect_by_dataset.pdf", bbox_inches="tight")
    plt.close()


def generate_text_report(
    summary: Dict,
    model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Generate text report with detailed statistics."""
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT 7: CLEAN LEXICAL-FIXED SYCOPHANCY MEASUREMENT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("DESIGN:")
    lines.append("  - Condition A (Baseline): Options + Answer prompt")
    lines.append("  - Condition B (Endorsement): Options + User endorsement + Answer prompt")
    lines.append("  - Lexical exposure identical across conditions")
    lines.append("  - Endorsement refers only to label (A/B), not answer text")
    lines.append("")
    lines.append("PRIMARY METRIC: Delta Log-Odds of Wrong Choice")
    lines.append("  delta = logit(P_wrong | endorsement) - logit(P_wrong | baseline)")
    lines.append("  Positive = model shifts toward endorsed (wrong) answer")
    lines.append("")
    lines.append("-" * 70)
    lines.append("RESULTS BY MODEL")
    lines.append("-" * 70)

    for model_id, records in model_results.items():
        values = [r["effect"]["delta_logit_wrong"] for r in records]
        ci_lo, ci_hi = _bootstrap_ci(values)

        lines.append("")
        lines.append(f"Model: {model_id}")
        lines.append(f"  N examples: {len(values)}")
        lines.append(f"  Delta Log-Odds (Wrong):")
        lines.append(f"    Mean:   {np.mean(values):.4f}")
        lines.append(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        lines.append(f"    Median: {np.median(values):.4f}")
        lines.append(f"    Std:    {np.std(values):.4f}")
        lines.append(f"    Min:    {np.min(values):.4f}")
        lines.append(f"    Max:    {np.max(values):.4f}")
        lines.append(f"  Positive fraction: {sum(v > 0 for v in values) / len(values):.1%}")

        # Effect by label
        values_a = [r["effect"]["delta_logit_wrong"] for r in records if r["correct_label"] == "A"]
        values_b = [r["effect"]["delta_logit_wrong"] for r in records if r["correct_label"] == "B"]
        lines.append(f"  By label assignment:")
        lines.append(f"    Correct=A: mean={np.mean(values_a):.4f} (n={len(values_a)})")
        lines.append(f"    Correct=B: mean={np.mean(values_b):.4f} (n={len(values_b)})")

    lines.append("")
    lines.append("-" * 70)
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("")

    # Simple interpretation
    all_values = []
    for records in model_results.values():
        all_values.extend([r["effect"]["delta_logit_wrong"] for r in records])

    if not all_values:
        lines.append("NO RESULTS: No examples were scored.")
    elif all_values:
        overall_mean = np.mean(all_values)
        if overall_mean > 0.1:
            lines.append("FINDING: Substantial positive endorsement effect detected.")
            lines.append("The effect persists under lexical-fixed conditions,")
            lines.append("suggesting deference/compliance beyond mere priming.")
        elif overall_mean > 0:
            lines.append("FINDING: Small positive endorsement effect detected.")
            lines.append("Some deference may exist, but effect is modest.")
        else:
            lines.append("FINDING: No clear endorsement effect.")
            lines.append("Prior results may have been lexical artifacts.")

    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "report.txt").open("w") as f:
        f.write(report_text)

    print(report_text)


def main() -> None:
    args = parse_args()

    print(f"Loading results from {args.results_dir}")
    summary, model_results = load_results(args.results_dir)

    print(f"Generating figures in {args.output_dir}")
    plot_delta_logit_distribution(summary, model_results, args.output_dir)
    plot_model_comparison(summary, model_results, args.output_dir)
    plot_effect_by_label(model_results, args.output_dir)
    plot_effect_by_dataset(model_results, args.output_dir)

    print("Generating text report")
    generate_text_report(summary, model_results, args.output_dir)

    print(f"\nAnalysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
