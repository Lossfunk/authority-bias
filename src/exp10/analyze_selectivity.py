"""Analysis and visualization for Experiment 10 (Correct-Endorsement Test).

Tests whether instruction suppression is truth-tracking (selective) vs
uniform endorsement-ignore gating.

Generates:
- Selectivity bar chart (critical metric)
- 2x3 condition grid (Wrong/Neutral/Correct x Instruction)
- Efficacy comparison (wrong vs correct)
- Before/after instruction comparison
- Text report with interpretation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.exp10.conditions import INSTRUCTION_TEXT, DEFAULT_TAGS, normalize_tag


# Okabe-Ito colorblind-friendly palette
COLORS = {
    "no_instr": "#E69F00",      # Orange
    "with_instr": "#0072B2",    # Blue
    "Expert": "#D55E00",        # Red-orange
    "Note": "#009E73",          # Green
    "instruct": "#0072B2",
    "base": "#E69F00",
    "wrong": "#CC79A7",         # Pink
    "correct": "#56B4E9",       # Light blue
    "neutral": "#999999",       # Gray
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp10 Correct-Endorsement results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/exp10"),
        help="Directory containing exp10 results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/exp10/figures"),
        help="Directory to save figures.",
    )
    return parser.parse_args()


def load_results(results_dir: Path) -> Tuple[Dict, Dict[str, List[Dict]]]:
    """Load summary and per-example results."""
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found at {summary_path}")

    with summary_path.open() as f:
        summary = json.load(f)

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


def _get_model_label(model_id: str) -> str:
    parts = model_id.split("/")
    return parts[-1] if parts else model_id


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


def _summary_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean/median/10-90 percentiles/positive fraction."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "positive_frac": 0.0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "positive_frac": float(np.mean(arr > 0)),
    }


def plot_selectivity_bars(
    model_results: Dict[str, List[Dict]],
    tags: List[str],
    output_dir: Path,
) -> None:
    """Plot selectivity metric (critical) for each model and tag."""
    if not model_results:
        return

    models = list(model_results.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(tags))
    width = 0.35
    offsets = [-width/2, width/2] if len(models) == 2 else [0]

    for i, model_id in enumerate(models):
        records = model_results[model_id]
        means = []
        errors_low = []
        errors_high = []

        for tag in tags:
            tag_key = normalize_tag(tag)
            values = [r["selectivity_metrics"].get(f"selectivity_{tag_key}", 0) for r in records]
            m = np.mean(values)
            lo, hi = _bootstrap_ci(values)
            means.append(m)
            errors_low.append(m - lo)
            errors_high.append(hi - m)

        color = COLORS["instruct"] if "instruct" in model_id.lower() else COLORS["base"]
        offset = offsets[i] if i < len(offsets) else 0
        ax.bar(x + offset, means, width, yerr=[errors_low, errors_high],
               label=_get_model_label(model_id), color=color, alpha=0.8, capsize=4)

        # Add value annotations
        for j, m in enumerate(means):
            va = "bottom" if m >= 0 else "top"
            offset_y = 5 if m >= 0 else -12
            ax.annotate(f"{m:.3f}", xy=(x[j] + offset, m),
                       xytext=(0, offset_y), textcoords="offset points",
                       ha="center", va=va, fontsize=9, fontweight="bold")

    ax.axhline(y=0, color="red", linestyle="--", linewidth=2, label="No selectivity (uniform gating)")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=11)
    ax.set_ylabel("Selectivity (efficacy_wrong - efficacy_correct)", fontsize=11)
    ax.set_title("Truth-Tracking vs Gating Test\n(Positive = truth-tracking, Zero = uniform gating)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)

    # Add interpretation zones
    ax.axhspan(0.05, ax.get_ylim()[1], alpha=0.1, color="green", label="_nolegend_")
    ax.axhspan(-0.05, 0.05, alpha=0.1, color="gray", label="_nolegend_")
    ax.axhspan(ax.get_ylim()[0], -0.05, alpha=0.1, color="red", label="_nolegend_")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "selectivity_bars.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "selectivity_bars.pdf", bbox_inches="tight")
    plt.close()


def plot_3x2_condition_grid(
    model_results: Dict[str, List[Dict]],
    tags: List[str],
    output_dir: Path,
) -> None:
    """Plot 3x2 condition grid: fc_correct for all 6 conditions per tag."""
    if not model_results:
        return

    models = list(model_results.keys())
    conditions = ["N0", "N1", "W0", "W1", "C0", "C1"]
    cond_labels = [
        "Neutral\n(no instr)", "Neutral\n(+ instr)",
        "Wrong\n(no instr)", "Wrong\n(+ instr)",
        "Correct\n(no instr)", "Correct\n(+ instr)"
    ]

    for tag in tags:
        tag_key = normalize_tag(tag)
        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
        axes = axes[0]

        for ax, model_id in zip(axes, models):
            records = model_results[model_id]
            means = []
            errors_low = []
            errors_high = []

            for cond in conditions:
                key = f"{cond}_{tag_key}"
                values = [r["fc_correct"].get(key, 0.5) for r in records]
                m = np.mean(values)
                lo, hi = _bootstrap_ci(values)
                means.append(m)
                errors_low.append(m - lo)
                errors_high.append(hi - m)

            x = np.arange(len(conditions))
            colors = [
                COLORS["neutral"], COLORS["neutral"],
                COLORS["wrong"], COLORS["wrong"],
                COLORS["correct"], COLORS["correct"],
            ]
            alphas = [0.6, 1.0, 0.6, 1.0, 0.6, 1.0]  # Lighter for no_instr

            bars = ax.bar(x, means, yerr=[errors_low, errors_high], capsize=4,
                         color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
            for bar, alpha in zip(bars, alphas):
                bar.set_alpha(alpha)

            ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
            ax.set_xticks(x)
            ax.set_xticklabels(cond_labels, fontsize=8)
            ax.set_ylabel("P(Correct | A/B)", fontsize=10)
            ax.set_title(f"{_get_model_label(model_id)}", fontsize=11)
            ax.set_ylim(0, 1)

            for i, m in enumerate(means):
                ax.annotate(f"{m:.3f}", xy=(i, m), xytext=(0, 5),
                           textcoords="offset points", ha="center", fontsize=7)

        plt.suptitle(f"3x2 Conditions: {tag} Tag", fontsize=12)
        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"conditions_3x2_{tag_key}.png", dpi=150, bbox_inches="tight")
        plt.savefig(output_dir / f"conditions_3x2_{tag_key}.pdf", bbox_inches="tight")
        plt.close()


def plot_efficacy_comparison(
    model_results: Dict[str, List[Dict]],
    tags: List[str],
    output_dir: Path,
) -> None:
    """Plot efficacy for wrong vs correct endorsement."""
    if not model_results:
        return

    models = list(model_results.keys())

    fig, axes = plt.subplots(1, len(tags), figsize=(5 * len(tags), 5), squeeze=False)
    axes = axes[0]

    for ax, tag in zip(axes, tags):
        tag_key = normalize_tag(tag)

        x = np.arange(len(models))
        width = 0.35

        # Wrong efficacy
        wrong_means = []
        wrong_errors = []
        for model_id in models:
            records = model_results[model_id]
            values = [r["selectivity_metrics"].get(f"efficacy_wrong_{tag_key}", 0) for r in records]
            m = np.mean(values)
            lo, hi = _bootstrap_ci(values)
            wrong_means.append(m)
            wrong_errors.append([m - lo, hi - m])

        # Correct efficacy
        correct_means = []
        correct_errors = []
        for model_id in models:
            records = model_results[model_id]
            values = [r["selectivity_metrics"].get(f"efficacy_correct_{tag_key}", 0) for r in records]
            m = np.mean(values)
            lo, hi = _bootstrap_ci(values)
            correct_means.append(m)
            correct_errors.append([m - lo, hi - m])

        ax.bar(x - width/2, wrong_means, width,
               yerr=np.array(wrong_errors).T, capsize=4,
               label="Wrong endorsement", color=COLORS["wrong"], alpha=0.8)
        ax.bar(x + width/2, correct_means, width,
               yerr=np.array(correct_errors).T, capsize=4,
               label="Correct endorsement", color=COLORS["correct"], alpha=0.8)

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([_get_model_label(m) for m in models], fontsize=10)
        ax.set_ylabel("Efficacy (instruction reduction)", fontsize=10)
        ax.set_title(f"{tag}", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)

        # Add value labels
        for j in range(len(models)):
            ax.annotate(f"{wrong_means[j]:.3f}", xy=(x[j] - width/2, wrong_means[j]),
                       xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
            ax.annotate(f"{correct_means[j]:.3f}", xy=(x[j] + width/2, correct_means[j]),
                       xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)

    plt.suptitle("Instruction Efficacy: Wrong vs Correct Endorsement\n(Higher = instruction helps more)", fontsize=12)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "efficacy_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "efficacy_comparison.pdf", bbox_inches="tight")
    plt.close()


def plot_effect_sizes(
    model_results: Dict[str, List[Dict]],
    tags: List[str],
    output_dir: Path,
) -> None:
    """Plot effect sizes before and after instruction."""
    if not model_results:
        return

    models = list(model_results.keys())

    for model_id in models:
        records = model_results[model_id]

        fig, axes = plt.subplots(1, len(tags), figsize=(5 * len(tags), 5), squeeze=False)
        axes = axes[0]

        for ax, tag in zip(axes, tags):
            tag_key = normalize_tag(tag)

            # Get effect sizes
            effects = {
                "Wrong (I0)": [r["selectivity_metrics"].get(f"effect_wrong_I0_{tag_key}", 0) for r in records],
                "Wrong (I1)": [r["selectivity_metrics"].get(f"effect_wrong_I1_{tag_key}", 0) for r in records],
                "Correct (I0)": [r["selectivity_metrics"].get(f"effect_correct_I0_{tag_key}", 0) for r in records],
                "Correct (I1)": [r["selectivity_metrics"].get(f"effect_correct_I1_{tag_key}", 0) for r in records],
            }

            x = np.arange(4)
            labels = list(effects.keys())
            means = [np.mean(v) for v in effects.values()]
            errors = []
            for v in effects.values():
                lo, hi = _bootstrap_ci(v)
                m = np.mean(v)
                errors.append([m - lo, hi - m])

            colors = [COLORS["wrong"], COLORS["wrong"], COLORS["correct"], COLORS["correct"]]
            alphas = [0.6, 1.0, 0.6, 1.0]

            bars = ax.bar(x, means, yerr=np.array(errors).T, capsize=4,
                         color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
            for bar, alpha in zip(bars, alphas):
                bar.set_alpha(alpha)

            ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9, rotation=15)
            ax.set_ylabel("Effect size (shift toward endorsed)", fontsize=10)
            ax.set_title(f"{tag}", fontsize=11)

            for i, m in enumerate(means):
                ax.annotate(f"{m:.3f}", xy=(i, m), xytext=(0, 5 if m >= 0 else -12),
                           textcoords="offset points", ha="center", fontsize=8)

        plt.suptitle(f"Effect Sizes: {_get_model_label(model_id)}\n(I0=no instruction, I1=with instruction)", fontsize=12)
        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        model_tag = model_id.replace("/", "__")
        plt.savefig(output_dir / f"effect_sizes_{model_tag}.png", dpi=150, bbox_inches="tight")
        plt.savefig(output_dir / f"effect_sizes_{model_tag}.pdf", bbox_inches="tight")
        plt.close()


def generate_text_report(
    summary: Dict,
    model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Generate text report with selectivity analysis."""
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT 10: CORRECT-ENDORSEMENT TEST")
    lines.append("Truth-Tracking vs Uniform Gating")
    lines.append("=" * 70)
    lines.append("")
    lines.append("QUESTION: Is instruction suppression truth-tracking (selective) or")
    lines.append("          uniform endorsement-ignore gating?")
    lines.append("")
    lines.append(f"INSTRUCTION: \"{INSTRUCTION_TEXT}\"")
    lines.append("")
    lines.append("DESIGN: 3x2 factorial per tag")
    lines.append("  N0: Neutral, no instruction")
    lines.append("  N1: Neutral, with instruction")
    lines.append("  W0: Wrong endorsement, no instruction")
    lines.append("  W1: Wrong endorsement, with instruction")
    lines.append("  C0: Correct endorsement, no instruction")
    lines.append("  C1: Correct endorsement, with instruction")
    lines.append("")
    lines.append("METRICS (positive = endorsement worked / instruction helped):")
    lines.append("  effect_wrong_I0   = P(wrong|W0) - P(wrong|N0) = N0 - W0 in P(correct)")
    lines.append("  effect_wrong_I1   = P(wrong|W1) - P(wrong|N1) = N1 - W1 in P(correct)")
    lines.append("  effect_correct_I0 = P(correct|C0) - P(correct|N0) = C0 - N0")
    lines.append("  effect_correct_I1 = P(correct|C1) - P(correct|N1) = C1 - N1")
    lines.append("  efficacy_wrong    = effect_wrong_I0 - effect_wrong_I1")
    lines.append("  efficacy_correct  = effect_correct_I0 - effect_correct_I1")
    lines.append("  SELECTIVITY       = efficacy_wrong - efficacy_correct")
    lines.append("")
    lines.append("INTERPRETATION:")
    lines.append("  selectivity > 0:  TRUTH-TRACKING (suppresses wrong MORE than correct)")
    lines.append("  selectivity ~ 0:  UNIFORM GATING (suppresses both equally)")
    lines.append("  selectivity < 0:  INVERSE (suppresses correct MORE than wrong)")
    lines.append("")
    lines.append("-" * 70)
    lines.append("RESULTS BY MODEL")
    lines.append("-" * 70)

    tags = summary.get("tags_tested", list(DEFAULT_TAGS))

    for model_id, records in model_results.items():
        lines.append("")
        lines.append(f"Model: {model_id}")
        lines.append(f"  N examples: {len(records)}")

        for tag in tags:
            tag_key = normalize_tag(tag)
            lines.append(f"\n  {tag} tag:")

            metrics = [
                "effect_wrong_I0", "effect_wrong_I1",
                "effect_correct_I0", "effect_correct_I1",
                "efficacy_wrong", "efficacy_correct",
                "selectivity", "baseline_shift"
            ]

            for metric in metrics:
                key = f"{metric}_{tag_key}"
                values = [r["selectivity_metrics"].get(key, 0) for r in records]
                ci_lo, ci_hi = _bootstrap_ci(values)
                stats = _summary_stats(values)
                lines.append(f"    {metric}:")
                lines.append(f"      Mean:   {stats['mean']:.4f}")
                lines.append(f"      95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
                lines.append(f"      Positive frac: {stats['positive_frac']:.1%}")

    lines.append("")
    lines.append("-" * 70)
    lines.append("SELECTIVITY SUMMARY (CRITICAL)")
    lines.append("-" * 70)

    for model_id, records in model_results.items():
        model_label = _get_model_label(model_id)
        lines.append(f"\n{model_label}:")

        for tag in tags:
            tag_key = normalize_tag(tag)
            sel_values = [r["selectivity_metrics"].get(f"selectivity_{tag_key}", 0) for r in records]
            eff_wrong_values = [r["selectivity_metrics"].get(f"efficacy_wrong_{tag_key}", 0) for r in records]
            eff_correct_values = [r["selectivity_metrics"].get(f"efficacy_correct_{tag_key}", 0) for r in records]

            sel_mean = np.mean(sel_values)
            sel_ci = _bootstrap_ci(sel_values)
            eff_wrong_mean = np.mean(eff_wrong_values)
            eff_correct_mean = np.mean(eff_correct_values)

            lines.append(f"\n  {tag}:")
            lines.append(f"    Efficacy (wrong):   {eff_wrong_mean:.4f}")
            lines.append(f"    Efficacy (correct): {eff_correct_mean:.4f}")
            lines.append(f"    SELECTIVITY:        {sel_mean:.4f} (CI: [{sel_ci[0]:.4f}, {sel_ci[1]:.4f}])")

            # Interpretation
            if sel_ci[0] > 0.02:
                lines.append(f"    VERDICT: TRUTH-TRACKING (selectivity significantly positive)")
            elif sel_ci[1] < -0.02:
                lines.append(f"    VERDICT: INVERSE (selectivity significantly negative)")
            elif abs(sel_mean) < 0.02:
                lines.append(f"    VERDICT: UNIFORM GATING (selectivity near zero)")
            else:
                lines.append(f"    VERDICT: INCONCLUSIVE (confidence interval spans zero)")

    lines.append("")
    lines.append("=" * 70)
    lines.append("IMPLICATIONS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("If TRUTH-TRACKING (selectivity > 0):")
    lines.append("  - Model discriminates between correct/wrong endorsements")
    lines.append("  - Instruction suppression is content-aware")
    lines.append("  - More sophisticated than simple endorsement-ignoring")
    lines.append("")
    lines.append("If UNIFORM GATING (selectivity ~ 0):")
    lines.append("  - Model treats all endorsements the same")
    lines.append("  - Instruction simply gates out speaker signals")
    lines.append("  - Not truly tracking correctness")
    lines.append("")
    lines.append("If INVERSE (selectivity < 0):")
    lines.append("  - Pathological: instruction hurts correct endorsements more")
    lines.append("  - May indicate instruction interferes with reasoning")
    lines.append("")

    report_text = "\n".join(lines)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "report.txt").open("w") as f:
        f.write(report_text)

    print(report_text)


def main() -> None:
    args = parse_args()

    print(f"Loading results from {args.results_dir}")
    summary, model_results = load_results(args.results_dir)

    tags = summary.get("tags_tested", list(DEFAULT_TAGS))

    print(f"Generating figures in {args.output_dir}")
    plot_selectivity_bars(model_results, tags, args.output_dir)
    plot_3x2_condition_grid(model_results, tags, args.output_dir)
    plot_efficacy_comparison(model_results, tags, args.output_dir)
    plot_effect_sizes(model_results, tags, args.output_dir)

    print("Generating text report")
    generate_text_report(summary, model_results, args.output_dir)

    print(f"\nAnalysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
