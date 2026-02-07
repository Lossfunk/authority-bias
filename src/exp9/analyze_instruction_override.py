"""Analysis and visualization for Experiment 9 (Instruction Override).

Tests whether endorsement effect is compliance vs belief-updating by
measuring how much an explicit "be correct" instruction suppresses
the endorsement effect.

Generates:
- 2x2 condition comparison (I0, I1, E0, E1) per tag
- Instruction efficacy bars (diff-in-diff)
- Before/after instruction comparison
- Sanity check: baseline shift
- Text report with interpretation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.exp9.instruction_conditions import INSTRUCTION_TAGS, INSTRUCTION_TEXT


# Okabe-Ito colorblind-friendly palette
COLORS = {
    "no_instr": "#E69F00",      # Orange
    "with_instr": "#0072B2",    # Blue
    "Expert": "#D55E00",        # Red-orange
    "Note": "#009E73",          # Green
    "instruct": "#0072B2",
    "base": "#E69F00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp9 Instruction Override results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/exp9"),
        help="Directory containing exp9 results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/exp9/figures"),
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


def plot_2x2_conditions(
    model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Plot fc_wrong for all 4 conditions (I0, I1, E0, E1) per tag."""
    if not model_results:
        return

    models = list(model_results.keys())
    tags = ["expert", "note"]
    conditions = ["I0", "I1", "E0", "E1"]
    cond_labels = ["Neutral\n(no instr)", "Neutral\n(+ instr)", "Endorse\n(no instr)", "Endorse\n(+ instr)"]

    for tag in tags:
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False)
        axes = axes[0]

        for ax, model_id in zip(axes, models):
            records = model_results[model_id]
            means = []
            errors_low = []
            errors_high = []

            for cond in conditions:
                key = f"{tag}_{cond}"
                values = [r["fc_wrong"][key] for r in records]
                m = np.mean(values)
                lo, hi = _bootstrap_ci(values)
                means.append(m)
                errors_low.append(m - lo)
                errors_high.append(hi - m)

            x = np.arange(len(conditions))
            colors = [COLORS["no_instr"], COLORS["with_instr"], COLORS["no_instr"], COLORS["with_instr"]]
            ax.bar(x, means, yerr=[errors_low, errors_high], capsize=4,
                   color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

            ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
            ax.set_xticks(x)
            ax.set_xticklabels(cond_labels, fontsize=9)
            ax.set_ylabel("P(Wrong | A/B)", fontsize=10)
            ax.set_title(f"{_get_model_label(model_id)}", fontsize=11)
            ax.set_ylim(0, 1)

            for i, m in enumerate(means):
                ax.annotate(f"{m:.3f}", xy=(i, m), xytext=(0, 5),
                           textcoords="offset points", ha="center", fontsize=8)

        plt.suptitle(f"2x2 Conditions: {tag.capitalize()} Tag", fontsize=12)
        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"conditions_2x2_{tag}.png", dpi=150, bbox_inches="tight")
        plt.savefig(output_dir / f"conditions_2x2_{tag}.pdf", bbox_inches="tight")
        plt.close()


def plot_instruction_efficacy(
    model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Plot instruction efficacy (diff-in-diff) for each tag."""
    if not model_results:
        return

    models = list(model_results.keys())
    tags = ["Expert", "Note"]

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(tags))
    width = 0.35
    offsets = [-width/2, width/2] if len(models) == 2 else [0]

    for i, model_id in enumerate(models):
        records = model_results[model_id]
        means = []
        errors_low = []
        errors_high = []

        for tag in tags:
            values = [r["decomposition"][f"instruction_efficacy_{tag}"] for r in records]
            m = np.mean(values)
            lo, hi = _bootstrap_ci(values)
            means.append(m)
            errors_low.append(m - lo)
            errors_high.append(hi - m)

        color = COLORS["instruct"] if "instruct" in model_id.lower() else COLORS["base"]
        offset = offsets[i] if i < len(offsets) else 0
        ax.bar(x + offset, means, width, yerr=[errors_low, errors_high],
               label=_get_model_label(model_id), color=color, alpha=0.8, capsize=4)

    ax.axhline(y=0, color="red", linestyle="--", linewidth=2, label="No effect")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=11)
    ax.set_ylabel("Instruction Efficacy (diff-in-diff)", fontsize=11)
    ax.set_title("Instruction Override Effect\n(Positive = instruction suppresses endorsement)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)

    for i, model_id in enumerate(models):
        records = model_results[model_id]
        offset = offsets[i] if i < len(offsets) else 0
        for j, tag in enumerate(tags):
            values = [r["decomposition"][f"instruction_efficacy_{tag}"] for r in records]
            m = np.mean(values)
            ax.annotate(f"{m:.2f}", xy=(x[j] + offset, m),
                       xytext=(0, 5 if m >= 0 else -12),
                       textcoords="offset points", ha="center",
                       fontsize=9, fontweight="bold")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "instruction_efficacy.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "instruction_efficacy.pdf", bbox_inches="tight")
    plt.close()


def plot_before_after_comparison(
    model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Plot endorsement effect before and after instruction."""
    if not model_results:
        return

    models = list(model_results.keys())
    tags = ["Expert", "Note"]

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False)
    axes = axes[0]

    for ax, model_id in zip(axes, models):
        records = model_results[model_id]

        x = np.arange(len(tags))
        width = 0.35

        # Before instruction
        means_before = []
        errors_before = []
        for tag in tags:
            values = [r["decomposition"][f"endorse_no_instr_{tag}"] for r in records]
            m = np.mean(values)
            lo, hi = _bootstrap_ci(values)
            means_before.append(m)
            errors_before.append([m - lo, hi - m])

        # After instruction
        means_after = []
        errors_after = []
        for tag in tags:
            values = [r["decomposition"][f"endorse_instr_{tag}"] for r in records]
            m = np.mean(values)
            lo, hi = _bootstrap_ci(values)
            means_after.append(m)
            errors_after.append([m - lo, hi - m])

        ax.bar(x - width/2, means_before,width,
               yerr=np.array(errors_before).T, capsize=4,
               label="Without instruction", color=COLORS["no_instr"], alpha=0.8)
        ax.bar(x + width/2, means_after, width,
               yerr=np.array(errors_after).T, capsize=4,
               label="With instruction", color=COLORS["with_instr"], alpha=0.8)

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(tags, fontsize=11)
        ax.set_ylabel("Endorsement Effect (logit shift)", fontsize=10)
        ax.set_title(f"{_get_model_label(model_id)}", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)

        # Add value labels
        for j in range(len(tags)):
            ax.annotate(f"{means_before[j]:.2f}", xy=(x[j] - width/2, means_before[j]),
                       xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
            ax.annotate(f"{means_after[j]:.2f}", xy=(x[j] + width/2, means_after[j]),
                       xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)

    plt.suptitle("Endorsement Effect: Before vs After Instruction", fontsize=12)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "before_after_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "before_after_comparison.pdf", bbox_inches="tight")
    plt.close()


def plot_baseline_shift_sanity(
    model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Plot baseline shift (I1 - I0) as sanity check."""
    if not model_results:
        return

    models = list(model_results.keys())
    tags = ["Expert", "Note"]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(tags))
    width = 0.35
    offsets = [-width/2, width/2] if len(models) == 2 else [0]

    for i, model_id in enumerate(models):
        records = model_results[model_id]
        means = []
        errors_low = []
        errors_high = []

        for tag in tags:
            values = [r["decomposition"][f"baseline_shift_{tag}"] for r in records]
            m = np.mean(values)
            lo, hi = _bootstrap_ci(values)
            means.append(m)
            errors_low.append(m - lo)
            errors_high.append(hi - m)

        color = COLORS["instruct"] if "instruct" in model_id.lower() else COLORS["base"]
        offset = offsets[i] if i < len(offsets) else 0
        ax.bar(x + offset, means, width, yerr=[errors_low, errors_high],
               label=_get_model_label(model_id), color=color, alpha=0.8, capsize=4)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=11)
    ax.set_ylabel("Baseline Shift (I1 - I0)", fontsize=11)
    ax.set_title("Sanity Check: Does Instruction Alone Shift P(Wrong)?\n(Should be ~0)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "baseline_shift_sanity.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "baseline_shift_sanity.pdf", bbox_inches="tight")
    plt.close()


def generate_text_report(
    summary: Dict,
    model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Generate text report with instruction override analysis."""
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT 9: INSTRUCTION OVERRIDE TEST")
    lines.append("Compliance vs Belief-Updating")
    lines.append("=" * 70)
    lines.append("")
    lines.append("QUESTION: Is the endorsement effect instruction-overridable compliance")
    lines.append("          or deep belief-like updating?")
    lines.append("")
    lines.append(f"INSTRUCTION: \"{INSTRUCTION_TEXT}\"")
    lines.append("")
    lines.append("DESIGN: 2x2 factorial per tag")
    lines.append("  I0: Neutral, no instruction")
    lines.append("  I1: Neutral, with instruction")
    lines.append("  E0: Endorse, no instruction")
    lines.append("  E1: Endorse, with instruction")
    lines.append("")
    lines.append("METRICS:")
    lines.append("  Endorse (no instr)   = E0 - I0")
    lines.append("  Endorse (with instr) = E1 - I1")
    lines.append("  Instruction efficacy = (E0 - I0) - (E1 - I1)  [diff-in-diff]")
    lines.append("  Baseline shift       = I1 - I0  [sanity check]")
    lines.append("")
    lines.append("INTERPRETATION:")
    lines.append("  Efficacy >> 0: Instruction suppresses effect -> COMPLIANCE")
    lines.append("  Efficacy ~ 0:  Effect persists -> DEEP INTEGRATION")
    lines.append("")
    lines.append("-" * 70)
    lines.append("RESULTS BY MODEL")
    lines.append("-" * 70)

    tags = ["Expert", "Note"]
    metrics = ["endorse_no_instr", "endorse_instr", "instruction_efficacy", "baseline_shift"]

    for model_id, records in model_results.items():
        lines.append("")
        lines.append(f"Model: {model_id}")
        lines.append(f"  N examples: {len(records)}")

        for tag in tags:
            lines.append(f"\n  {tag} tag:")
            for metric in metrics:
                key = f"{metric}_{tag}"
                values = [r["decomposition"][key] for r in records]
                ci_lo, ci_hi = _bootstrap_ci(values)
                stats = _summary_stats(values)
                lines.append(f"    {metric}:")
                lines.append(f"      Mean:   {stats['mean']:.4f}")
                lines.append(f"      95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
                lines.append(f"      Median: {stats['median']:.4f}")
                lines.append(f"      Positive frac: {stats['positive_frac']:.1%}")

            # Compute reduction percentage
            endorse_no = np.mean([r["decomposition"][f"endorse_no_instr_{tag}"] for r in records])
            efficacy = np.mean([r["decomposition"][f"instruction_efficacy_{tag}"] for r in records])
            reduction_pct = (efficacy / endorse_no * 100) if endorse_no != 0 else 0

            lines.append(f"\n    Reduction: {reduction_pct:.1f}%")
            if reduction_pct > 80:
                lines.append(f"    => STRONG COMPLIANCE: Instruction nearly eliminates effect")
            elif reduction_pct > 40:
                lines.append(f"    => PARTIAL: Instruction helps but doesn't eliminate")
            elif reduction_pct > 10:
                lines.append(f"    => WEAK: Small instruction effect")
            else:
                lines.append(f"    => PERSISTENT: Effect largely unaffected by instruction")

        # Stratify by dataset
        lines.append(f"\n  Stratified by dataset:")
        datasets = sorted({r.get("metadata", {}).get("dataset", "unknown") for r in records})
        for ds in datasets:
            ds_records = [r for r in records if r.get("metadata", {}).get("dataset", "unknown") == ds]
            lines.append(f"    Dataset: {ds} (n={len(ds_records)})")
            for tag in tags:
                efficacy_vals = [r["decomposition"][f"instruction_efficacy_{tag}"] for r in ds_records]
                endorse_no_vals = [r["decomposition"][f"endorse_no_instr_{tag}"] for r in ds_records]
                eff_mean = np.mean(efficacy_vals)
                endorse_no_mean = np.mean(endorse_no_vals)
                reduction = (eff_mean / endorse_no_mean * 100) if endorse_no_mean != 0 else 0
                lines.append(f"      {tag}: efficacy={eff_mean:.3f}, reduction={reduction:.1f}%")

    lines.append("")
    lines.append("-" * 70)
    lines.append("DECISION SUMMARY")
    lines.append("-" * 70)

    for model_id, records in model_results.items():
        model_label = _get_model_label(model_id)
        lines.append(f"\n{model_label}:")

        for tag in tags:
            endorse_no = np.mean([r["decomposition"][f"endorse_no_instr_{tag}"] for r in records])
            endorse_with = np.mean([r["decomposition"][f"endorse_instr_{tag}"] for r in records])
            efficacy = np.mean([r["decomposition"][f"instruction_efficacy_{tag}"] for r in records])
            baseline = np.mean([r["decomposition"][f"baseline_shift_{tag}"] for r in records])

            efficacy_ci = _bootstrap_ci([r["decomposition"][f"instruction_efficacy_{tag}"] for r in records])

            lines.append(f"\n  {tag}:")
            lines.append(f"    Endorse (no instr):   {endorse_no:.3f}")
            lines.append(f"    Endorse (with instr): {endorse_with:.3f}")
            lines.append(f"    Instruction efficacy: {efficacy:.3f} (CI: [{efficacy_ci[0]:.3f}, {efficacy_ci[1]:.3f}])")
            lines.append(f"    Baseline shift:       {baseline:.3f}")

            reduction_pct = (efficacy / endorse_no * 100) if endorse_no != 0 else 0
            lines.append(f"    Reduction:            {reduction_pct:.1f}%")

            if efficacy_ci[0] > 0:
                if reduction_pct > 60:
                    lines.append(f"    VERDICT: COMPLIANCE (instruction significantly reduces effect)")
                else:
                    lines.append(f"    VERDICT: MIXED (instruction helps but effect persists)")
            else:
                lines.append(f"    VERDICT: PERSISTENT (instruction doesn't reliably reduce effect)")

    lines.append("")
    lines.append("=" * 70)
    lines.append("IMPLICATIONS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("If COMPLIANCE (efficacy high):")
    lines.append("  - Endorsement effect is 'shallow' instruction-following")
    lines.append("  - Prompting-based mitigation is viable")
    lines.append("  - Less concerning for deep belief corruption")
    lines.append("")
    lines.append("If PERSISTENT (efficacy low):")
    lines.append("  - Effect operates below explicit instruction level")
    lines.append("  - May require mechanistic intervention")
    lines.append("  - More concerning: hard to override with prompts")
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

    print(f"Generating figures in {args.output_dir}")
    plot_2x2_conditions(model_results, args.output_dir)
    plot_instruction_efficacy(model_results, args.output_dir)
    plot_before_after_comparison(model_results, args.output_dir)
    plot_baseline_shift_sanity(model_results, args.output_dir)

    print("Generating text report")
    generate_text_report(summary, model_results, args.output_dir)

    print(f"\nAnalysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
