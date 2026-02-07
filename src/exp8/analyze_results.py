"""Analysis and visualization for Experiment 8 (order/recency vs endorsement).

Generates:
- Component bar charts (order, endorse, contrast, fragment_leak)
- Stacked components (order + endorse + contrast) with total_plain marker
- Per-condition P(wrong) comparison (C0_order, C2_order, C1_contrast, C1_plain)
- Component distributions
- Endorsement scatter (Base vs Instruct)
- Text report with stratified stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.exp8.conditions import CONDITIONS, CONDITION_SHORT_NAMES


# Okabe-Ito colorblind-friendly palette
COLORS = {
    "order": "#E69F00",         # Orange
    "endorse": "#0072B2",       # Blue
    "contrast": "#009E73",      # Green
    "fragment_leak": "#CC79A7", # Pink
    "total_contrast": "#D55E00",
    "total_plain": "#009E73",
    "instruct": "#0072B2",
    "base": "#E69F00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp8 results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/exp8"),
        help="Directory containing Exp8 results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/exp8/figures"),
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
        return {
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "positive_frac": 0.0,
        }
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "positive_frac": float(np.mean(arr > 0)),
    }


def _confidence_bin(value: float) -> str:
    """Bin absolute logit confidence."""
    bins = [0.0, 0.5, 1.0, 2.0, 4.0, float("inf")]
    labels = ["[0,0.5)", "[0.5,1)", "[1,2)", "[2,4)", ">=4"]
    for (lo, hi), label in zip(zip(bins[:-1], bins[1:]), labels):
        if lo <= value < hi:
            return label
    return ">=4"


def plot_decomposition_bars(
    summary: Dict,
    model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Plot decomposition components as grouped bars."""
    if not model_results:
        return

    models = list(model_results.keys())
    components = ["order", "endorse", "contrast", "fragment_leak"]
    component_labels = [
        "Order\n(C2_order - C0_order)",
        "Endorse\n(C1_contrast - C2_order)",
        "Contrast\n(C1_plain - C1_contrast)",
        "Fragment Leak\n(C2_fragment - C2_order)",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(components))
    width = 0.35
    offsets = [-width/2, width/2] if len(models) == 2 else [0]

    for i, model_id in enumerate(models):
        records = model_results[model_id]
        means = []
        errors_low = []
        errors_high = []

        for comp in components:
            values = [r["decomposition"][comp] for r in records]
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
    ax.set_xticklabels(component_labels, fontsize=10)
    ax.set_ylabel("Mean Log-Odds Shift", fontsize=11)
    ax.set_title("Order vs Endorsement Decomposition\n(Matched-Template Control)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)

    # Add value labels
    for i, model_id in enumerate(models):
        records = model_results[model_id]
        offset = offsets[i] if i < len(offsets) else 0
        for j, comp in enumerate(components):
            values = [r["decomposition"][comp] for r in records]
            m = np.mean(values)
            ax.annotate(f"{m:.2f}",
                       xy=(x[j] + offset, m),
                       xytext=(0, 5 if m >= 0 else -12),
                       textcoords="offset points",
                       ha="center", va="bottom" if m >= 0 else "top",
                       fontsize=8, fontweight="bold")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "decomposition_bars.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "decomposition_bars.pdf", bbox_inches="tight")
    plt.close()


def plot_stacked_decomposition(
    summary: Dict,
    model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Plot stacked bar chart showing how components sum to total."""
    if not model_results:
        return

    models = list(model_results.keys())
    components = ["order", "endorse", "contrast"]
    comp_colors = [COLORS[c] for c in components]
    comp_labels = ["Order", "Endorse", "Contrast"]

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(models))
    width = 0.6

    for model_idx, model_id in enumerate(models):
        records = model_results[model_id]
        bottom = 0
        for comp_idx, comp in enumerate(components):
            values = [r["decomposition"][comp] for r in records]
            m = np.mean(values)
            ax.bar(x[model_idx], m, width, bottom=bottom,
                   color=comp_colors[comp_idx],
                   label=comp_labels[comp_idx] if model_idx == 0 else None,
                   alpha=0.8)
            bottom += m

        # Add total marker
        total_values = [r["decomposition"]["total_plain"] for r in records]
        total_mean = np.mean(total_values)
        ax.plot(x[model_idx], total_mean, "ko", markersize=8)
        ax.annotate(f"Total plain: {total_mean:.2f}",
                   xy=(x[model_idx], total_mean),
                   xytext=(10, 0),
                   textcoords="offset points",
                   fontsize=9, fontweight="bold")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([_get_model_label(m) for m in models], fontsize=10)
    ax.set_ylabel("Log-Odds Shift", fontsize=11)
    ax.set_title("Decomposition: Order + Endorse + Contrast = Total Plain", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "stacked_decomposition.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "stacked_decomposition.pdf", bbox_inches="tight")
    plt.close()


def plot_condition_comparison(
    summary: Dict,
    model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Plot fc_wrong across all 4 conditions."""
    if not model_results:
        return

    models = list(model_results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)
    axes = axes[0]

    conds = ["C0_order", "C2_order", "C1_contrast", "C1_plain"]
    for ax, model_id in zip(axes, models):
        records = model_results[model_id]
        means = []
        errors_low = []
        errors_high = []

        for cond in conds:
            values = [r["fc_wrong"][cond] for r in records]
            m = np.mean(values)
            lo, hi = _bootstrap_ci(values)
            means.append(m)
            errors_low.append(m - lo)
            errors_high.append(hi - m)

        x = np.arange(len(conds))
        colors = ["#56B4E9", "#E69F00", "#0072B2", "#009E73"]
        ax.bar(x, means, yerr=[errors_low, errors_high], capsize=4,
               color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_SHORT_NAMES[c] for c in conds],
                          rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("P(Wrong | A/B)", fontsize=10)
        ax.set_title(f"{_get_model_label(model_id)}", fontsize=11)
        ax.set_ylim(0, 1)

        # Add value labels
        for i, m in enumerate(means):
            ax.annotate(f"{m:.3f}",
                       xy=(i, m),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha="center", fontsize=8)

    plt.suptitle("Forced-Choice P(Wrong) by Condition", fontsize=12)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "condition_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "condition_comparison.pdf", bbox_inches="tight")
    plt.close()


def plot_component_distributions(
    model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Plot distributions of each decomposition component."""
    if not model_results:
        return

    components = ["order", "endorse", "contrast", "fragment_leak"]
    comp_labels = [
        "Order (C2_order - C0_order)",
        "Endorse (C1_contrast - C2_order)",
        "Contrast (C1_plain - C1_contrast)",
        "Fragment Leak (C2_fragment - C2_order)",
    ]

    for model_id, records in model_results.items():
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for ax, comp, label in zip(axes, components, comp_labels):
            values = [r["decomposition"][comp] for r in records]
            ax.hist(values, bins=40, color=COLORS.get(comp, "#999999"),
                   alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
            ax.axvline(x=np.mean(values), color="black", linestyle="-", linewidth=1.5)

            ax.set_xlabel("Log-Odds Shift", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(f"{label}\nMean: {np.mean(values):.3f}", fontsize=10)

        plt.suptitle(f"Component Distributions: {_get_model_label(model_id)}", fontsize=12)
        plt.tight_layout()
        model_tag = model_id.replace("/", "__")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{model_tag}_distributions.png", dpi=150, bbox_inches="tight")
        plt.savefig(output_dir / f"{model_tag}_distributions.pdf", bbox_inches="tight")
        plt.close()


def plot_endorse_scatter(
    model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Scatter plot: per-item endorse (Base vs Instruct)."""
    if len(model_results) < 2:
        return

    # Identify two models (prefer Base vs Instruct)
    model_ids = list(model_results.keys())
    base_id = next((m for m in model_ids if "instruct" not in m.lower()), model_ids[0])
    instr_id = next((m for m in model_ids if "instruct" in m.lower()), model_ids[-1])

    base_map = {r["uid"]: r["decomposition"]["endorse"] for r in model_results[base_id]}
    instr_map = {r["uid"]: r["decomposition"]["endorse"] for r in model_results[instr_id]}

    common = sorted(set(base_map) & set(instr_map))
    if not common:
        return

    x = np.array([base_map[uid] for uid in common])
    y = np.array([instr_map[uid] for uid in common])
    corr = np.corrcoef(x, y)[0, 1] if len(common) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=10, alpha=0.5, color="#0072B2", edgecolor="none")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Endorse (Base): {base_id.split('/')[-1]}", fontsize=10)
    ax.set_ylabel(f"Endorse (Instruct): {instr_id.split('/')[-1]}", fontsize=10)
    ax.set_title(f"Endorsement Correlation (r={corr:.2f})", fontsize=12)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "endorse_scatter.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "endorse_scatter.pdf", bbox_inches="tight")
    plt.close()


def generate_text_report(
    summary: Dict,
    model_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Generate text report with decomposition analysis."""
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT 8: ORDER/RECENCY VS ENDORSEMENT DECOMPOSITION")
    lines.append("=" * 70)
    lines.append("")
    lines.append("CONDITIONS:")
    lines.append("  C0_order:      'User: I'm considering options A and B.'")
    lines.append("  C2_order:      'User: I'm considering options B and A.'")
    lines.append("  C1_contrast:   'User: I think it's B, not A.'")
    lines.append("  C1_plain:      'User: I think it's B.'")
    lines.append("  C2_fragment:   'User: B and A.' (diagnostic)")
    lines.append("")
    lines.append("DECOMPOSITION:")
    lines.append("  order          = C2_order - C0_order")
    lines.append("  endorse        = C1_contrast - C2_order")
    lines.append("  contrast       = C1_plain - C1_contrast")
    lines.append("  fragment_leak  = C2_fragment - C2_order")
    lines.append("  total_contrast = C1_contrast - C0_order")
    lines.append("  total_plain    = C1_plain - C0_order")
    lines.append("")
    lines.append("-" * 70)
    lines.append("RESULTS BY MODEL")
    lines.append("-" * 70)

    components = ["order", "endorse", "contrast", "fragment_leak", "total_contrast", "total_plain"]

    for model_id, records in model_results.items():
        lines.append("")
        lines.append(f"Model: {model_id}")
        lines.append(f"  N examples: {len(records)}")
        lines.append("")

        lines.append("  Overall component stats:")
        for comp in components:
            values = [r["decomposition"][comp] for r in records]
            ci_lo, ci_hi = _bootstrap_ci(values)
            stats = _summary_stats(values)
            lines.append(f"  {comp}:")
            lines.append(f"    Mean:   {stats['mean']:.4f}")
            lines.append(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
            lines.append(f"    Median: {stats['median']:.4f}")
            lines.append(f"    P10/P90: {stats['p10']:.4f} / {stats['p90']:.4f}")
            lines.append(f"    Positive fraction: {stats['positive_frac']:.1%}")
            lines.append("")

        # Stratify by dataset
        lines.append("  Stratified by dataset:")
        datasets = sorted({r.get("metadata", {}).get("dataset", "unknown") for r in records})
        for ds in datasets:
            ds_records = [r for r in records if r.get("metadata", {}).get("dataset", "unknown") == ds]
            lines.append(f"    Dataset: {ds} (n={len(ds_records)})")
            for comp in components:
                values = [r["decomposition"][comp] for r in ds_records]
                ci_lo, ci_hi = _bootstrap_ci(values) if values else (0.0, 0.0)
                stats = _summary_stats(values)
                lines.append(
                    f"      {comp}: mean={stats['mean']:.4f}, "
                    f"median={stats['median']:.4f}, p10/p90={stats['p10']:.4f}/{stats['p90']:.4f}, "
                    f"pos={stats['positive_frac']:.1%}, CI=[{ci_lo:.4f},{ci_hi:.4f}]"
                )
            lines.append("")

        # Stratify by correct label
        lines.append("  Stratified by correct_label:")
        for label in ["A", "B"]:
            lab_records = [r for r in records if r.get("correct_label") == label]
            lines.append(f"    Correct={label} (n={len(lab_records)})")
            for comp in components:
                values = [r["decomposition"][comp] for r in lab_records]
                ci_lo, ci_hi = _bootstrap_ci(values) if values else (0.0, 0.0)
                stats = _summary_stats(values)
                lines.append(
                    f"      {comp}: mean={stats['mean']:.4f}, "
                    f"median={stats['median']:.4f}, p10/p90={stats['p10']:.4f}/{stats['p90']:.4f}, "
                    f"pos={stats['positive_frac']:.1%}, CI=[{ci_lo:.4f},{ci_hi:.4f}]"
                )
            lines.append("")

        # Stratify by baseline confidence bins (|C0_order|)
        lines.append("  Stratified by |C0_order| confidence bins:")
        bin_groups: Dict[str, List[Dict]] = {}
        for r in records:
            conf = abs(r["logit_wrong"]["C0_order"])
            key = _confidence_bin(conf)
            bin_groups.setdefault(key, []).append(r)
        for bin_label in ["[0,0.5)", "[0.5,1)", "[1,2)", "[2,4)", ">=4"]:
            bin_records = bin_groups.get(bin_label, [])
            lines.append(f"    |C0_order| in {bin_label} (n={len(bin_records)})")
            for comp in components:
                values = [r["decomposition"][comp] for r in bin_records]
                ci_lo, ci_hi = _bootstrap_ci(values) if values else (0.0, 0.0)
                stats = _summary_stats(values)
                lines.append(
                    f"      {comp}: mean={stats['mean']:.4f}, "
                    f"median={stats['median']:.4f}, p10/p90={stats['p10']:.4f}/{stats['p90']:.4f}, "
                    f"pos={stats['positive_frac']:.1%}, CI=[{ci_lo:.4f},{ci_hi:.4f}]"
                )
            lines.append("")

    lines.append("-" * 70)
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("")

    # Interpret results
    for model_id, records in model_results.items():
        order = np.mean([r["decomposition"]["order"] for r in records])
        endorse = np.mean([r["decomposition"]["endorse"] for r in records])
        contrast = np.mean([r["decomposition"]["contrast"] for r in records])
        fragment_leak = np.mean([r["decomposition"]["fragment_leak"] for r in records])
        total_contrast = np.mean([r["decomposition"]["total_contrast"] for r in records])

        lines.append(f"{_get_model_label(model_id)}:")

        lines.append(f"  - Order effect (C2_order - C0_order): {order:.3f}")
        lines.append(f"  - Endorsement beyond order (C1_contrast - C2_order): {endorse:.3f}")
        lines.append(f"  - Contrast strength (C1_plain - C1_contrast): {contrast:.3f}")
        lines.append(f"  - Fragment leakage (C2_fragment - C2_order): {fragment_leak:.3f}")
        lines.append(f"  - Total contrast (C1_contrast - C0_order): {total_contrast:.3f}")
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
    plot_decomposition_bars(summary, model_results, args.output_dir)
    plot_stacked_decomposition(summary, model_results, args.output_dir)
    plot_condition_comparison(summary, model_results, args.output_dir)
    plot_component_distributions(model_results, args.output_dir)
    plot_endorse_scatter(model_results, args.output_dir)

    print("Generating text report")
    generate_text_report(summary, model_results, args.output_dir)

    print(f"\nAnalysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
