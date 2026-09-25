"""Part A: Logit-Space Replication of Exp10.

Replicates exp10 selectivity analysis in log-odds margin space with:
- Proper paired bootstrap
- Three-layer reporting (r_w, r_c, dr)
- Both sign-consistent and absolute mask robustness
- τ-sensitivity analysis with CIs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.prior_analysis.logit_metrics import (
    ItemMetrics,
    TAU_DEFAULT,
    TAU_VALUES,
    bootstrap_all_metrics_coherent,
    compute_item_metrics,
    compute_margin_from_logits,
    compute_margin_from_probs,
    run_tau_sensitivity_with_ci,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp11 Part A: Logit-Space Replication")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("new-phase-results/endorsement"),
        help="Directory containing exp10 JSONL results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("new-phase-results/prior_analysis/part_a"),
        help="Directory to save analysis outputs",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to analyze",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=TAU_DEFAULT,
        help="Threshold for validity mask",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=10000,
        help="Number of bootstrap replicates",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["expert", "note"],
        help=(
            "Tags to analyze (e.g., expert note user someone_online). "
            "Aliases supported: 'someone online', 'online', 'someone-online'."
        ),
    )
    return parser.parse_args()


def normalize_tag(raw_tag: str) -> str:
    """Normalize user-provided tag names to exp10 condition key format."""
    tag = raw_tag.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "someoneonline": "someone_online",
        "someone_online": "someone_online",
        "online": "someone_online",
    }
    return aliases.get(tag, tag)


def normalize_tags(raw_tags: List[str]) -> List[str]:
    """Normalize tags and preserve order while removing duplicates."""
    seen = set()
    normalized = []
    for raw in raw_tags:
        tag = normalize_tag(raw)
        if tag not in seen:
            seen.add(tag)
            normalized.append(tag)
    return normalized


def load_exp10_results(results_path: Path) -> List[Dict]:
    """Load per-example JSONL results from exp10."""
    records = []
    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_margins_per_tag(
    records: List[Dict],
    tag: str,
) -> List[ItemMetrics]:
    """Extract log-odds margins for a specific tag.

    Uses raw logits when available, falls back to probabilities.

    Args:
        records: List of per-example dicts from exp10 JSONL
        tag: Tag to extract (e.g., "expert", "note")

    Returns:
        List of ItemMetrics
    """
    items = []

    for record in records:
        uid = record["uid"]
        condition_results = record.get("condition_results", {})

        # Get results for each condition
        conditions = ["N0", "N1", "W0", "W1", "C0", "C1"]
        margins = {}

        for cond in conditions:
            cond_key = f"{cond}_{tag}"
            cond_data = condition_results.get(cond_key, {})

            # Priority 1: Use raw logits if available
            if "logit_a" in cond_data and "logit_b" in cond_data:
                l_c = cond_data["logit_a"]  # Assuming A is always correct (check correct_label)
                l_w = cond_data["logit_b"]

                # Verify label ordering (A should be correct)
                if record.get("correct_label") != "A":
                    # Swap if B is correct
                    l_c, l_w = l_w, l_c

                m = compute_margin_from_logits(l_c, l_w)

            # Priority 2: Fallback to forced-choice probabilities
            elif "fc_a" in cond_data and "fc_b" in cond_data:
                p_a = cond_data["fc_a"]
                p_b = cond_data["fc_b"]

                # Get correct probability
                if record.get("correct_label") == "A":
                    p_c = p_a
                else:
                    p_c = p_b

                m = compute_margin_from_probs(p_c)

            else:
                # Skip item if no data available
                print(f"Warning: Missing data for {uid} {cond_key}")
                continue

            margins[cond] = m

        # Skip if any condition is missing
        if len(margins) != 6:
            continue

        # Compute per-item metrics
        item = compute_item_metrics(
            m_N0=margins["N0"],
            m_N1=margins["N1"],
            m_W0=margins["W0"],
            m_W1=margins["W1"],
            m_C0=margins["C0"],
            m_C1=margins["C1"],
            question_id=uid,
        )
        items.append(item)

    return items


def format_metric_with_ci(value: Tuple[float, Tuple[float, float]]) -> str:
    """Format a metric with CI as 'mean [ci_low, ci_high]'."""
    if isinstance(value, tuple) and len(value) == 2:
        point, (ci_low, ci_high) = value
        return f"{point:.4f} [{ci_low:.4f}, {ci_high:.4f}]"
    else:
        return str(value)


def print_report(
    tag: str,
    model: str,
    metrics: Dict,
    tau_sensitivity: Dict,
    metrics_abs: Dict,
) -> None:
    """Print human-readable report for Part A."""
    print(f"\n{'='*80}")
    print(f"Part A: Logit-Space Replication")
    print(f"Model: {model}")
    print(f"Tag: {tag}")
    print(f"{'='*80}\n")

    # Primary additive metrics (stable when ratio masks thin out)
    print("Primary Additive Metrics (recommended):")
    print(f"  eff_diff = efficacy_w - efficacy_c:")
    print(f"    mean: {format_metric_with_ci(metrics['eff_diff_mean'])}")
    print(f"    median: {format_metric_with_ci(metrics['eff_diff_median'])}")
    print(f"    trimmed_mean(10%): {format_metric_with_ci(metrics['eff_diff_trimmed_mean'])}")
    print(f"  frac(efficacy_w > efficacy_c): {format_metric_with_ci(metrics['frac_eff_w_gt_c'])}")
    print(f"  baseline_shift (m_N1 - m_N0): {format_metric_with_ci(metrics['baseline_shift_mean'])}\n")

    # Primary analysis (sign-consistent mask)
    print(f"Sign-Consistent Mask (τ={TAU_DEFAULT}):")
    print(f"  n_total: {metrics['n_total']}\n")

    print(f"  Layer 1 (r_w, mask: effect_w > τ):")
    print(f"    n_mask_valid_w: {metrics['n_mask_valid_w']}, pct: {format_metric_with_ci(metrics['pct_mask_valid_w'])}")
    print(f"    n_finite_w: {metrics['n_finite_w']}, pct_ratio_nonfinite: {format_metric_with_ci(metrics['pct_ratio_nonfinite_given_valid_w'])}")
    print(f"    pct_boot_zero_mask_w: {metrics['pct_boot_zero_mask_w']:.2f}%, pct_boot_zero_finite_w: {metrics['pct_boot_zero_finite_w']:.2f}%")
    if "r_w_mean" in metrics:
        print(f"    r_w: mean {format_metric_with_ci(metrics['r_w_mean'])}, median {format_metric_with_ci(metrics['r_w_median'])}")
        print(f"    % outside [0,1]: {format_metric_with_ci(metrics['pct_r_w_outside'])}\n")

    print(f"  Layer 2 (r_c, mask: effect_c > τ):")
    print(f"    n_mask_valid_c: {metrics['n_mask_valid_c']}, pct: {format_metric_with_ci(metrics['pct_mask_valid_c'])}")
    print(f"    n_finite_c: {metrics['n_finite_c']}, pct_ratio_nonfinite: {format_metric_with_ci(metrics['pct_ratio_nonfinite_given_valid_c'])}")
    print(f"    pct_boot_zero_mask_c: {metrics['pct_boot_zero_mask_c']:.2f}%, pct_boot_zero_finite_c: {metrics['pct_boot_zero_finite_c']:.2f}%")
    if "r_c_mean" in metrics:
        print(f"    r_c: mean {format_metric_with_ci(metrics['r_c_mean'])}, median {format_metric_with_ci(metrics['r_c_median'])}")
        print(f"    % outside [0,1]: {format_metric_with_ci(metrics['pct_r_c_outside'])}\n")

    print(f"  Layer 3 (dr, intersection mask + finite):")
    print(f"    n_mask_valid_both: {metrics['n_mask_valid_both']}, pct: {format_metric_with_ci(metrics['pct_mask_valid_both'])}")
    print(f"    n_finite_both: {metrics['n_finite_both']}, pct_ratio_nonfinite: {format_metric_with_ci(metrics['pct_ratio_nonfinite_given_valid_both'])}")
    print(f"    pct_boot_zero_mask_both: {metrics['pct_boot_zero_mask_both']:.2f}%, pct_boot_zero_finite_both: {metrics['pct_boot_zero_finite_both']:.2f}%")
    if "dr_mean" in metrics:
        print(f"    dr = r_w - r_c: mean {format_metric_with_ci(metrics['dr_mean'])}, median {format_metric_with_ci(metrics['dr_median'])}")
        print(f"    frac(dr > 0): {format_metric_with_ci(metrics['frac_dr_pos'])}\n")

    print(f"  Unconditional:")
    print(f"    frac(eff_w > eff_c): {format_metric_with_ci(metrics['frac_eff_w_gt_c'])}")
    print(f"    eff_diff mean: {format_metric_with_ci(metrics['eff_diff_mean'])}")
    print(f"    eff_diff median: {format_metric_with_ci(metrics['eff_diff_median'])}")
    print(f"    eff_diff trimmed mean: {format_metric_with_ci(metrics['eff_diff_trimmed_mean'])}")
    print(f"    Baseline shift: {format_metric_with_ci(metrics['baseline_shift_mean'])}\n")

    # Robustness (absolute mask)
    print(f"Robustness (Absolute Mask, τ={TAU_DEFAULT}):")
    print(f"  n_mask_valid_abs: {metrics_abs['n_mask_valid_abs']}, pct: {format_metric_with_ci(metrics_abs['pct_mask_valid_abs'])}")
    print(f"  n_finite_abs: {metrics_abs['n_finite_abs']}, pct_ratio_nonfinite: {format_metric_with_ci(metrics_abs['pct_ratio_nonfinite_given_valid_abs'])}")
    print(f"  pct_boot_zero_mask_abs: {metrics_abs['pct_boot_zero_mask_abs']:.2f}%, pct_boot_zero_finite_abs: {metrics_abs['pct_boot_zero_finite_abs']:.2f}%")
    if "norm_sel_mean" in metrics_abs:
        print(f"  norm_sel (magnitude suppression): mean {format_metric_with_ci(metrics_abs['norm_sel_mean'])}, median {format_metric_with_ci(metrics_abs['norm_sel_median'])}\n")

    # τ-sensitivity table
    print(f"τ-Sensitivity Table (Sign-Consistent Mask):")
    print(f"  {'τ':<8} {'n_mask_w':<12} {'n_finite_w':<12} {'n_mask_c':<12} {'n_finite_c':<12} {'n_mask_both':<15} {'n_finite_both':<15} {'r_w mean [CI]':<35} {'r_c mean [CI]':<35} {'dr mean [CI]':<35}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*15} {'-'*15} {'-'*35} {'-'*35} {'-'*35}")

    for tau_key in sorted(tau_sensitivity.keys()):
        tau_results = tau_sensitivity[tau_key]
        tau = tau_results["tau"]
        print(f"  {tau:<8.0e} {tau_results['n_mask_valid_w']:<12} {tau_results['n_finite_w']:<12} {tau_results['n_mask_valid_c']:<12} {tau_results['n_finite_c']:<12} {tau_results['n_mask_valid_both']:<15} {tau_results['n_finite_both']:<15} {format_metric_with_ci(tau_results['r_w_mean']):<35} {format_metric_with_ci(tau_results['r_c_mean']):<35} {format_metric_with_ci(tau_results['dr_mean']):<35}")

    print(f"\n{'='*80}\n")


def main():
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load exp10 results
    model_tag = args.model.replace("/", "__")
    results_path = args.results_dir / f"{model_tag}_results.jsonl"

    if not results_path.exists():
        raise FileNotFoundError(f"Results not found at {results_path}")

    print(f"Loading results from {results_path}...")
    records = load_exp10_results(results_path)
    print(f"Loaded {len(records)} items")

    # Analyze requested tags
    tags = normalize_tags(args.tags)

    for tag in tags:
        print(f"\nProcessing tag: {tag}")

        # Extract margins
        items = extract_margins_per_tag(records, tag)
        print(f"Extracted {len(items)} items with complete data")

        if len(items) == 0:
            print(f"Warning: No items found for tag {tag}")
            continue

        # Run primary analysis (sign-consistent mask)
        print(f"Running bootstrap analysis (sign-consistent mask)...")
        metrics = bootstrap_all_metrics_coherent(
            items,
            n_boot=args.n_boot,
            use_sign_consistent_mask=True,
            tau=args.tau,
        )

        # Run robustness analysis (absolute mask)
        print(f"Running bootstrap analysis (absolute mask)...")
        metrics_abs = bootstrap_all_metrics_coherent(
            items,
            n_boot=args.n_boot,
            use_sign_consistent_mask=False,
            tau=args.tau,
        )

        # Run τ-sensitivity analysis
        print(f"Running τ-sensitivity analysis...")
        tau_sensitivity = run_tau_sensitivity_with_ci(
            items,
            tau_values=TAU_VALUES,
            n_boot=5000,  # Use fewer bootstrap samples for sensitivity
            use_sign_consistent_mask=True,
        )

        # Print report
        print_report(tag, args.model, metrics, tau_sensitivity, metrics_abs)

        # Save results to JSON
        output_file = args.output_dir / f"{model_tag}_{tag}_metrics.json"
        with output_file.open("w") as f:
            json.dump({
                "model": args.model,
                "tag": tag,
                "n_items": len(items),
                "tau_default": args.tau,
                "metrics_sign_consistent": metrics,
                "metrics_absolute": metrics_abs,
                "tau_sensitivity": tau_sensitivity,
            }, f, indent=2)
        print(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()
