"""Part B: Inverted-Prior Test (CRITICAL).

Distinguishes truth-tracking from prior-consistency control by analyzing items
where m_N0 < 0 (model's neutral prediction is WRONG).

On these items:
- Correct endorsement CONFLICTS with model's prior
- Wrong endorsement AGREES with model's prior

If truth-tracking: r_w stays positive in high-confidence-wrong slice
If prior-consistency control: r_w collapses toward zero or negative
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.exp11.logit_metrics import (
    ItemMetrics,
    TAU_DEFAULT,
    TAU_VALUES,
    bootstrap_all_metrics_coherent,
    compute_endorsement_susceptibility,
    compute_slice_m_N0_stats,
    run_tau_sensitivity_with_ci,
)
from src.exp11.analyze_logit_space import (
    extract_margins_per_tag,
    format_metric_with_ci,
    load_exp10_results,
    normalize_tags,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp11 Part B: Inverted-Prior Test")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("new-phase-results/exp10"),
        help="Directory containing exp10 JSONL results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("new-phase-results/exp11/part_b"),
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


def create_inverted_prior_slices(items: List[ItemMetrics]) -> Dict[str, List[ItemMetrics]]:
    """Create multiple slices of inverted-prior items.

    Slices:
    1. Full inverted-prior set (m_N0 < 0)
    2. Low-confidence wrong (bottom 50% of |m_N0|)
    3. High-confidence wrong (top 25% of |m_N0|) - CRITICAL discriminant
    4. Top-10% confidence wrong (maximum discriminant power)

    Args:
        items: All items

    Returns:
        Dict mapping slice name to list of items
    """
    # Filter to inverted-prior set
    inverted_items = [item for item in items if item.m_N0 < 0]

    if len(inverted_items) == 0:
        return {}

    # Compute |m_N0| for sorting
    abs_m_N0 = np.array([abs(item.m_N0) for item in inverted_items])

    # Sort by |m_N0| (ascending = low confidence first)
    sorted_indices = np.argsort(abs_m_N0)

    # Create slices
    n = len(inverted_items)
    slices = {}

    # Slice 1: Full inverted-prior set
    slices["full"] = inverted_items

    # Slice 2: Low-confidence wrong (bottom 50%)
    low_conf_indices = sorted_indices[:n // 2]
    slices["low_conf"] = [inverted_items[i] for i in low_conf_indices]

    # Slice 3: High-confidence wrong (top 25%) - CRITICAL
    high_conf_indices = sorted_indices[int(0.75 * n):]
    slices["high_conf_top25"] = [inverted_items[i] for i in high_conf_indices]

    # Slice 4: Top-10% confidence wrong (if n allows)
    if n >= 10:
        top10_indices = sorted_indices[int(0.9 * n):]
        slices["top10"] = [inverted_items[i] for i in top10_indices]

    return slices


def print_slice_report(
    slice_name: str,
    slice_items: List[ItemMetrics],
    metrics: Dict,
    metrics_abs: Dict,
    m_N0_stats: Dict,
    susceptibility: Dict,
    tau: float,
) -> None:
    """Print report for a single slice."""
    print(f"\n{'='*80}")
    print(f"Slice: {slice_name}")
    print(f"{'='*80}\n")

    # m_N0 distribution
    print(f"m_N0 Distribution:")
    print(f"  n: {m_N0_stats['n']}")
    print(f"  mean: {m_N0_stats['mean']:.4f}, median: {m_N0_stats['median']:.4f}")
    print(f"  range: [{m_N0_stats['min']:.4f}, {m_N0_stats['max']:.4f}]\n")

    print("Primary Additive Metrics (recommended):")
    print(f"  eff_diff = efficacy_w - efficacy_c:")
    print(f"    mean: {format_metric_with_ci(metrics['eff_diff_mean'])}")
    print(f"    median: {format_metric_with_ci(metrics['eff_diff_median'])}")
    print(f"    trimmed_mean(10%): {format_metric_with_ci(metrics['eff_diff_trimmed_mean'])}")
    print(f"  frac(efficacy_w > efficacy_c): {format_metric_with_ci(metrics['frac_eff_w_gt_c'])}")
    print(f"  baseline_shift (m_N1 - m_N0): {format_metric_with_ci(metrics['baseline_shift_mean'])}\n")

    # Endorsement susceptibility
    print(f"Endorsement Susceptibility (τ={tau}):")
    print(f"  p(effect_c_I0 > τ): {susceptibility['p_effect_c_works']:.4f} (WHY intersection shrinks)")
    print(f"  effect_c_I0: mean={susceptibility['effect_c_I0_mean']:.4f}, median={susceptibility['effect_c_I0_median']:.4f}")
    print(f"              q25={susceptibility['effect_c_I0_q25']:.4f}, q75={susceptibility['effect_c_I0_q75']:.4f}\n")

    # Sign-consistent mask results
    print(f"Sign-Consistent Mask (τ={tau}):")
    print(f"  n_total: {metrics['n_total']}\n")

    print(f"  Layer 1 (r_w, PRIMARY for inverted-prior):")
    print(f"    n_mask_valid_w: {metrics['n_mask_valid_w']}, pct: {format_metric_with_ci(metrics['pct_mask_valid_w'])}")
    print(f"    n_finite_w: {metrics['n_finite_w']}, pct_ratio_nonfinite: {format_metric_with_ci(metrics['pct_ratio_nonfinite_given_valid_w'])}")
    print(f"    pct_boot_zero_mask_w: {metrics['pct_boot_zero_mask_w']:.2f}%, pct_boot_zero_finite_w: {metrics['pct_boot_zero_finite_w']:.2f}%")
    if "r_w_mean" in metrics:
        print(f"    r_w: mean {format_metric_with_ci(metrics['r_w_mean'])}, median {format_metric_with_ci(metrics['r_w_median'])}")
        print(f"    % outside [0,1]: {format_metric_with_ci(metrics['pct_r_w_outside'])}\n")

    print(f"  Layer 2 (r_c):")
    print(f"    n_mask_valid_c: {metrics['n_mask_valid_c']}, pct: {format_metric_with_ci(metrics['pct_mask_valid_c'])}")
    print(f"    n_finite_c: {metrics['n_finite_c']}, pct_ratio_nonfinite: {format_metric_with_ci(metrics['pct_ratio_nonfinite_given_valid_c'])}")
    print(f"    pct_boot_zero_mask_c: {metrics['pct_boot_zero_mask_c']:.2f}%, pct_boot_zero_finite_c: {metrics['pct_boot_zero_finite_c']:.2f}%")
    if "r_c_mean" in metrics:
        print(f"    r_c: mean {format_metric_with_ci(metrics['r_c_mean'])}, median {format_metric_with_ci(metrics['r_c_median'])}")
        print(f"    % outside [0,1]: {format_metric_with_ci(metrics['pct_r_c_outside'])}\n")

    print(f"  Layer 3 (dr, SECONDARY):")
    print(f"    n_mask_valid_both: {metrics['n_mask_valid_both']}, pct: {format_metric_with_ci(metrics['pct_mask_valid_both'])} (shows shrinkage)")
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
    print(f"    baseline shift: {format_metric_with_ci(metrics['baseline_shift_mean'])}\n")

    # Robustness (absolute mask)
    print(f"Robustness (Absolute Mask):")
    print(f"  n_mask_valid_abs: {metrics_abs['n_mask_valid_abs']}, pct: {format_metric_with_ci(metrics_abs['pct_mask_valid_abs'])}")
    print(f"  n_finite_abs: {metrics_abs['n_finite_abs']}, pct_ratio_nonfinite: {format_metric_with_ci(metrics_abs['pct_ratio_nonfinite_given_valid_abs'])}")
    if "norm_sel_mean" in metrics_abs:
        print(f"  norm_sel (magnitude): mean {format_metric_with_ci(metrics_abs['norm_sel_mean'])}, median {format_metric_with_ci(metrics_abs['norm_sel_median'])}\n")


def print_summary_report(
    tag: str,
    model: str,
    slice_results: Dict[str, Dict],
) -> None:
    """Print summary report with key findings."""
    print(f"\n{'='*80}")
    print(f"SUMMARY: Inverted-Prior Test")
    print(f"Model: {model}, Tag: {tag}")
    print(f"{'='*80}\n")

    # High-confidence-wrong slice is the critical discriminant
    if "high_conf_top25" in slice_results:
        hc = slice_results["high_conf_top25"]
        metrics = hc["metrics"]
        m_N0_stats = hc["m_N0_stats"]

        print(f"CRITICAL SLICE: High-Confidence-Wrong (top-25%)")
        print(f"  n: {m_N0_stats['n']}")
        print(f"  m_N0: mean={m_N0_stats['mean']:.4f}, median={m_N0_stats['median']:.4f} (confirms strongly negative)\n")

        print(f"PRIMARY EVIDENCE (Layer 1: r_w):")
        if "r_w_mean" in metrics:
            r_w_mean = metrics["r_w_mean"][0]
            r_w_ci = metrics["r_w_mean"][1]
            print(f"  r_w = {format_metric_with_ci(metrics['r_w_mean'])}")
            print(f"  n_mask_valid_w: {metrics['n_mask_valid_w']}, n_finite_w: {metrics['n_finite_w']}\n")

            # Interpretation
            if r_w_mean > 0 and r_w_ci[0] > 0:
                print(f"INTERPRETATION: TRUTH-TRACKING")
                print(f"  r_w is positive even in high-confidence-wrong slice.")
                print(f"  Instruction still suppresses wrong endorsements when they agree with model's (wrong) prior.")
            elif r_w_mean < 0 or r_w_ci[1] < 0:
                print(f"INTERPRETATION: PRIOR-CONSISTENCY CONTROL")
                print(f"  r_w collapses toward/below zero in high-confidence-wrong slice.")
                print(f"  Instruction fails to suppress wrong endorsements when they agree with model's confident prior.")
            else:
                print(f"INTERPRETATION: AMBIGUOUS")
                print(f"  r_w CI includes zero. Results not conclusive.")
        else:
            print(f"  WARNING: No valid r_w data (n_finite_w={metrics['n_finite_w']})\n")

        print(f"SECONDARY EVIDENCE (Layer 3: dr):")
        if "dr_mean" in metrics:
            print(f"  dr = {format_metric_with_ci(metrics['dr_mean'])}")
            print(f"  n_mask_valid_both: {metrics['n_mask_valid_both']}, n_finite_both: {metrics['n_finite_both']} (aggregates on isfinite(dr))")
            print(f"  frac(dr > 0): {format_metric_with_ci(metrics['frac_dr_pos'])}\n")
        else:
            print(f"  WARNING: No valid dr data (n_finite_both={metrics['n_finite_both']})\n")

    print(f"{'='*80}\n")


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
        print(f"\n{'='*80}")
        print(f"Processing tag: {tag}")
        print(f"{'='*80}\n")

        # Extract margins
        items = extract_margins_per_tag(records, tag)
        print(f"Extracted {len(items)} items with complete data")

        if len(items) == 0:
            print(f"Warning: No items found for tag {tag}")
            continue

        # Create inverted-prior slices
        print(f"Creating inverted-prior slices...")
        slices = create_inverted_prior_slices(items)

        if not slices:
            print(f"Warning: No inverted-prior items found (no items with m_N0 < 0)")
            continue

        print(f"Created {len(slices)} slices:")
        for slice_name, slice_items in slices.items():
            print(f"  {slice_name}: {len(slice_items)} items")

        # Analyze each slice
        slice_results = {}

        for slice_name, slice_items in slices.items():
            print(f"\nAnalyzing slice: {slice_name}...")

            # Compute m_N0 stats
            m_N0_stats = compute_slice_m_N0_stats(slice_items)

            # Compute endorsement susceptibility
            susceptibility = compute_endorsement_susceptibility(slice_items, tau=args.tau)

            # Run primary analysis (sign-consistent mask)
            metrics = bootstrap_all_metrics_coherent(
                slice_items,
                n_boot=args.n_boot,
                use_sign_consistent_mask=True,
                tau=args.tau,
            )

            # Run robustness analysis (absolute mask)
            metrics_abs = bootstrap_all_metrics_coherent(
                slice_items,
                n_boot=args.n_boot,
                use_sign_consistent_mask=False,
                tau=args.tau,
            )

            # Print report for this slice
            print_slice_report(
                slice_name,
                slice_items,
                metrics,
                metrics_abs,
                m_N0_stats,
                susceptibility,
                args.tau,
            )

            # Store results
            slice_results[slice_name] = {
                "n_items": len(slice_items),
                "m_N0_stats": m_N0_stats,
                "susceptibility": susceptibility,
                "metrics": metrics,
                "metrics_abs": metrics_abs,
            }

        # Print summary report
        print_summary_report(tag, args.model, slice_results)

        # Run τ-sensitivity on high-confidence-wrong slice (if it exists)
        if "high_conf_top25" in slices:
            print(f"\nRunning τ-sensitivity analysis on high-confidence-wrong slice...")
            tau_sensitivity = run_tau_sensitivity_with_ci(
                slices["high_conf_top25"],
                tau_values=TAU_VALUES,
                n_boot=5000,
                use_sign_consistent_mask=True,
            )

            print(f"\nτ-Sensitivity Table (High-Conf Slice):")
            print(f"  {'τ':<8} {'n_mask_w':<12} {'n_finite_w':<12} {'n_mask_c':<12} {'n_finite_c':<12} {'n_mask_both':<15} {'n_finite_both':<15} {'r_w mean [CI]':<35} {'dr mean [CI]':<35}")
            print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*15} {'-'*15} {'-'*35} {'-'*35}")

            for tau_key in sorted(tau_sensitivity.keys()):
                tau_results = tau_sensitivity[tau_key]
                tau = tau_results["tau"]
                print(f"  {tau:<8.0e} {tau_results['n_mask_valid_w']:<12} {tau_results['n_finite_w']:<12} {tau_results['n_mask_valid_c']:<12} {tau_results['n_finite_c']:<12} {tau_results['n_mask_valid_both']:<15} {tau_results['n_finite_both']:<15} {format_metric_with_ci(tau_results['r_w_mean']):<35} {format_metric_with_ci(tau_results['dr_mean']):<35}")

            slice_results["high_conf_top25"]["tau_sensitivity"] = tau_sensitivity

        # Save results to JSON
        output_file = args.output_dir / f"{model_tag}_{tag}_inverted_prior.json"
        with output_file.open("w") as f:
            json.dump({
                "model": args.model,
                "tag": tag,
                "n_items_total": len(items),
                "tau_default": args.tau,
                "slices": slice_results,
            }, f, indent=2)
        print(f"\nSaved results to {output_file}")


if __name__ == "__main__":
    main()
