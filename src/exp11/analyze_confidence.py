"""Part C: Confidence-Binned Analysis (Descriptive).

Explores whether selectivity varies with model confidence |m_N0|.
This is descriptive - both hypotheses can produce confidence-dependent patterns.
Do NOT use Part C alone to distinguish truth-tracking vs prior-consistency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.exp11.logit_metrics import (
    ItemMetrics,
    TAU_DEFAULT,
    bootstrap_all_metrics_coherent,
)
from src.exp11.analyze_logit_space import (
    extract_margins_per_tag,
    format_metric_with_ci,
    load_exp10_results,
    normalize_tags,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp11 Part C: Confidence-Binned Analysis")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("new-phase-results/exp10"),
        help="Directory containing exp10 JSONL results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("new-phase-results/exp11/part_c"),
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


def create_confidence_bins(items: List[ItemMetrics], n_bins: int = 4) -> Dict[str, List[ItemMetrics]]:
    """Bin items by |m_N0| quartiles.

    Args:
        items: All items
        n_bins: Number of bins (default: 4 for quartiles)

    Returns:
        Dict mapping bin name to list of items
    """
    abs_m_N0 = np.array([abs(item.m_N0) for item in items])

    # Compute bin edges (quartiles)
    bin_edges = np.percentile(abs_m_N0, np.linspace(0, 100, n_bins + 1))

    # Assign items to bins
    bins = {}
    for i in range(n_bins):
        bin_name = f"Q{i+1}"
        if i == 0:
            # First bin: <= Q1
            bin_items = [item for item, m in zip(items, abs_m_N0) if m <= bin_edges[i + 1]]
        elif i == n_bins - 1:
            # Last bin: > Q3
            bin_items = [item for item, m in zip(items, abs_m_N0) if m > bin_edges[i]]
        else:
            # Middle bins: (Q_i, Q_{i+1}]
            bin_items = [item for item, m in zip(items, abs_m_N0) if bin_edges[i] < m <= bin_edges[i + 1]]

        bins[bin_name] = bin_items

    return bins


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
        print(f"Part C: Confidence-Binned Analysis")
        print(f"Model: {args.model}, Tag: {tag}")
        print(f"{'='*80}\n")

        # Extract margins
        items = extract_margins_per_tag(records, tag)
        print(f"Extracted {len(items)} items with complete data")

        if len(items) == 0:
            print(f"Warning: No items found for tag {tag}")
            continue

        # Create confidence bins
        print(f"Creating confidence bins (quartiles by |m_N0|)...")
        bins = create_confidence_bins(items, n_bins=4)

        print(f"Created {len(bins)} bins:")
        for bin_name, bin_items in bins.items():
            print(f"  {bin_name}: {len(bin_items)} items")

        # Analyze each bin
        bin_results = {}

        for bin_name, bin_items in bins.items():
            print(f"\n{'-'*80}")
            print(f"Bin: {bin_name} (n={len(bin_items)})")
            print(f"{'-'*80}")

            # Compute |m_N0| range for this bin
            abs_m_N0 = np.array([abs(item.m_N0) for item in bin_items])
            print(f"|m_N0| range: [{np.min(abs_m_N0):.4f}, {np.max(abs_m_N0):.4f}]")
            print(f"|m_N0| mean: {np.mean(abs_m_N0):.4f}, median: {np.median(abs_m_N0):.4f}\n")

            # Run bootstrap analysis
            metrics = bootstrap_all_metrics_coherent(
                bin_items,
                n_boot=args.n_boot,
                use_sign_consistent_mask=True,
                tau=args.tau,
            )

            # Print key metrics
            print(f"dr (differential suppression):")
            if "dr_mean" in metrics:
                print(f"  mean: {format_metric_with_ci(metrics['dr_mean'])}")
                print(f"  median: {format_metric_with_ci(metrics['dr_median'])}")
                print(f"  frac(dr > 0): {format_metric_with_ci(metrics['frac_dr_pos'])}")
            else:
                print(f"  No valid data (n_finite_both={metrics['n_finite_both']})")

            print(f"\nMask coverage:")
            print(f"  n_mask_valid_both: {metrics['n_mask_valid_both']}")
            print(f"  n_finite_both: {metrics['n_finite_both']}")

            print(f"\nUnconditional:")
            print(f"  frac(eff_w > eff_c): {format_metric_with_ci(metrics['frac_eff_w_gt_c'])}")

            # Store results
            bin_results[bin_name] = {
                "n_items": len(bin_items),
                "abs_m_N0_min": float(np.min(abs_m_N0)),
                "abs_m_N0_max": float(np.max(abs_m_N0)),
                "abs_m_N0_mean": float(np.mean(abs_m_N0)),
                "abs_m_N0_median": float(np.median(abs_m_N0)),
                "metrics": metrics,
            }

        # Print summary
        print(f"\n{'='*80}")
        print(f"SUMMARY: Confidence Pattern")
        print(f"{'='*80}\n")

        print(f"{'Bin':<8} {'|m_N0| median':<15} {'dr mean':<20} {'frac(dr > 0)':<20}")
        print(f"{'-'*8} {'-'*15} {'-'*20} {'-'*20}")

        for bin_name in ["Q1", "Q2", "Q3", "Q4"]:
            if bin_name in bin_results:
                br = bin_results[bin_name]
                abs_m_N0_median = br["abs_m_N0_median"]
                if "dr_mean" in br["metrics"]:
                    dr_mean = br["metrics"]["dr_mean"][0]
                    frac_dr_pos = br["metrics"]["frac_dr_pos"][0]
                    print(f"{bin_name:<8} {abs_m_N0_median:<15.4f} {dr_mean:<20.4f} {frac_dr_pos:<20.4f}")
                else:
                    print(f"{bin_name:<8} {abs_m_N0_median:<15.4f} {'N/A':<20} {'N/A':<20}")

        print(f"\nNOTE: This analysis is DESCRIPTIVE. Both truth-tracking and prior-consistency")
        print(f"      can produce confidence-dependent patterns. Use Part B (inverted-prior)")
        print(f"      for discriminating between hypotheses.\n")

        # Save results to JSON
        output_file = args.output_dir / f"{model_tag}_{tag}_confidence.json"
        with output_file.open("w") as f:
            json.dump({
                "model": args.model,
                "tag": tag,
                "n_items_total": len(items),
                "tau_default": args.tau,
                "bins": bin_results,
            }, f, indent=2)
        print(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()
