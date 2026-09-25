"""Visualizations for Exp11: Inverted-Prior Test Results.

Creates three focused figures:
1. Decision figure: dr (selectivity) with CIs and sample sizes
2. Decomposition figure: r_w vs r_c to explain why dr flips
3. Reliability figure: mask coverage to show data quality per slice
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

# Style settings
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13,
    'font.family': 'sans-serif',
})

# Colors
EXPERT_COLOR = '#2E86AB'  # Blue
NOTE_COLOR = '#A23B72'    # Magenta/Pink
ZERO_LINE_COLOR = '#888888'
R_W_COLOR = '#E07A5F'     # Terracotta for r_w
R_C_COLOR = '#81B29A'     # Sage green for r_c


def load_results(results_dir: Path) -> Dict:
    """Load all exp11 results (auto-discovers model prefix from filenames)."""
    results = {}

    # Part A: Full dataset metrics
    part_a_dir = results_dir / "part_a"
    for tag in ["expert", "note"]:
        matches = list(part_a_dir.glob(f"*_{tag}_metrics.json"))
        if matches:
            with open(matches[0]) as f:
                results[f"part_a_{tag}"] = json.load(f)

    # Part B: Inverted-prior results
    part_b_dir = results_dir / "part_b"
    for tag in ["expert", "note"]:
        matches = list(part_b_dir.glob(f"*_{tag}_inverted_prior.json"))
        if matches:
            with open(matches[0]) as f:
                results[f"part_b_{tag}"] = json.load(f)

    return results


def extract_metric_with_ci(data: List) -> Tuple[float, float, float]:
    """Extract point estimate and CI from [mean, [low, high]] format."""
    mean = data[0]
    ci_low, ci_high = data[1]
    return mean, ci_low, ci_high


def get_slice_data(results: Dict, tag: str, slice_name: str, metric: str) -> Optional[Tuple[float, float, float, int]]:
    """Get metric value, CI, and sample size for a slice.

    Returns: (value, ci_low, ci_high, n_finite) or None if not available.
    """
    if slice_name == "full_dataset":
        # Use part_a data
        part_a = results.get(f"part_a_{tag}", {})
        metrics = part_a.get("metrics_sign_consistent", {})
        if metric in metrics:
            val, lo, hi = extract_metric_with_ci(metrics[metric])
            n_finite = metrics.get("n_finite_both", metrics.get("n_finite_w", "?"))
            return val, lo, hi, n_finite
    else:
        # Use part_b slices
        part_b = results.get(f"part_b_{tag}", {})
        slices_data = part_b.get("slices", {})
        if slice_name in slices_data:
            slice_metrics = slices_data[slice_name]["metrics"]
            if metric in slice_metrics:
                val, lo, hi = extract_metric_with_ci(slice_metrics[metric])
                n_finite = slice_metrics.get("n_finite_both", slice_metrics.get("n_finite_w", "?"))
                return val, lo, hi, n_finite
    return None


def create_decision_figure(results: Dict, output_dir: Path):
    """Create primary decision figure: dr (selectivity) with CIs and sample sizes.

    This is the headline figure showing the inverted-prior result.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Slice definitions with precise statistical labels
    slice_configs = [
        ("full_dataset", "All items"),
        ("low_conf", "Wrong prior\n(m_N0 < 0, bottom 50%)"),
        ("high_conf_top25", "Wrong prior\n(m_N0 < 0, top 25% |m_N0|)"),
    ]

    x_positions = np.arange(len(slice_configs))
    offset = 0.15

    for tag, color, x_off, marker in [
        ("expert", EXPERT_COLOR, -offset, 'o'),
        ("note", NOTE_COLOR, offset, 's'),
    ]:
        vals, ci_los, ci_his, ns = [], [], [], []

        for slice_name, _ in slice_configs:
            data = get_slice_data(results, tag, slice_name, "dr_median")
            if data:
                val, lo, hi, n = data
                vals.append(val)
                ci_los.append(val - lo)
                ci_his.append(hi - val)
                ns.append(n)
            else:
                vals.append(np.nan)
                ci_los.append(0)
                ci_his.append(0)
                ns.append(0)

        # Plot points with error bars
        x = x_positions + x_off
        ax.errorbar(x, vals, yerr=[ci_los, ci_his],
                   fmt=marker, color=color, markersize=8, capsize=4, capthick=1.5,
                   label=f"{tag.capitalize()}", linewidth=1.5, markeredgecolor='white', markeredgewidth=1)

        # Mark significant points (CI excludes 0)
        for i, (xi, val, lo, hi) in enumerate(zip(x, vals, ci_los, ci_his)):
            if not np.isnan(val):
                ci_low_val = val - lo
                ci_high_val = val + hi
                # If CI excludes zero (both bounds same sign), mark as significant
                if (ci_low_val > 0 and ci_high_val > 0) or (ci_low_val < 0 and ci_high_val < 0):
                    # Place star closer to the point (just above upper CI)
                    ax.annotate('*', (xi, ci_high_val + 0.02),
                               ha='center', va='bottom', fontsize=11, color=color, fontweight='bold')

    # Zero line (critical reference)
    ax.axhline(y=0, color=ZERO_LINE_COLOR, linestyle='-', linewidth=1.5, alpha=0.8)

    # Shaded regions for interpretation
    ylim = ax.get_ylim()
    ax.axhspan(0, max(ylim[1], 0.5), alpha=0.06, color='green')
    ax.axhspan(min(ylim[0], -1.0), 0, alpha=0.06, color='red')

    # Region labels (corrected: positive is up)
    ax.text(0.98, 0.95, 'dr > 0: suppresses wrong > correct', transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color='darkgreen', alpha=0.8)
    ax.text(0.98, 0.05, 'dr < 0: suppresses correct > wrong', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color='darkred', alpha=0.8)

    # Add sample size table below x-axis
    # Collect n values for each tag and slice
    sample_text = "n (Expert / Note): "
    for i, (slice_name, _) in enumerate(slice_configs):
        n_expert = get_slice_data(results, "expert", slice_name, "dr_median")
        n_note = get_slice_data(results, "note", slice_name, "dr_median")
        n_e = n_expert[3] if n_expert else "?"
        n_n = n_note[3] if n_note else "?"
        if i > 0:
            sample_text += " | "
        sample_text += f"{n_e}/{n_n}"

    ax.text(0.5, -0.18, sample_text, transform=ax.transAxes,
            ha='center', va='top', fontsize=9, color='gray')

    # Labels
    ax.set_ylabel('Selectivity (dr = r_w − r_c)\nMedian with 95% CI')
    ax.set_xlabel('')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, label in slice_configs])
    ax.legend(loc='lower left', framealpha=0.95, title='* = CI excludes 0')

    ax.set_title('Instruction Override Selectivity by Model Prior Accuracy\n'
                 'Llama-3.1-8B-Instruct', fontweight='bold')

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig1_selectivity.png", dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / "fig1_selectivity.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved decision figure to {output_dir}/fig1_selectivity.png")


def create_decomposition_figure(results: Dict, output_dir: Path):
    """Create decomposition figure: r_w vs r_c side-by-side.

    Shows WHY selectivity flips by comparing suppression of wrong vs correct endorsements.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    slice_configs = [
        ("full_dataset", "All"),
        ("low_conf", "Wrong\n(bottom 50%)"),
        ("high_conf_top25", "Wrong\n(top 25% |m_N0|)"),
    ]
    x_positions = np.arange(len(slice_configs))
    width = 0.35

    for ax, (tag, title) in zip(axes, [("expert", "Expert Tag"), ("note", "Note Tag")]):
        r_w_vals, r_w_los, r_w_his, r_w_ns = [], [], [], []
        r_c_vals, r_c_los, r_c_his, r_c_ns = [], [], [], []

        for slice_name, _ in slice_configs:
            # r_w
            data_w = get_slice_data(results, tag, slice_name, "r_w_median")
            if data_w:
                val, lo, hi, n = data_w
                r_w_vals.append(val)
                r_w_los.append(val - lo)
                r_w_his.append(hi - val)
                r_w_ns.append(n)
            else:
                r_w_vals.append(np.nan)
                r_w_los.append(0)
                r_w_his.append(0)
                r_w_ns.append(0)

            # r_c
            data_c = get_slice_data(results, tag, slice_name, "r_c_median")
            if data_c:
                val, lo, hi, n = data_c
                r_c_vals.append(val)
                r_c_los.append(val - lo)
                r_c_his.append(hi - val)
                r_c_ns.append(n)
            else:
                r_c_vals.append(np.nan)
                r_c_los.append(0)
                r_c_his.append(0)
                r_c_ns.append(0)

        # Plot bars
        x = x_positions
        bars_w = ax.bar(x - width/2, r_w_vals, width, label='r_w (wrong endorsement)',
                       color=R_W_COLOR, alpha=0.8)
        ax.errorbar(x - width/2, r_w_vals, yerr=[r_w_los, r_w_his],
                   fmt='none', color='black', capsize=3, capthick=1)

        bars_c = ax.bar(x + width/2, r_c_vals, width, label='r_c (correct endorsement)',
                       color=R_C_COLOR, alpha=0.8)
        ax.errorbar(x + width/2, r_c_vals, yerr=[r_c_los, r_c_his],
                   fmt='none', color='black', capsize=3, capthick=1)

        # Zero line
        ax.axhline(y=0, color=ZERO_LINE_COLOR, linestyle='--', linewidth=1)

        # Reference line at 1 (full suppression)
        ax.axhline(y=1, color=ZERO_LINE_COLOR, linestyle=':', linewidth=1, alpha=0.5)

        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in slice_configs])

        if ax == axes[0]:
            ax.set_ylabel('Relative suppression ratio (median)\n0 = no effect, 1 = full suppression')
            ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Decomposition: Suppression of Wrong vs Correct Endorsements\n'
                 'dr = r_w − r_c (selectivity flips when r_c > r_w)',
                 fontweight='bold', y=1.02)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig2_decomposition.png", dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / "fig2_decomposition.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved decomposition figure to {output_dir}/fig2_decomposition.png")


def create_reliability_figure(results: Dict, output_dir: Path):
    """Create reliability figure: mask coverage by slice.

    Shows data quality and prevents overinterpretation of noisy high-confidence bins.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    slice_configs = [
        ("full_dataset", "All"),
        ("low_conf", "Wrong\n(bottom 50%)"),
        ("high_conf_top25", "Wrong\n(top 25% |m_N0|)"),
    ]
    x_positions = np.arange(len(slice_configs))
    width = 0.25

    for ax, (tag, title) in zip(axes, [("expert", "Expert Tag"), ("note", "Note Tag")]):
        pct_w_vals, pct_c_vals, pct_both_vals = [], [], []
        n_totals = []

        for slice_name, _ in slice_configs:
            if slice_name == "full_dataset":
                part_a = results.get(f"part_a_{tag}", {})
                metrics = part_a.get("metrics_sign_consistent", {})
                if metrics:
                    pct_w = extract_metric_with_ci(metrics["pct_mask_valid_w"])[0] if "pct_mask_valid_w" in metrics else np.nan
                    pct_c = extract_metric_with_ci(metrics["pct_mask_valid_c"])[0] if "pct_mask_valid_c" in metrics else np.nan
                    pct_both = extract_metric_with_ci(metrics["pct_mask_valid_both"])[0] if "pct_mask_valid_both" in metrics else np.nan
                    n_total = metrics.get("n_total", 0)
                else:
                    pct_w, pct_c, pct_both, n_total = np.nan, np.nan, np.nan, 0
            else:
                part_b = results.get(f"part_b_{tag}", {})
                slices_data = part_b.get("slices", {})
                if slice_name in slices_data:
                    metrics = slices_data[slice_name]["metrics"]
                    pct_w = extract_metric_with_ci(metrics["pct_mask_valid_w"])[0] if "pct_mask_valid_w" in metrics else np.nan
                    pct_c = extract_metric_with_ci(metrics["pct_mask_valid_c"])[0] if "pct_mask_valid_c" in metrics else np.nan
                    pct_both = extract_metric_with_ci(metrics["pct_mask_valid_both"])[0] if "pct_mask_valid_both" in metrics else np.nan
                    n_total = metrics.get("n_total", 0)
                else:
                    pct_w, pct_c, pct_both, n_total = np.nan, np.nan, np.nan, 0

            pct_w_vals.append(pct_w)
            pct_c_vals.append(pct_c)
            pct_both_vals.append(pct_both)
            n_totals.append(n_total)

        x = x_positions
        bars_w = ax.bar(x - width, pct_w_vals, width, color=R_W_COLOR, alpha=0.7)
        bars_c = ax.bar(x, pct_c_vals, width, color=R_C_COLOR, alpha=0.7)
        bars_both = ax.bar(x + width, pct_both_vals, width, color='#4A4E69', alpha=0.7)

        # Add n_total annotations
        for i, (xi, n) in enumerate(zip(x, n_totals)):
            ax.annotate(f'n={n}', (xi, 5), ha='center', va='bottom', fontsize=9, color='black')

        # Add exact percentages for high-conf slice only (index 2)
        high_conf_idx = 2
        for bars, val, offset in [(bars_w, pct_w_vals[high_conf_idx], -width),
                                   (bars_c, pct_c_vals[high_conf_idx], 0),
                                   (bars_both, pct_both_vals[high_conf_idx], width)]:
            if not np.isnan(val):
                ax.annotate(f'{val:.0f}%', (x[high_conf_idx] + offset, val + 2),
                           ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in slice_configs])
        ax.set_ylim(0, 115)  # Increase to fit percentage labels

        if ax == axes[0]:
            ax.set_ylabel('% of items with valid ratio computation')

    # Add single legend below the figure
    handles = [
        plt.Rectangle((0,0), 1, 1, color=R_W_COLOR, alpha=0.7),
        plt.Rectangle((0,0), 1, 1, color=R_C_COLOR, alpha=0.7),
        plt.Rectangle((0,0), 1, 1, color='#4A4E69', alpha=0.7),
    ]
    labels = ['Valid for r_w', 'Valid for r_c', 'Valid for dr']
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.02), framealpha=0.9)

    fig.suptitle('Data Quality: Mask Coverage by Slice\n'
                 'Lower coverage in high-confidence slices = interpret with caution',
                 fontweight='bold', y=1.02)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig3_reliability.png", dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / "fig3_reliability.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved reliability figure to {output_dir}/fig3_reliability.png")


def create_compact_summary(results: Dict, output_dir: Path):
    """Create a compact 1x2 summary: selectivity + decomposition only.

    No interpretation text in figure - keep it clean.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    slice_configs = [
        ("full_dataset", "All"),
        ("low_conf", "Wrong\n(bot 50%)"),
        ("high_conf_top25", "Wrong\n(top 25%)"),
    ]
    x_positions = np.arange(len(slice_configs))

    # Panel A: Selectivity (dr)
    ax = axes[0]
    offset = 0.12

    for tag, color, x_off, marker in [
        ("expert", EXPERT_COLOR, -offset, 'o'),
        ("note", NOTE_COLOR, offset, 's'),
    ]:
        vals, ci_los, ci_his, ns = [], [], [], []

        for slice_name, _ in slice_configs:
            data = get_slice_data(results, tag, slice_name, "dr_median")
            if data:
                val, lo, hi, n = data
                vals.append(val)
                ci_los.append(val - lo)
                ci_his.append(hi - val)
                ns.append(n)
            else:
                vals.append(np.nan)
                ci_los.append(0)
                ci_his.append(0)
                ns.append(0)

        x = x_positions + x_off
        ax.errorbar(x, vals, yerr=[ci_los, ci_his],
                   fmt=marker, color=color, markersize=8, capsize=4, capthick=1.5,
                   label=f"{tag.capitalize()} (n={ns[-1]})", linewidth=1.5,
                   markeredgecolor='white', markeredgewidth=1)

    ax.axhline(y=0, color=ZERO_LINE_COLOR, linestyle='-', linewidth=1.5, alpha=0.8)
    ax.axhspan(0, 0.5, alpha=0.06, color='green')
    ax.axhspan(-1.2, 0, alpha=0.06, color='red')

    ax.set_ylabel('Selectivity: dr = r_w − r_c\n(median, 95% CI)')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, label in slice_configs])
    ax.legend(loc='upper left', fontsize=9, title='High-conf slice n')
    ax.set_title('A. Does instruction selectively suppress wrong advice?', fontweight='bold')
    # Corrected: positive (top) = dr > 0, negative (bottom) = dr < 0
    ax.text(0.98, 0.95, 'dr > 0: blocks wrong > correct', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color='darkgreen', alpha=0.8)
    ax.text(0.98, 0.05, 'dr < 0: blocks correct > wrong', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color='darkred', alpha=0.8)

    # Panel B: Decomposition for Note tag only (the interesting case)
    ax = axes[1]
    width = 0.35
    tag = "note"

    r_w_vals, r_w_los, r_w_his = [], [], []
    r_c_vals, r_c_los, r_c_his = [], [], []

    for slice_name, _ in slice_configs:
        data_w = get_slice_data(results, tag, slice_name, "r_w_median")
        if data_w:
            val, lo, hi, _ = data_w
            r_w_vals.append(val)
            r_w_los.append(val - lo)
            r_w_his.append(hi - val)
        else:
            r_w_vals.append(np.nan)
            r_w_los.append(0)
            r_w_his.append(0)

        data_c = get_slice_data(results, tag, slice_name, "r_c_median")
        if data_c:
            val, lo, hi, _ = data_c
            r_c_vals.append(val)
            r_c_los.append(val - lo)
            r_c_his.append(hi - val)
        else:
            r_c_vals.append(np.nan)
            r_c_los.append(0)
            r_c_his.append(0)

    x = x_positions
    ax.bar(x - width/2, r_w_vals, width, label='r_w (wrong)', color=R_W_COLOR, alpha=0.8)
    ax.errorbar(x - width/2, r_w_vals, yerr=[r_w_los, r_w_his], fmt='none', color='black', capsize=3)

    ax.bar(x + width/2, r_c_vals, width, label='r_c (correct)', color=R_C_COLOR, alpha=0.8)
    ax.errorbar(x + width/2, r_c_vals, yerr=[r_c_los, r_c_his], fmt='none', color='black', capsize=3)

    ax.axhline(y=0, color=ZERO_LINE_COLOR, linestyle='--', linewidth=1)
    ax.axhline(y=1, color=ZERO_LINE_COLOR, linestyle=':', linewidth=1, alpha=0.5)

    ax.set_ylabel('Relative suppression ratio\n(median, 95% CI)')
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in slice_configs])
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('B. Why selectivity flips (Note tag)', fontweight='bold')
    ax.set_ylim(-0.1, 1.1)

    fig.suptitle('Inverted-Prior Test: "Be Correct" Induces Prior-Consistency, Not Truth-Tracking\n'
                 'Llama-3.1-8B-Instruct', fontweight='bold', y=1.02)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "exp11_summary.png", dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / "exp11_summary.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved compact summary to {output_dir}/exp11_summary.png")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Exp11 results")
    parser.add_argument("--results-dir", type=Path, default=Path("new-phase-results/prior_analysis"),
                        help="Directory containing exp11 results")
    parser.add_argument("--output-dir", type=Path, default=Path("new-phase-results/prior_analysis"),
                        help="Directory to save figures")
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)

    if not results:
        print("No results found!")
        return

    print(f"Found results for: {list(results.keys())}")

    # Create focused figures
    print("\n1. Creating decision figure (selectivity)...")
    create_decision_figure(results, args.output_dir)

    print("2. Creating decomposition figure (r_w vs r_c)...")
    create_decomposition_figure(results, args.output_dir)

    print("3. Creating reliability figure (mask coverage)...")
    create_reliability_figure(results, args.output_dir)

    print("4. Creating compact summary...")
    create_compact_summary(results, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
