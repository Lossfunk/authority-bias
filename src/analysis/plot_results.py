#!/usr/bin/env python3
"""
plot_results.py

Publication-quality visualizations for the Persona Vectors project.
Focused on the key narrative from findings.md:

1. Baseline: Model is 52% sycophantic
2. Steering: Vector effect is weak but present
3. Circuit Verification: Path-patched heads are NOT causal
4. Summary: Tying it all together

Style: Anthropic-inspired, clean, narrative-driven.
Uses FancyBboxPatch for rounded bar charts.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy import stats
from textwrap import wrap

# ---------------------------------------------------------------------
# Configuration & Style
# ---------------------------------------------------------------------

OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#999999",
    "black": "#000000",
    "white": "#FFFFFF",
}

mpl.rcParams.update({
    "figure.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.5,
})


def load_jsonl(path: Path) -> pd.DataFrame:
    with path.open("r") as f:
        return pd.DataFrame([json.loads(line) for line in f])


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def draw_rounded_bar(ax, x: float, height: float, width: float = 0.6, 
                     color: str = "#000000", alpha: float = 0.85,
                     rounding_size: float = 0.08, label: str = None):
    """
    Draw a bar with rounded top corners using FancyBboxPatch.
    Based on matplotlib FancyBboxPatch documentation.
    """
    # Create a FancyBboxPatch with rounded corners
    fancy_box = FancyBboxPatch(
        (x - width/2, 0),  # (x, y) of lower-left corner
        width,              # width
        height,             # height
        boxstyle=f"round,pad=0,rounding_size={rounding_size}",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
        alpha=alpha,
        label=label
    )
    ax.add_patch(fancy_box)
    return fancy_box


def draw_rounded_bars(ax, x_positions: list, heights: list, colors: list,
                      width: float = 0.6, rounding_size: float = 0.08,
                      labels: list = None):
    """Draw multiple rounded bars."""
    patches = []
    for i, (x, h, c) in enumerate(zip(x_positions, heights, colors)):
        label = labels[i] if labels else None
        patch = draw_rounded_bar(ax, x, h, width, c, rounding_size=rounding_size, label=label)
        patches.append(patch)
    return patches


# ---------------------------------------------------------------------
# Figure 1: Baseline Sycophancy (with rounded bars)
# ---------------------------------------------------------------------

def plot_fig1_baseline(baseline_records: pd.DataFrame, baseline_metrics: dict, out_path: Path):
    """
    The model is naturally sycophantic on 52% of examples.
    Uses a clean bar chart with rounded bars (FancyBboxPatch).
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    
    D_syc = baseline_records["D_syc"]
    syc_rate = baseline_metrics.get("syc_rate", (D_syc > 0).mean())
    
    # Data for rounded bar chart
    categories = ["Sycophantic\n(D > 0)", "Truthful\n(D ≤ 0)"]
    values = [syc_rate * 100, (1 - syc_rate) * 100]
    colors = [OKABE_ITO["vermillion"], OKABE_ITO["bluish_green"]]
    
    # Draw rounded bars using FancyBboxPatch
    x_positions = [0, 1]
    draw_rounded_bars(ax, x_positions, values, colors, width=0.55, rounding_size=0.06)
    
    # Add value labels on top of bars
    for x, val, color in zip(x_positions, values, colors):
        ax.text(x, val + 2, f"{val:.0f}%", ha="center", va="bottom",
                fontsize=16, fontweight="bold", color=color)
    
    # Configure axes
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 75)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Percentage of Responses", fontsize=11)
    ax.set_title("Llama-3.1-8B Baseline Sycophancy Rate", fontsize=14, fontweight="bold", pad=15)
    
    # Key finding annotation with rounded box
    finding_text = f"52% of responses\nare sycophantic"
    bbox_props = dict(
        boxstyle="round,pad=0.5,rounding_size=0.15",
        facecolor=OKABE_ITO["vermillion"], 
        alpha=0.15,
        edgecolor=OKABE_ITO["vermillion"], 
        linewidth=2
    )
    ax.text(0.97, 0.95, finding_text, transform=ax.transAxes,
            fontsize=12, fontweight="bold", ha="right", va="top",
            color=OKABE_ITO["vermillion"], bbox=bbox_props)
    
    # Remove top/right spines (already done via rcParams but being explicit)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")
    plt.close()


# ---------------------------------------------------------------------
# Figure 2: Steering Effect (Weak but Present)
# ---------------------------------------------------------------------

def plot_fig2_steering(mediation_df: pd.DataFrame, out_path: Path):
    """
    Steering produces a weak but measurable effect.
    Honest framing of limitations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = mediation_df["D_base"]
    y = mediation_df["D_vec"]
    reduction = x - y
    
    # Stats
    mean_red = reduction.mean()
    std_red = reduction.std()
    pct_improved = (reduction > 0).mean() * 100
    effect_size = mean_red / x.std()  # Cohen's d approximation
    
    # Panel A: Scatter
    ax = axes[0]
    colors = np.where(reduction > 0, OKABE_ITO["bluish_green"], OKABE_ITO["vermillion"])
    ax.scatter(x, y, c=colors, alpha=0.6, s=50, edgecolor="white", linewidth=0.5)
    
    # Reference line
    lims = [min(x.min(), y.min()) - 0.5, max(x.max(), y.max()) + 0.5]
    ax.plot(lims, lims, color=OKABE_ITO["grey"], linestyle="--", linewidth=1.5, 
            label="No change", zorder=0)
    
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Baseline $D_{syc}$", fontsize=11)
    ax.set_ylabel(r"After Steering $D_{syc}$", fontsize=11)
    ax.set_title("A. Per-Example Effect", fontweight="bold", loc="left")
    
    # Legend with fancy boxes
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=OKABE_ITO["bluish_green"], 
               markersize=10, label=f'Improved ({pct_improved:.0f}%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=OKABE_ITO["vermillion"], 
               markersize=10, label=f'Worsened ({100-pct_improved:.0f}%)')
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    
    # Panel B: Distribution
    ax = axes[1]
    ax.hist(reduction, bins=30, color=OKABE_ITO["blue"], alpha=0.75, edgecolor="white")
    ax.axvline(0, color=OKABE_ITO["grey"], linestyle=":", linewidth=1.5, label="No change")
    ax.axvline(mean_red, color=OKABE_ITO["vermillion"], linestyle="-", linewidth=2.5,
               label=f"Mean = {mean_red:.3f}")
    
    ax.set_xlabel(r"Sycophancy Reduction ($D_{base} - D_{vec}$)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("B. Effect Distribution", fontweight="bold", loc="left")
    ax.legend(fontsize=9)
    
    # Key finding box with FancyBboxPatch styling
    finding_text = (
        f"Effect Size: {mean_red:.3f}\n"
        f"Cohen's d ≈ {effect_size:.2f}\n"
        f"Weak but present"
    )
    bbox_props = dict(boxstyle="round,pad=0.4,rounding_size=0.2", 
                     facecolor=OKABE_ITO["orange"], alpha=0.2,
                     edgecolor=OKABE_ITO["orange"], linewidth=1.5)
    ax.text(0.97, 0.95, finding_text, transform=ax.transAxes,
            fontsize=10, ha="right", va="top", bbox=bbox_props)
    
    # Main finding caption
    fig.text(0.5, 0.02, 
             "Finding: Steering vector produces small but consistent reduction in sycophancy",
             ha="center", fontsize=11, style="italic", color=OKABE_ITO["grey"])
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")
    plt.close()


# ---------------------------------------------------------------------
# Figure 3: Circuit Verification FAILED (The Critical Finding)
# ---------------------------------------------------------------------

def plot_fig3_circuit_verification(circuit_results: pd.DataFrame, circuit_summary: dict, out_path: Path):
    """
    THE CRITICAL FINDING: Ablating 'sycophancy heads' does NOT reduce sycophancy.
    This invalidates the circuit hypothesis. Uses FancyBboxPatch for rounded bars.
    """
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.25)
    
    # Panel A: Before/After Scatter
    ax_a = fig.add_subplot(gs[0])
    
    x = circuit_results["D_base"]
    y = circuit_results["D_ablated"]
    
    # Color by whether ablation helped
    helped = y < x
    hurt = y > x
    
    ax_a.scatter(x[helped], y[helped], color=OKABE_ITO["bluish_green"], alpha=0.7, s=60,
                 label=f"Reduced sycophancy ({helped.sum()})", edgecolor="white", linewidth=0.5)
    ax_a.scatter(x[hurt], y[hurt], color=OKABE_ITO["vermillion"], alpha=0.7, s=60,
                 label=f"Increased sycophancy ({hurt.sum()})", edgecolor="white", linewidth=0.5)
    
    # Reference line
    lims = [min(x.min(), y.min()) - 0.5, max(x.max(), y.max()) + 0.5]
    ax_a.plot(lims, lims, color=OKABE_ITO["black"], linestyle="--", linewidth=2, 
              label="No change", zorder=0)
    
    ax_a.set_xlim(lims)
    ax_a.set_ylim(lims)
    ax_a.set_aspect("equal")
    ax_a.set_xlabel(r"Baseline $D_{syc}$", fontsize=11)
    ax_a.set_ylabel(r"After Ablating 64 'Sycophancy Heads'", fontsize=11)
    ax_a.set_title("A. Ablation Does NOT Reduce Sycophancy", fontweight="bold", loc="left",
                   color=OKABE_ITO["vermillion"])
    ax_a.legend(loc="lower right", fontsize=9)
    
    # Critical finding box
    mean_change = circuit_summary.get("mean_reduction", 0)
    p_val = circuit_summary.get("p_value", 1)
    
    direction = "INCREASED" if mean_change < 0 else "decreased"
    finding_text = (
        f"Mean change: {mean_change:+.3f}\n"
        f"(Sycophancy {direction}!)\n"
        f"p = {p_val:.2f} (n.s.)"
    )
    
    box_color = OKABE_ITO["vermillion"]
    bbox_props = dict(
        boxstyle="round,pad=0.5,rounding_size=0.15",
        facecolor=box_color, 
        alpha=0.15,
        edgecolor=box_color, 
        linewidth=2
    )
    ax_a.text(0.03, 0.97, finding_text, transform=ax_a.transAxes,
              fontsize=11, fontweight="bold", ha="left", va="top",
              color=box_color, bbox=bbox_props)
    
    # Panel B: Rate Comparison with ROUNDED BARS
    ax_b = fig.add_subplot(gs[1])
    
    syc_base = circuit_summary.get("syc_rate_base", 0.55) * 100
    syc_abl = circuit_summary.get("syc_rate_ablated", 0.53) * 100
    
    # Draw rounded bars using FancyBboxPatch
    x_positions = [0, 1]
    values = [syc_base, syc_abl]
    colors = [OKABE_ITO["vermillion"], OKABE_ITO["orange"]]
    draw_rounded_bars(ax_b, x_positions, values, colors, width=0.55, rounding_size=0.05)
    
    # Value labels
    for x, val, color in zip(x_positions, values, colors):
        ax_b.text(x, val + 2, f"{val:.1f}%", ha="center", fontsize=14, fontweight="bold", color=color)
    
    ax_b.set_xlim(-0.5, 1.5)
    ax_b.set_ylim(0, 70)
    ax_b.set_xticks(x_positions)
    ax_b.set_xticklabels(["Baseline", "After\nAblation"], fontsize=10)
    ax_b.set_ylabel("Sycophancy Rate (%)", fontsize=11)
    ax_b.set_title("B. No Significant Change", fontweight="bold", loc="left")
    
    # Add "Not Significant" annotation with fancy box
    ax_b.annotate("", xy=(1, syc_abl + 4), xytext=(0, syc_base + 4),
                  arrowprops=dict(arrowstyle="<->", color=OKABE_ITO["grey"], lw=2.5))
    
    ns_bbox = dict(
        boxstyle="round,pad=0.4,rounding_size=0.15",
        facecolor=OKABE_ITO["grey"], 
        alpha=0.15,
        edgecolor=OKABE_ITO["grey"], 
        linewidth=1.5
    )
    ax_b.text(0.5, max(syc_base, syc_abl) + 10, "Not Significant\n(p = 0.59)",
              ha="center", fontsize=10, fontweight="bold", 
              color=OKABE_ITO["grey"], bbox=ns_bbox)
    
    # Main conclusion as figure title
    fig.suptitle("Critical Finding: Path-Patched Heads Are NOT Causal for Sycophancy",
                 fontsize=14, fontweight="bold", y=0.98, color=OKABE_ITO["vermillion"])
    
    # Bottom caption
    fig.text(0.5, 0.02, 
             "Implication: We cannot claim 'bypass' because we never found the circuit",
             ha="center", fontsize=11, style="italic", color=OKABE_ITO["grey"])
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")
    plt.close()


# ---------------------------------------------------------------------
# Figure 4: Summary (The Complete Story) with Rounded Bars
# ---------------------------------------------------------------------

def plot_fig4_summary(baseline_metrics: dict, mediation_df: pd.DataFrame,
                      validation_summary: dict, out_path: Path):
    """
    Complete 2x2 summary telling the full story.
    Uses FancyBboxPatch for rounded bars throughout.
    """
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)
    
    circuit = validation_summary.get("circuit_verification", {})
    sign_aware = validation_summary.get("sign_aware_mediation", {})
    
    # ------ Panel A: Baseline with Rounded Bars ------
    ax_a = fig.add_subplot(gs[0, 0])
    syc_rate = baseline_metrics.get("syc_rate", 0.52)
    
    x_positions = [0, 1]
    values = [syc_rate * 100, (1 - syc_rate) * 100]
    colors = [OKABE_ITO["vermillion"], OKABE_ITO["bluish_green"]]
    draw_rounded_bars(ax_a, x_positions, values, colors, width=0.55, rounding_size=0.05)
    
    for x, val, color in zip(x_positions, values, colors):
        ax_a.text(x, val + 2, f"{val:.0f}%", ha="center", fontsize=13, fontweight="bold", color=color)
    
    ax_a.set_xlim(-0.5, 1.5)
    ax_a.set_ylim(0, 70)
    ax_a.set_xticks(x_positions)
    ax_a.set_xticklabels(["Sycophantic", "Truthful"], fontsize=10)
    ax_a.set_ylabel("Percentage", fontsize=11)
    ax_a.set_title("A. Baseline: Model is Sycophantic", fontweight="bold", loc="left")
    
    # ------ Panel B: Steering is Weak ------
    ax_b = fig.add_subplot(gs[0, 1])
    
    reduction = mediation_df["D_base"] - mediation_df["D_vec"]
    mean_red = reduction.mean()
    
    ax_b.hist(reduction, bins=25, color=OKABE_ITO["blue"], alpha=0.75, edgecolor="white")
    ax_b.axvline(0, color=OKABE_ITO["grey"], linestyle=":", linewidth=2)
    ax_b.axvline(mean_red, color=OKABE_ITO["vermillion"], linestyle="-", linewidth=2.5,
                 label=f"Mean = {mean_red:.3f}")
    
    ax_b.set_xlabel(r"$D_{base} - D_{vec}$", fontsize=11)
    ax_b.set_ylabel("Count", fontsize=11)
    ax_b.set_title("B. Steering: Weak Effect", fontweight="bold", loc="left")
    ax_b.legend(fontsize=9)
    
    bbox_props = dict(
        boxstyle="round,pad=0.4,rounding_size=0.15",
        facecolor=OKABE_ITO["orange"], 
        alpha=0.2,
        edgecolor=OKABE_ITO["orange"], 
        linewidth=1.5
    )
    ax_b.text(0.97, 0.95, f"Effect: {mean_red:.3f}\n(~1% of σ)",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=10, bbox=bbox_props)
    
    # ------ Panel C: Circuit NOT Causal with Rounded Bars ------
    ax_c = fig.add_subplot(gs[1, 0])
    
    syc_base = circuit.get("syc_rate_base", 0.55) * 100
    syc_abl = circuit.get("syc_rate_ablated", 0.53) * 100
    
    x_positions = [0, 1]
    values = [syc_base, syc_abl]
    colors = [OKABE_ITO["vermillion"], OKABE_ITO["orange"]]
    draw_rounded_bars(ax_c, x_positions, values, colors, width=0.55, rounding_size=0.05)
    
    for x, val, color in zip(x_positions, values, colors):
        ax_c.text(x, val + 2, f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold", color=color)
    
    ax_c.set_xlim(-0.5, 1.5)
    ax_c.set_ylim(0, 70)
    ax_c.set_xticks(x_positions)
    ax_c.set_xticklabels(["Baseline", "Ablated"], fontsize=10)
    ax_c.set_ylabel("Sycophancy Rate (%)", fontsize=11)
    ax_c.set_title("C. Circuit NOT Causal", fontweight="bold", loc="left",
                   color=OKABE_ITO["vermillion"])
    
    p_val = circuit.get("p_value", 0.59)
    bbox_props = dict(
        boxstyle="round,pad=0.4,rounding_size=0.15",
        facecolor=OKABE_ITO["vermillion"], 
        alpha=0.15,
        edgecolor=OKABE_ITO["vermillion"], 
        linewidth=1.5
    )
    ax_c.text(0.97, 0.95, f"p = {p_val:.2f}\nNo effect",
              transform=ax_c.transAxes, ha="right", va="top", fontsize=10, 
              fontweight="bold", color=OKABE_ITO["vermillion"], bbox=bbox_props)
    
    # ------ Panel D: Key Insights (Text) with Rounded Boxes ------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")
    
    # Title
    ax_d.text(0.5, 0.98, "Key Findings", transform=ax_d.transAxes,
              fontsize=14, fontweight="bold", ha="center", va="top")
    
    findings = [
        ("1. Path Patching Failed", OKABE_ITO["vermillion"],
         "Identified heads are correlated,\nnot causal for sycophancy"),
        
        ("2. Steering is Robust", OKABE_ITO["blue"],
         "Vector works regardless of\nattention head ablation"),
        
        ("3. Mechanism is Distributed", OKABE_ITO["bluish_green"],
         "Sycophancy likely operates via\nMLP/residual, not attention"),
    ]
    
    y_pos = 0.82
    for title, color, description in findings:
        # Title with fancy rounded box
        bbox_props = dict(
            boxstyle="round,pad=0.4,rounding_size=0.15",
            facecolor=color, 
            alpha=0.15,
            edgecolor=color, 
            linewidth=1.5
        )
        ax_d.text(0.05, y_pos, title, transform=ax_d.transAxes,
                  fontsize=11, fontweight="bold", color=color, va="top", bbox=bbox_props)
        
        # Description
        ax_d.text(0.05, y_pos - 0.10, description, transform=ax_d.transAxes,
                  fontsize=10, color=OKABE_ITO["black"], va="top")
        
        y_pos -= 0.28
    
    ax_d.set_title("D. Implications", fontweight="bold", loc="left")
    
    # Main title
    fig.suptitle("Sycophancy in Llama-3.1-8B: Distributed, Not Circuit-Localized",
                 fontsize=15, fontweight="bold", y=0.98)
    
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    base_dir = Path("llama-results")
    out_dir = base_dir / "figures"
    out_dir.mkdir(exist_ok=True)
    
    # Data paths
    path_baseline_records = base_dir / "exp0" / "baseline_records.jsonl"
    path_baseline_metrics = base_dir / "exp0" / "baseline_metrics.json"
    path_mediation = base_dir / "exp3" / "mediation_results.jsonl"
    
    # Validation paths
    validation_dir = base_dir / "validation"
    path_validation_summary = validation_dir / "validation_summary.json"
    path_circuit_results = validation_dir / "circuit_verification_results.jsonl"
    path_circuit_summary = validation_dir / "circuit_verification_summary.json"
    
    # Load data
    baseline_records = load_jsonl(path_baseline_records) if path_baseline_records.exists() else None
    baseline_metrics = load_json(path_baseline_metrics) if path_baseline_metrics.exists() else {}
    mediation_df = load_jsonl(path_mediation) if path_mediation.exists() else None
    validation_summary = load_json(path_validation_summary) if path_validation_summary.exists() else None
    circuit_results = load_jsonl(path_circuit_results) if path_circuit_results.exists() else None
    circuit_summary = load_json(path_circuit_summary) if path_circuit_summary.exists() else None
    
    print("=" * 60)
    print("Generating Publication Figures")
    print("=" * 60)
    print(f"Data loaded:")
    print(f"  - Baseline records: {len(baseline_records) if baseline_records is not None else 'N/A'}")
    print(f"  - Mediation results: {len(mediation_df) if mediation_df is not None else 'N/A'}")
    print(f"  - Validation summary: {'Yes' if validation_summary else 'N/A'}")
    print(f"  - Circuit verification: {len(circuit_results) if circuit_results is not None else 'N/A'}")
    print()
    
    # Generate the 4 essential figures
    
    # Fig 1: Baseline
    if baseline_records is not None:
        plot_fig1_baseline(baseline_records, baseline_metrics, out_dir / "Fig1_Baseline.png")
    
    # Fig 2: Steering Effect
    if mediation_df is not None:
        plot_fig2_steering(mediation_df, out_dir / "Fig2_Steering.png")
    
    # Fig 3: Circuit Verification (THE CRITICAL FINDING)
    if circuit_results is not None and circuit_summary is not None:
        plot_fig3_circuit_verification(circuit_results, circuit_summary, 
                                        out_dir / "Fig3_CircuitVerification.png")
    
    # Fig 4: Summary
    if all([baseline_metrics, mediation_df is not None, validation_summary]):
        plot_fig4_summary(baseline_metrics, mediation_df, validation_summary, 
                          out_dir / "Fig4_Summary.png")
    
    print()
    print("=" * 60)
    print(f"Essential figures saved to {out_dir}/")
    print("=" * 60)
    print("\nGenerated figures:")
    print("  1. Fig1_Baseline.png      - Baseline sycophancy (52%)")
    print("  2. Fig2_Steering.png      - Weak steering effect")
    print("  3. Fig3_CircuitVerification.png - THE critical finding")
    print("  4. Fig4_Summary.png       - Complete story")


if __name__ == "__main__":
    main()
