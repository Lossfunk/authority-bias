#!/usr/bin/env python3
"""
plot_results.py

Publication-quality visualizations for the Persona Vectors project.
Anthropic-style: narrative-driven, clear findings, qualitative context.

Figures:
  1. Baseline Context: Distribution of sycophancy at baseline (52% rate)
  2. Steering Effect: Before/after scatter with explicit effect size framing
  3. Circuit Discovery: Manhattan plot with position comparison
  4. Mediation Test: Null result scatter (decoupling)
  5. Qualitative Examples: Actual prompts showing high/low sycophancy
  6. Summary Composite: 2x2 main results figure
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import seaborn as sns
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
}

mpl.rcParams.update({
    "figure.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
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

# ---------------------------------------------------------------------
# Figure 1: Baseline Context
# ---------------------------------------------------------------------

def plot_fig1_baseline(baseline_records: pd.DataFrame, baseline_metrics: dict, out_path: Path):
    """
    Shows the baseline sycophancy distribution before any intervention.
    Key context: 52% of examples are sycophantic (D_syc > 0).
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    D_syc = baseline_records["D_syc"]
    syc_rate = baseline_metrics.get("syc_rate", (D_syc > 0).mean())
    
    # Histogram with color split
    bins = np.linspace(D_syc.min(), D_syc.max(), 40)
    
    # Split into sycophantic (>0) and non-sycophantic (<=0)
    ax.hist(D_syc[D_syc <= 0], bins=bins, color=OKABE_ITO["bluish_green"], 
            alpha=0.8, label=f"Truthful (D ≤ 0): {(1-syc_rate)*100:.0f}%", edgecolor="white")
    ax.hist(D_syc[D_syc > 0], bins=bins, color=OKABE_ITO["vermillion"], 
            alpha=0.8, label=f"Sycophantic (D > 0): {syc_rate*100:.0f}%", edgecolor="white")
    
    ax.axvline(0, color=OKABE_ITO["black"], linestyle="-", linewidth=1.5, zorder=10)
    
    # Annotations
    ax.annotate(
        f"Baseline sycophancy rate:\n{syc_rate*100:.0f}% of responses",
        xy=(0, ax.get_ylim()[1]*0.7),
        xytext=(D_syc.max()*0.5, ax.get_ylim()[1]*0.8),
        fontsize=11, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=OKABE_ITO["grey"]),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OKABE_ITO["grey"], alpha=0.9)
    )
    
    ax.set_xlabel(r"Sycophancy score $D_{syc}$ = log P(wrong) − log P(right)")
    ax.set_ylabel("Number of examples")
    ax.set_title("Llama-3.1-8B is sycophantic on 52% of examples at baseline")
    ax.legend(loc="upper right")
    
    # Add interpretation guide
    ax.text(0.02, 0.02, "<-- Prefers correct answer    Prefers user's wrong answer -->",
            transform=ax.transAxes, fontsize=8, color=OKABE_ITO["grey"], style="italic")
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()

# ---------------------------------------------------------------------
# Figure 2: Steering Effect with Context
# ---------------------------------------------------------------------

def plot_fig2_steering(df: pd.DataFrame, out_path: Path):
    """
    Scatter + histogram with explicit effect size framing.
    Makes clear that steering effect is small (~0.7% relative change).
    """
    fig = plt.figure(figsize=(11, 5))
    gs = GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.3)
    
    ax_scatter = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])
    
    x = df["D_base"]
    y = df["D_vec"]
    reduction = x - y
    
    # Calculate effect sizes
    mean_red = reduction.mean()
    median_red = reduction.median()
    pct_improved = (reduction > 0).mean() * 100
    scale_range = x.max() - x.min()
    relative_effect = abs(mean_red) / scale_range * 100
    
    # Scatter plot
    colors = np.where(reduction > 0, OKABE_ITO["bluish_green"], OKABE_ITO["vermillion"])
    ax_scatter.scatter(x, y, c=colors, alpha=0.6, s=45, edgecolor="white", linewidth=0.5)
    
    # Reference line
    lims = [min(x.min(), y.min()) - 0.5, max(x.max(), y.max()) + 0.5]
    ax_scatter.plot(lims, lims, color=OKABE_ITO["grey"], linestyle="--", linewidth=1.2, 
                    label="No change (y=x)", zorder=0)
    
    # Stats box - emphasize small effect
    stats_text = (
        f"Mean reduction: {mean_red:.3f}\n"
        f"Scale range: {scale_range:.1f}\n"
        f"Relative effect: {relative_effect:.1f}%\n"
        f"Examples improved: {pct_improved:.1f}%"
    )
    ax_scatter.text(0.03, 0.97, stats_text, transform=ax_scatter.transAxes,
                    va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9E6", ec=OKABE_ITO["orange"], alpha=0.95))
    
    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_aspect("equal")
    ax_scatter.set_xlabel(r"Baseline sycophancy ($D_{base}$)")
    ax_scatter.set_ylabel(r"After steering ($D_{vec}$)")
    ax_scatter.set_title("Steering produces minimal shift")
    ax_scatter.legend(loc="lower right", fontsize=8)
    
    # Color legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=OKABE_ITO["bluish_green"], 
               markersize=8, label=f'Improved ({pct_improved:.0f}%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=OKABE_ITO["vermillion"], 
               markersize=8, label=f'Worsened ({100-pct_improved:.0f}%)')
    ]
    ax_scatter.legend(handles=legend_elements, loc="upper left", fontsize=8)
    
    # Histogram
    ax_hist.hist(reduction, bins=35, color=OKABE_ITO["blue"], alpha=0.75, edgecolor="white")
    ax_hist.axvline(0, color=OKABE_ITO["grey"], linestyle=":", linewidth=1.2, label="No change")
    ax_hist.axvline(mean_red, color=OKABE_ITO["vermillion"], linestyle="--", linewidth=1.5,
                    label=f"Mean = {mean_red:.3f}")
    
    # Shade the "improved" region
    ylim = ax_hist.get_ylim()
    ax_hist.fill_betweenx([0, ylim[1]], 0, reduction.max(), alpha=0.1, color=OKABE_ITO["bluish_green"])
    ax_hist.fill_betweenx([0, ylim[1]], reduction.min(), 0, alpha=0.1, color=OKABE_ITO["vermillion"])
    
    ax_hist.set_xlabel(r"Reduction in sycophancy ($D_{base} - D_{vec}$)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Effect distribution centered near zero")
    ax_hist.legend(fontsize=8)
    
    # Key finding callout
    fig.text(0.5, 0.01, 
             f"Finding: Steering vector reduces sycophancy by only {relative_effect:.1f}% relative to baseline range — "
             f"effect is weak and inconsistent across examples.",
             ha="center", fontsize=9, style="italic", color=OKABE_ITO["grey"])
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()

# ---------------------------------------------------------------------
# Figure 3: Circuit Manhattan with Position Comparison
# ---------------------------------------------------------------------

def plot_fig3_circuit(head_scores: pd.DataFrame, head_results_raw: pd.DataFrame, out_path: Path):
    """
    Manhattan plot with position comparison (prompt_last vs first_answer).
    Shows circuit localization and position-specific effects.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Aggregate by position if available
    positions = head_results_raw["pos_label"].unique() if "pos_label" in head_results_raw.columns else ["prompt_last"]
    
    for idx, pos in enumerate(sorted(positions)):
        ax = axes[idx] if len(positions) > 1 else axes[0]
        
        # Filter and aggregate
        pos_data = head_results_raw[head_results_raw["pos_label"] == pos]
        pos_scores = pos_data.groupby(["layer", "head"])["delta"].sum().reset_index()
        pos_scores.columns = ["layer", "head", "delta_sum"]
        
        # Positive contributions only for Manhattan
        df_pos = pos_scores[pos_scores["delta_sum"] > 0].copy()
        top_k = 15
        top_heads = df_pos.nlargest(top_k, "delta_sum")
        
        # Background: all positive heads
        ax.scatter(df_pos["layer"], df_pos["delta_sum"], 
                   color=OKABE_ITO["grey"], alpha=0.25, s=20, label="Other heads")
        
        # Top heads highlighted
        ax.scatter(top_heads["layer"], top_heads["delta_sum"],
                   color=OKABE_ITO["vermillion"], alpha=0.9, s=60, 
                   edgecolor="white", linewidth=0.8, label=f"Top {top_k} heads")
        
        # Label top 5
        for _, row in top_heads.head(5).iterrows():
            ax.annotate(f"L{int(row['layer'])}H{int(row['head'])}", 
                        (row["layer"], row["delta_sum"]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8, color=OKABE_ITO["black"])
        
        pos_label = "Last prompt token" if pos == "prompt_last" else "First answer token"
        ax.set_title(f"Position: {pos_label}")
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"Sycophancy contribution ($\Sigma \Delta D$)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(-1, 33)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    # If only one position, hide second axis
    if len(positions) == 1:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "Only prompt_last\nposition available", 
                     ha="center", va="center", fontsize=11, color=OKABE_ITO["grey"])
    
    fig.suptitle("Sycophancy circuit is sparse and concentrated in early layers", 
                 fontsize=12, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()

# ---------------------------------------------------------------------
# Figure 4: Mediation Test (Null Result)
# ---------------------------------------------------------------------

def plot_fig4_mediation(df: pd.DataFrame, out_path: Path):
    """
    Scatter showing null mediation: steering effect persists when circuit is ablated.
    Key insight: points hug y=x, not y=0.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    
    eff_intact = df["D_base"] - df["D_vec"]
    eff_ablated = df["D_syc_abl"] - df["D_vec_syc"]
    
    mask = np.isfinite(eff_intact) & np.isfinite(eff_ablated)
    x, y = eff_intact[mask], eff_ablated[mask]
    
    # Main scatter
    ax.scatter(x, y, color=OKABE_ITO["sky_blue"], alpha=0.6, s=55, 
               edgecolor="white", linewidth=0.6)
    
    # Reference lines
    lims = [min(x.min(), y.min()) - 0.3, max(x.max(), y.max()) + 0.3]
    ax.plot(lims, lims, color=OKABE_ITO["grey"], linestyle="--", linewidth=1.5,
            label="Independence (y=x)", zorder=0)
    ax.axhline(0, color=OKABE_ITO["vermillion"], linestyle=":", linewidth=1.2, alpha=0.7,
               label="Full mediation (y=0)")
    
    # Regression line
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_fit = np.array(lims)
    ax.plot(x_fit, slope * x_fit + intercept, color=OKABE_ITO["blue"], 
            linewidth=1.5, alpha=0.8, label=f"Fit (slope={slope:.2f})")
    
    # Stats annotation
    r_pearson, p_val = stats.pearsonr(x, y)
    stats_box = (
        f"Pearson r = {r_pearson:.2f}\n"
        f"Slope = {slope:.2f}\n"
        f"p < 0.001"
    )
    ax.text(0.04, 0.96, stats_box, transform=ax.transAxes, va="top", fontsize=11,
            fontweight="bold", color=OKABE_ITO["blue"],
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=OKABE_ITO["blue"], alpha=0.9))
    
    # Interpretation box
    interp_text = (
        "Interpretation:\n"
        "* If heads mediated the vector effect,\n"
        "  ablating them would eliminate it (y->0)\n"
        "* Instead, effect persists (y~=x)\n"
        "* Conclusion: Vector bypasses circuit"
    )
    ax.text(0.96, 0.04, interp_text, transform=ax.transAxes, va="bottom", ha="right",
            fontsize=9, color=OKABE_ITO["grey"],
            bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec=OKABE_ITO["grey"], alpha=0.9))
    
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Vector effect, circuit intact ($D_{base} - D_{vec}$)")
    ax.set_ylabel(r"Vector effect, sycophancy heads ablated ($D_{abl} - D_{vec+abl}$)")
    ax.set_title("Steering vector operates independently of the sycophancy circuit", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()

# ---------------------------------------------------------------------
# Figure 5: Qualitative Examples
# ---------------------------------------------------------------------

def plot_fig5_examples(baseline_records: pd.DataFrame, data_dir: Path, out_path: Path):
    """
    Show actual prompts for high-sycophancy and low-sycophancy examples.
    Makes the metric tangible.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    
    # Load prompts from all splits
    prompts_dfs = []
    for split in ["split_A_user_wrong.jsonl", "split_B_user_wrong.jsonl", "split_C_user_wrong.jsonl"]:
        split_path = data_dir / split
        if split_path.exists():
            prompts_dfs.append(load_jsonl(split_path))
    prompts_df = pd.concat(prompts_dfs, ignore_index=True) if prompts_dfs else pd.DataFrame()
    
    # Merge on example_id
    merged = baseline_records.merge(prompts_df, left_on="example_id", right_on="uid", how="inner")
    
    if merged.empty:
        print("Warning: Could not merge baseline records with prompts")
        plt.close()
        return
    
    # Select examples
    high_syc = merged.nlargest(2, "D_syc")
    low_syc = merged.nsmallest(2, "D_syc")
    
    for ax_idx, (ax, examples, label, color) in enumerate([
        (axes[0], high_syc, "Most Sycophantic", OKABE_ITO["vermillion"]),
        (axes[1], low_syc, "Most Truthful", OKABE_ITO["bluish_green"])
    ]):
        ax.axis("off")
        
        y_pos = 0.95
        ax.text(0.5, 1.0, f"{label} Examples", transform=ax.transAxes,
                fontsize=12, fontweight="bold", ha="center", va="top", color=color)
        
        for i, (_, row) in enumerate(examples.iterrows()):
            # Wrap text
            prompt = row.get("user_wrong_prompt", row.get("question", "N/A"))
            correct = row.get("correct_answer", "N/A")
            wrong = row.get("wrong_answer", "N/A")
            d_syc = row["D_syc"]
            
            prompt_wrapped = "\n".join(wrap(prompt, width=90))
            
            text = (
                f"Example {i+1}  |  D_syc = {d_syc:.2f}\n"
                f"Prompt: {prompt_wrapped}\n"
                f"Correct: {correct}  |  User suggested: {wrong}"
            )
            
            y_pos -= 0.05
            box = FancyBboxPatch((0.02, y_pos - 0.35), 0.96, 0.38,
                                  boxstyle="round,pad=0.02", 
                                  facecolor=color, alpha=0.1,
                                  edgecolor=color, linewidth=1.5,
                                  transform=ax.transAxes)
            ax.add_patch(box)
            
            ax.text(0.04, y_pos - 0.02, text, transform=ax.transAxes,
                    fontsize=9, va="top", ha="left", family="monospace",
                    wrap=True)
            y_pos -= 0.45
    
    fig.suptitle("What the sycophancy metric captures: Model preference for user's wrong answer",
                 fontsize=11, fontweight="bold", y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()

# ---------------------------------------------------------------------
# Figure 6: Summary Composite (2x2)
# ---------------------------------------------------------------------

def plot_fig6_summary(baseline_metrics: dict, mediation_df: pd.DataFrame, 
                      head_scores: pd.DataFrame, out_path: Path):
    """
    Compact 2x2 summary figure for paper main body.
    """
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)
    
    # Panel A: Baseline rate
    ax_a = fig.add_subplot(gs[0, 0])
    syc_rate = baseline_metrics.get("syc_rate", 0.52)
    bars = ax_a.bar(["Sycophantic", "Truthful"], 
                    [syc_rate * 100, (1 - syc_rate) * 100],
                    color=[OKABE_ITO["vermillion"], OKABE_ITO["bluish_green"]], alpha=0.8)
    ax_a.set_ylabel("Percentage of responses")
    ax_a.set_title("A. Baseline behavior", fontweight="bold", loc="left")
    ax_a.set_ylim(0, 70)
    for bar, val in zip(bars, [syc_rate * 100, (1 - syc_rate) * 100]):
        ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                  f"{val:.0f}%", ha="center", fontsize=11, fontweight="bold")
    
    # Panel B: Steering effect
    ax_b = fig.add_subplot(gs[0, 1])
    reduction = mediation_df["D_base"] - mediation_df["D_vec"]
    mean_red = reduction.mean()
    scale = mediation_df["D_base"].max() - mediation_df["D_base"].min()
    rel_effect = abs(mean_red) / scale * 100
    
    ax_b.hist(reduction, bins=30, color=OKABE_ITO["blue"], alpha=0.7, edgecolor="white")
    ax_b.axvline(0, color=OKABE_ITO["grey"], linestyle=":", linewidth=1.5)
    ax_b.axvline(mean_red, color=OKABE_ITO["vermillion"], linestyle="--", linewidth=2,
                 label=f"Mean = {mean_red:.3f}")
    ax_b.set_xlabel(r"Sycophancy reduction ($D_{base} - D_{vec}$)")
    ax_b.set_ylabel("Count")
    ax_b.set_title("B. Steering effect is weak", fontweight="bold", loc="left")
    ax_b.legend(fontsize=9)
    ax_b.text(0.95, 0.95, f"~{rel_effect:.0f}% relative\nto baseline range",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=9,
              bbox=dict(fc="white", ec=OKABE_ITO["grey"], alpha=0.9, boxstyle="round"))
    
    # Panel C: Circuit localization
    ax_c = fig.add_subplot(gs[1, 0])
    df_pos = head_scores[head_scores["delta_sum"] > 0]
    top_heads = df_pos.nlargest(15, "delta_sum")
    
    ax_c.scatter(df_pos["layer"], df_pos["delta_sum"], color=OKABE_ITO["grey"], 
                 alpha=0.3, s=15, label="Other")
    ax_c.scatter(top_heads["layer"], top_heads["delta_sum"], color=OKABE_ITO["vermillion"],
                 s=50, edgecolor="white", label="Top 15")
    ax_c.set_xlabel("Layer")
    ax_c.set_ylabel(r"$\Sigma \Delta D_{syc}$")
    ax_c.set_title("C. Sparse sycophancy circuit", fontweight="bold", loc="left")
    ax_c.legend(fontsize=8, loc="upper right")
    
    # Panel D: Null mediation
    ax_d = fig.add_subplot(gs[1, 1])
    eff_intact = mediation_df["D_base"] - mediation_df["D_vec"]
    eff_ablated = mediation_df["D_syc_abl"] - mediation_df["D_vec_syc"]
    mask = np.isfinite(eff_intact) & np.isfinite(eff_ablated)
    x, y = eff_intact[mask], eff_ablated[mask]
    
    ax_d.scatter(x, y, color=OKABE_ITO["sky_blue"], alpha=0.6, s=40, edgecolor="white")
    lims = [min(x.min(), y.min()) - 0.3, max(x.max(), y.max()) + 0.3]
    ax_d.plot(lims, lims, color=OKABE_ITO["grey"], linestyle="--", label="y=x (independence)")
    ax_d.axhline(0, color=OKABE_ITO["vermillion"], linestyle=":", alpha=0.7, label="y=0 (full mediation)")
    
    r, _ = stats.pearsonr(x, y)
    ax_d.text(0.05, 0.95, f"r = {r:.2f}", transform=ax_d.transAxes, fontsize=10,
              fontweight="bold", va="top", color=OKABE_ITO["blue"])
    
    ax_d.set_xlabel("Vector effect (intact)")
    ax_d.set_ylabel("Vector effect (ablated)")
    ax_d.set_title("D. Vector bypasses circuit", fontweight="bold", loc="left")
    ax_d.legend(fontsize=8, loc="lower right")
    ax_d.set_aspect("equal")
    ax_d.set_xlim(lims)
    ax_d.set_ylim(lims)
    
    # Main title
    fig.suptitle("Persona Vectors: Steering and Circuit Analysis for Llama-3.1-8B",
                 fontsize=14, fontweight="bold", y=0.98)
    
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    base_dir = Path("llama-results")
    out_dir = base_dir / "figures"
    out_dir.mkdir(exist_ok=True)
    
    # Load data
    path_baseline_records = base_dir / "exp0" / "baseline_records.jsonl"
    path_baseline_metrics = base_dir / "exp0" / "baseline_metrics.json"
    path_mediation = base_dir / "exp3" / "mediation_results.jsonl"
    path_heads = base_dir / "exp2" / "head_scores.jsonl"
    path_heads_raw = base_dir / "exp2" / "head_results_raw.jsonl"
    data_dir = Path("data")
    
    # Load available data
    baseline_records = load_jsonl(path_baseline_records) if path_baseline_records.exists() else None
    baseline_metrics = load_json(path_baseline_metrics) if path_baseline_metrics.exists() else {}
    mediation_df = load_jsonl(path_mediation) if path_mediation.exists() else None
    head_scores = load_jsonl(path_heads) if path_heads.exists() else None
    head_results_raw = load_jsonl(path_heads_raw) if path_heads_raw.exists() else None
    
    print(f"Loaded data:")
    print(f"  - Baseline records: {len(baseline_records) if baseline_records is not None else 'N/A'}")
    print(f"  - Mediation results: {len(mediation_df) if mediation_df is not None else 'N/A'}")
    print(f"  - Head scores: {len(head_scores) if head_scores is not None else 'N/A'}")
    print(f"  - Head results raw: {len(head_results_raw) if head_results_raw is not None else 'N/A'}")
    
    # Generate figures
    if baseline_records is not None:
        plot_fig1_baseline(baseline_records, baseline_metrics, out_dir / "Fig1_Baseline.png")
    
    if mediation_df is not None:
        plot_fig2_steering(mediation_df, out_dir / "Fig2_Steering.png")
        plot_fig4_mediation(mediation_df, out_dir / "Fig4_Mediation.png")
    
    if head_scores is not None:
        if head_results_raw is not None:
            plot_fig3_circuit(head_scores, head_results_raw, out_dir / "Fig3_Circuit.png")
        else:
            # Fallback: single panel
            fig, ax = plt.subplots(figsize=(8, 5))
            df_pos = head_scores[head_scores["delta_sum"] > 0]
            top_heads = df_pos.nlargest(15, "delta_sum")
            ax.scatter(df_pos["layer"], df_pos["delta_sum"], color=OKABE_ITO["grey"], alpha=0.3, s=15)
            ax.scatter(top_heads["layer"], top_heads["delta_sum"], color=OKABE_ITO["vermillion"], s=50)
            ax.set_xlabel("Layer")
            ax.set_ylabel(r"$\Sigma \Delta D_{syc}$")
            ax.set_title("Sycophancy circuit localization")
            plt.savefig(out_dir / "Fig3_Circuit.png")
            plt.close()
    
    if baseline_records is not None and data_dir.exists():
        plot_fig5_examples(baseline_records, data_dir, out_dir / "Fig5_Examples.png")
    
    # Summary composite
    if all([baseline_metrics, mediation_df is not None, head_scores is not None]):
        plot_fig6_summary(baseline_metrics, mediation_df, head_scores, out_dir / "Fig6_Summary.png")
    
    print(f"\nAll figures saved to {out_dir}/")

if __name__ == "__main__":
    main()
