"""
Create publication-quality figures for MATS application.

Visualizes the systematic failure of steering interventions on Llama-3.1-8B-Instruct.
Uses Okabe-Ito colorblind-friendly palette.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# Okabe-Ito Colorblind-Friendly Palette
# ============================================================================

OKABE_ITO = {
    'black': '#000000',
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'bluish_green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'reddish_purple': '#CC79A7',
}

# Semantic color assignments
COLORS = {
    'negative': OKABE_ITO['vermillion'],     # Wrong direction
    'positive': OKABE_ITO['bluish_green'],   # Correct direction
    'neutral': OKABE_ITO['sky_blue'],        # Neutral/comparison
    'primary': OKABE_ITO['blue'],            # Primary elements
    'secondary': OKABE_ITO['orange'],        # Secondary elements
    'accent': OKABE_ITO['reddish_purple'],   # Accent
    'highlight': OKABE_ITO['yellow'],        # Highlights
    'text': '#2C3E50',                        # Dark text
    'background': '#FAFBFC',                  # Light background
    'grid': '#E0E0E0',                        # Subtle grid
}

# Figure styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'medium',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})


SAVE_PDF = False


def save_figure(output_dir: Path, filename_stem: str):
    plt.savefig(output_dir / f"{filename_stem}.png")
    if SAVE_PDF:
        plt.savefig(output_dir / f"{filename_stem}.pdf")


def load_json(path: Path) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def add_fancy_frame(ax, facecolor='white', edgecolor='#CCCCCC', linewidth=1.5, pad=0.02):
    """Add a fancy rounded frame around the axes."""
    # Get the axes position in figure coordinates
    bbox = ax.get_position()
    
    # Create fancy box patch
    fancy_box = FancyBboxPatch(
        (bbox.x0 - pad, bbox.y0 - pad),
        bbox.width + 2*pad,
        bbox.height + 2*pad,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=ax.figure.transFigure,
        zorder=-1
    )
    ax.figure.patches.append(fancy_box)


# ============================================================================
# Figure 1: CAA Steering Effects (Exp4)
# ============================================================================

def create_figure1_caa_effects(results_dir: Path, output_dir: Path):
    """Bar chart showing CAA steering effects across alpha values."""
    
    data = load_json(results_dir / "exp4/exp4/alpha_sweep_results.json")
    
    alphas = []
    baseline_effects = []
    baseline_stds = []
    attn_effects = []
    attn_stds = []
    
    for alpha_str, result in data["per_alpha_results"].items():
        alpha = float(alpha_str)
        alphas.append(alpha)
        n = result["results"]["baseline"]["n"]
        baseline_effects.append(result["results"]["baseline"]["mean_effect"])
        baseline_stds.append(result["results"]["baseline"]["std_effect"] / np.sqrt(n))
        attn_effects.append(result["results"]["all_attn_ablated"]["mean_effect"])
        attn_stds.append(result["results"]["all_attn_ablated"]["std_effect"] / np.sqrt(n))
    
    # Sort by alpha
    order = np.argsort(alphas)
    alphas = np.array(alphas)[order]
    baseline_effects = np.array(baseline_effects)[order]
    baseline_stds = np.array(baseline_stds)[order]
    attn_effects = np.array(attn_effects)[order]
    attn_stds = np.array(attn_stds)[order]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(alphas))
    width = 0.35
    
    # Bars with Okabe-Ito colors
    bars1 = ax.bar(x - width/2, baseline_effects, width, 
                   color=COLORS['negative'], alpha=0.85,
                   yerr=baseline_stds, capsize=4, 
                   error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'color': COLORS['text']},
                   label='Steering Active', edgecolor='white', linewidth=1.5)
    
    bars2 = ax.bar(x + width/2, attn_effects, width,
                   color=COLORS['neutral'], alpha=0.85,
                   yerr=attn_stds, capsize=4,
                   error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'color': COLORS['text']},
                   label='Attention Ablated', edgecolor='white', linewidth=1.5)
    
    # Reference line at zero
    ax.axhline(y=0, color=COLORS['text'], linestyle='-', linewidth=1.5, alpha=0.4)
    
    ax.set_xlabel('Steering Strength (α)', fontsize=12, fontweight='medium')
    ax.set_ylabel('Effect on D_syc (Δ)', fontsize=12, fontweight='medium')
    ax.set_title('CAA Steering Effects by Alpha', fontsize=14, pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a:.0f}' for a in alphas])
    
    ax.legend(loc='lower left', framealpha=0.95, edgecolor=COLORS['grid'])
    ax.set_ylim(-0.15, 0.12)
    ax.grid(axis='y', alpha=0.4, color=COLORS['grid'], linestyle='-', linewidth=0.8)
    
    # Style spines
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(COLORS['text'])
        ax.spines[spine].set_linewidth(1.2)
    
    plt.tight_layout()
    save_figure(output_dir, "fig1_caa_steering_effects")
    plt.close()
    print(f"  Created Figure 1: CAA Steering Effects")


# ============================================================================
# Figure 2: Late-Layer Injection Heatmap (Exp4b)
# ============================================================================

def create_figure2_layer_alpha_heatmap(results_dir: Path, output_dir: Path):
    """Heatmap showing effects across layers and alpha values."""
    
    data = load_json(results_dir / "exp4/exp4/layer_alpha_sweep_results.json")
    
    layers = sorted(data["layer_sweep"])
    alphas = sorted(data["alpha_sweep"])
    
    # Build effect matrix
    effects = np.zeros((len(layers), len(alphas)))
    
    for i, layer in enumerate(layers):
        for j, alpha in enumerate(alphas):
            key = f"L{layer}_a{alpha}"
            if key in data["per_config_results"]:
                effects[i, j] = data["per_config_results"][key]["results"]["baseline"]["mean_effect"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Custom diverging colormap using Okabe-Ito
    cmap = LinearSegmentedColormap.from_list('okabe_diverging', 
        [COLORS['negative'], 'white', COLORS['positive']])
    
    # Symmetric color scale
    vmax = max(abs(effects.min()), abs(effects.max()))
    vmax = max(vmax, 0.08)
    
    im = ax.imshow(effects, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)
    
    # Add value annotations
    for i in range(len(layers)):
        for j in range(len(alphas)):
            val = effects[i, j]
            color = 'white' if abs(val) > vmax * 0.65 else COLORS['text']
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                   fontsize=11, color=color, fontweight='medium')
    
    ax.set_xticks(np.arange(len(alphas)))
    ax.set_yticks(np.arange(len(layers)))
    ax.set_xticklabels([f'{a:.0f}' for a in alphas])
    ax.set_yticklabels([f'Layer {l}' for l in layers])
    
    ax.set_xlabel('Steering Strength (α)', fontsize=12, fontweight='medium')
    ax.set_ylabel('Injection Layer', fontsize=12, fontweight='medium')
    ax.set_title('Late-Layer CAA Injection Effects', fontsize=14, pad=15, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Effect on D_syc', fontsize=11)
    cbar.outline.set_linewidth(1.2)
    
    plt.tight_layout()
    save_figure(output_dir, "fig2_layer_alpha_heatmap")
    plt.close()
    print(f"  Created Figure 2: Layer-Alpha Heatmap")


# ============================================================================
# Figure 3: CCM Head Effects Distribution
# ============================================================================

def create_figure3_head_effects(results_dir: Path, output_dir: Path):
    """Scatter plot of head indirect effects."""
    
    data = load_json(results_dir / "exp5/head_effects.json")
    
    layers = [d["layer"] for d in data]
    heads = [d["head"] for d in data]
    effects = [d["effect"] for d in data]
    
    # Sort by absolute effect for top-50 highlighting
    abs_effects = [abs(e) for e in effects]
    top_50_indices = set(np.argsort(abs_effects)[-50:])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Scatter plot
    for i, (l, e) in enumerate(zip(layers, effects)):
        if i in top_50_indices:
            color = COLORS['positive'] if e > 0 else COLORS['negative']
            alpha = 0.9
            size = 60
            zorder = 3
        else:
            color = COLORS['neutral']
            alpha = 0.25
            size = 20
            zorder = 1
        ax1.scatter(l, e, c=color, alpha=alpha, s=size, edgecolors='white', linewidths=0.5, zorder=zorder)
    
    ax1.axhline(y=0, color=COLORS['text'], linestyle='--', linewidth=1.2, alpha=0.5)
    
    ax1.set_xlabel('Layer', fontsize=12, fontweight='medium')
    ax1.set_ylabel('Indirect Effect', fontsize=12, fontweight='medium')
    ax1.set_title('Per-Head Indirect Effects', fontsize=14, pad=10, fontweight='bold')
    ax1.grid(alpha=0.3, color=COLORS['grid'], linestyle='-', linewidth=0.8)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['positive'], alpha=0.9, label='Top-50 (positive)', edgecolor='white'),
        mpatches.Patch(facecolor=COLORS['negative'], alpha=0.9, label='Top-50 (negative)', edgecolor='white'),
        mpatches.Patch(facecolor=COLORS['neutral'], alpha=0.25, label='Other heads', edgecolor='white'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.95, edgecolor=COLORS['grid'])
    
    # Right: Histogram
    ax2.hist(effects, bins=50, color=COLORS['primary'], alpha=0.8, edgecolor='white', linewidth=0.8)
    ax2.axvline(x=0, color=COLORS['text'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axvline(x=np.mean(effects), color=COLORS['secondary'], linestyle='-', linewidth=2.5, 
                label=f'Mean: {np.mean(effects):.4f}')
    
    ax2.set_xlabel('Indirect Effect', fontsize=12, fontweight='medium')
    ax2.set_ylabel('Count', fontsize=12, fontweight='medium')
    ax2.set_title('Distribution of Indirect Effects', fontsize=14, pad=10, fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.95, edgecolor=COLORS['grid'])
    ax2.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='-', linewidth=0.8)
    
    # Style spines
    for ax in [ax1, ax2]:
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(COLORS['text'])
            ax.spines[spine].set_linewidth(1.2)
    
    plt.tight_layout()
    save_figure(output_dir, "fig3_head_indirect_effects")
    plt.close()
    print(f"  Created Figure 3: Head Indirect Effects")


# ============================================================================
# Figure 4: Method Comparison
# ============================================================================

def create_figure4_method_comparison(results_dir: Path, output_dir: Path):
    """Bar chart comparing all steering methods."""
    
    # Collect all results
    methods = []
    effects = []
    stds = []
    hatches = []

    def add_method(label: str, mean_effect: float, std_effect: float, n: int, hatch: str = ""):
        methods.append(label)
        effects.append(mean_effect)
        stds.append(std_effect / np.sqrt(n))
        hatches.append(hatch)

    def add_eval_method(label: str, eval_path: Path, alpha_key: str = "1.0", hatch: str = ""):
        if not eval_path.exists():
            return
        data = load_json(eval_path)
        if alpha_key not in data:
            return
        rec = data[alpha_key]
        add_method(label, rec["mean_effect"], rec["std_effect"], rec["n"], hatch=hatch)
    
    # CAA (exp4)
    caa_data = load_json(results_dir / "exp4/exp4/alpha_sweep_results.json")
    best_caa = caa_data["per_alpha_results"]["16.0"]["results"]["baseline"]
    add_method("CAA\n(α=16)", best_caa["mean_effect"], best_caa["std_effect"], best_caa["n"])
    
    # CAA late-layer (exp4b)
    late_data = load_json(results_dir / "exp4/exp4/layer_alpha_sweep_results.json")
    best_late = late_data["per_config_results"]["L29_a16.0"]["results"]["baseline"]
    add_method("CAA Late\n(L29, α=16)", best_late["mean_effect"], best_late["std_effect"], best_late["n"])
    
    # CCM mean-diff (exp5)
    add_eval_method("CCM\nMean-Diff", results_dir / "exp5/evaluation_results.json")
    
    # CCM patching (exp5c)
    if (results_dir / "exp5/evaluation_results_patching.json").exists():
        add_eval_method("CCM\nPatching", results_dir / "exp5/evaluation_results_patching.json")
    
    # Replace-mode patching (Instruct)
    add_eval_method(
        "Patch Top-50\n(Instruct)",
        results_dir / "exp5/evaluation_results_patching_selected_replace.json",
    )
    add_eval_method(
        "Patch All\n(1024, Instruct)",
        results_dir / "exp5/evaluation_results_patching_all_replace.json",
    )

    # Replace-mode patching (Base)
    base_dir = results_dir / "exp5_llama31_base"
    add_eval_method(
        "Patch Top-50\n(Base)",
        base_dir / "evaluation_results_patching_selected_replace.json",
        hatch="///",
    )
    add_eval_method(
        "Patch All\n(1024, Base)",
        base_dir / "evaluation_results_patching_all_replace.json",
        hatch="///",
    )
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    x = np.arange(len(methods))
    
    # Color by effect direction
    bar_colors = [COLORS['negative'] if e < 0 else COLORS['positive'] for e in effects]
    
    bars = ax.bar(x, effects, color=bar_colors, alpha=0.85,
                  yerr=stds, capsize=5,
                  error_kw={'elinewidth': 2, 'capthick': 2, 'color': COLORS['text']},
                  edgecolor='white', linewidth=2)

    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
    
    # Reference line
    ax.axhline(y=0, color=COLORS['text'], linestyle='-', linewidth=1.5, alpha=0.4)
    
    ax.set_ylabel('Effect on D_syc (Δ)', fontsize=13, fontweight='medium')
    ax.set_title('Comparison of Steering Methods', fontsize=15, pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    
    ax.set_ylim(-0.16, 0.08)
    ax.grid(axis='y', alpha=0.4, color=COLORS['grid'], linestyle='-', linewidth=0.8)
    
    # Style spines
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(COLORS['text'])
        ax.spines[spine].set_linewidth(1.2)
    
    plt.tight_layout()
    save_figure(output_dir, "fig4_method_comparison")
    plt.close()
    print(f"  Created Figure 4: Method Comparison")


# ============================================================================
# Figure 5: Alpha Sweep Detail (All Conditions)
# ============================================================================

def create_figure5_alpha_sweep_detail(results_dir: Path, output_dir: Path):
    """Detailed alpha sweep showing all three conditions."""
    
    data = load_json(results_dir / "exp4/exp4/alpha_sweep_results.json")
    
    alphas = []
    baseline = []
    attn_ablated = []
    mlp_ablated = []
    
    for alpha_str in ["4.0", "8.0", "16.0", "24.0"]:
        result = data["per_alpha_results"][alpha_str]
        alphas.append(float(alpha_str))
        baseline.append(result["results"]["baseline"]["mean_effect"])
        attn_ablated.append(result["results"]["all_attn_ablated"]["mean_effect"])
        mlp_ablated.append(result["results"]["mlp_ablated"]["mean_effect"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(alphas))
    width = 0.25
    
    bars1 = ax.bar(x - width, baseline, width, label='Baseline', 
                   color=COLORS['negative'], alpha=0.85, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x, attn_ablated, width, label='Attn Ablated', 
                   color=COLORS['neutral'], alpha=0.85, edgecolor='white', linewidth=1.5)
    bars3 = ax.bar(x + width, mlp_ablated, width, label='MLP Ablated', 
                   color=COLORS['secondary'], alpha=0.85, edgecolor='white', linewidth=1.5)
    
    ax.axhline(y=0, color=COLORS['text'], linestyle='-', linewidth=1.5, alpha=0.4)
    
    ax.set_xlabel('Steering Strength (α)', fontsize=12, fontweight='medium')
    ax.set_ylabel('Effect on D_syc (Δ)', fontsize=12, fontweight='medium')
    ax.set_title('Steering Effects by Condition', fontsize=14, pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a:.0f}' for a in alphas])
    
    ax.legend(loc='lower left', framealpha=0.95, edgecolor=COLORS['grid'])
    ax.grid(axis='y', alpha=0.4, color=COLORS['grid'], linestyle='-', linewidth=0.8)
    
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(COLORS['text'])
        ax.spines[spine].set_linewidth(1.2)
    
    plt.tight_layout()
    save_figure(output_dir, "fig5_alpha_sweep_detail")
    plt.close()
    print(f"  Created Figure 5: Alpha Sweep Detail")


# ============================================================================
# Figure 5b: Summary Infographic (Systematic Failure + Base Contrast)
# ============================================================================

def create_figure5_summary_infographic(results_dir: Path, output_dir: Path):
    """Narrative infographic summarizing failures + Base comparison."""

    # Key numbers
    caa_data = load_json(results_dir / "exp4/exp4/alpha_sweep_results.json")
    caa_effect = caa_data["per_alpha_results"]["16.0"]["results"]["baseline"]["mean_effect"]

    late_data = load_json(results_dir / "exp4/exp4/layer_alpha_sweep_results.json")
    late_effect = late_data["per_config_results"]["L29_a16.0"]["results"]["baseline"]["mean_effect"]

    head_effects = load_json(results_dir / "exp5/head_effects.json")
    max_abs_ie = max(abs(d["effect"]) for d in head_effects)

    ccm_patch = load_json(results_dir / "exp5/evaluation_results_patching.json")
    ccm_patch_effect = ccm_patch["1.0"]["mean_effect"]

    full_instruct = load_json(results_dir / "exp5/evaluation_results_patching_all_replace.json")
    full_instruct_effect = full_instruct["1.0"]["mean_effect"]

    full_base = load_json(results_dir / "exp5_llama31_base/evaluation_results_patching_all_replace.json")
    full_base_effect = full_base["1.0"]["mean_effect"]

    fig = plt.figure(figsize=(14, 4.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Title
    ax.text(
        0.5,
        0.93,
        "Systematic Failure of Activation Steering on Llama-3.1-8B-Instruct",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        0.5,
        0.885,
        "All attention-based interventions fail; Base model flips sign under full patching",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#6B7280",
    )

    def fmt_delta(x: float) -> str:
        return f"{x:+.3f}"

    def draw_method_box(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        subtitle: str,
        result: str,
        result_fill: str,
    ):
        outer = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="white",
            edgecolor=COLORS["text"],
            linewidth=1.4,
        )
        ax.add_patch(outer)

        ax.text(
            x + w / 2,
            y + h * 0.78,
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=COLORS["text"],
        )
        ax.text(
            x + w / 2,
            y + h * 0.60,
            subtitle,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#6B7280",
            wrap=True,
        )

        inner = FancyBboxPatch(
            (x + w * 0.08, y + h * 0.14),
            w * 0.84,
            h * 0.30,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=result_fill,
            edgecolor="none",
            alpha=0.25,
        )
        ax.add_patch(inner)
        ax.text(
            x + w / 2,
            y + h * 0.29,
            result,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=result_fill,
        )

    # Layout
    y_top = 0.58
    box_h = 0.23
    box_w = 0.14
    gap = 0.02
    x0 = 0.03
    xs = [x0 + i * (box_w + gap) for i in range(6)]

    draw_method_box(
        xs[0],
        y_top,
        box_w,
        box_h,
        "CAA Vectors",
        "Mean activation diff\nin residual stream",
        f"Wrong dir\n(Δ={fmt_delta(caa_effect)})",
        COLORS["negative"],
    )
    draw_method_box(
        xs[1],
        y_top,
        box_w,
        box_h,
        "Late-Layer CAA",
        "Inject at last layers\n(L29–31)",
        f"No effect / wrong\n(Δ={fmt_delta(late_effect)})",
        COLORS["negative"],
    )
    draw_method_box(
        xs[2],
        y_top,
        box_w,
        box_h,
        "CCM Head Selection",
        "Rank heads by\nindirect effect",
        f"Weak mediators\n(max |IE|≈{max_abs_ie:.2f})",
        COLORS["secondary"],
    )
    draw_method_box(
        xs[3],
        y_top,
        box_w,
        box_h,
        "CCM Patching",
        "Activation patching\non top-50 heads",
        f"Wrong dir\n(Δ={fmt_delta(ccm_patch_effect)})",
        COLORS["negative"],
    )
    draw_method_box(
        xs[4],
        y_top,
        box_w,
        box_h,
        "Full Model Patch",
        "Patch ALL 1024 heads\n(Instruct)",
        f"Wrong dir\n(Δ={fmt_delta(full_instruct_effect)})",
        COLORS["negative"],
    )
    draw_method_box(
        xs[5],
        y_top,
        box_w,
        box_h,
        "Base Check",
        "Patch ALL 1024 heads\n(Base)",
        f"Correct dir\n(Δ={fmt_delta(full_base_effect)})",
        COLORS["positive"],
    )

    # Arrows
    y_mid = y_top + box_h * 0.52
    for i in range(5):
        ax.annotate(
            "",
            xy=(xs[i + 1] - gap * 0.2, y_mid),
            xytext=(xs[i] + box_w + gap * 0.2, y_mid),
            arrowprops=dict(arrowstyle="->", color=COLORS["neutral"], lw=2),
        )

    # Conclusion + Key Insight
    concl = FancyBboxPatch(
        (0.03, 0.27),
        0.94,
        0.16,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=COLORS["negative"],
        edgecolor="none",
        alpha=0.12,
    )
    ax.add_patch(concl)
    ax.text(0.5, 0.38, "CONCLUSION", ha="center", va="center", fontsize=11, fontweight="bold", color=COLORS["negative"])
    ax.text(
        0.5,
        0.315,
        "Instruct is resistant/inverted: even patching ALL attention heads increases sycophancy (Δ = -0.107).\n"
        "Base moves in the intended direction under the same patch (Δ = +0.035), but the effect is weak/noisy.",
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLORS["text"],
    )

    insight = FancyBboxPatch(
        (0.03, 0.08),
        0.94,
        0.14,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=COLORS["neutral"],
        edgecolor="none",
        alpha=0.16,
    )
    ax.add_patch(insight)
    ax.text(0.5, 0.16, "KEY INSIGHT", ha="center", va="center", fontsize=11, fontweight="bold", color=COLORS["primary"])
    ax.text(
        0.5,
        0.115,
        "Mean-difference vectors (CAA/CCM) fail on Instruct; the Base-vs-Instruct sign flip suggests RLHF changes the relevant internal geometry.",
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLORS["text"],
    )

    save_figure(output_dir, "fig5_summary_infographic")
    plt.close()
    print("  Created Figure 5b: Summary Infographic")


# ============================================================================
# Figure 6: CCM Steering Sweep
# ============================================================================

def create_figure6_ccm_alpha_sweep(results_dir: Path, output_dir: Path):
    """CCM steering results across alpha values."""
    
    ccm_mean = load_json(results_dir / "exp5/evaluation_results.json")
    
    alphas = []
    effects = []
    stds = []
    
    for alpha_str in ["1.0", "2.0", "4.0", "8.0"]:
        if alpha_str in ccm_mean:
            alphas.append(float(alpha_str))
            effects.append(ccm_mean[alpha_str]["mean_effect"])
            stds.append(ccm_mean[alpha_str]["std_effect"] / np.sqrt(ccm_mean[alpha_str]["n"]))
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    x = np.arange(len(alphas))
    
    bar_colors = [COLORS['negative'] if e < 0 else COLORS['positive'] for e in effects]
    
    bars = ax.bar(x, effects, color=bar_colors, alpha=0.85,
                  yerr=stds, capsize=5,
                  error_kw={'elinewidth': 2, 'capthick': 2, 'color': COLORS['text']},
                  edgecolor='white', linewidth=2)
    
    ax.axhline(y=0, color=COLORS['text'], linestyle='-', linewidth=1.5, alpha=0.4)
    
    ax.set_xlabel('Steering Strength (α)', fontsize=12, fontweight='medium')
    ax.set_ylabel('Effect on D_syc (Δ)', fontsize=12, fontweight='medium')
    ax.set_title('CCM Mean-Diff Steering Results', fontsize=14, pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a:.0f}' for a in alphas])
    
    ax.grid(axis='y', alpha=0.4, color=COLORS['grid'], linestyle='-', linewidth=0.8)
    
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(COLORS['text'])
        ax.spines[spine].set_linewidth(1.2)
    
    plt.tight_layout()
    save_figure(output_dir, "fig6_ccm_alpha_sweep")
    plt.close()
    print(f"  Created Figure 6: CCM Alpha Sweep")


# ============================================================================
# Figure 7: Head Layer Distribution
# ============================================================================

def create_figure7_head_layer_distribution(results_dir: Path, output_dir: Path):
    """Distribution of top heads by layer."""
    
    data = load_json(results_dir / "exp5/head_effects.json")
    
    layers = [d["layer"] for d in data]
    effects = [d["effect"] for d in data]
    
    # Get top 50 heads
    abs_effects = [abs(e) for e in effects]
    top_50_indices = np.argsort(abs_effects)[-50:]
    
    top_layers = [layers[i] for i in top_50_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Count heads per layer
    layer_counts = {}
    for l in top_layers:
        layer_counts[l] = layer_counts.get(l, 0) + 1
    
    all_layers = list(range(32))
    counts = [layer_counts.get(l, 0) for l in all_layers]
    
    bars = ax.bar(all_layers, counts, color=COLORS['primary'], alpha=0.85, 
                  edgecolor='white', linewidth=1)
    
    # Highlight early layers
    for i, count in enumerate(counts):
        if count > 0 and i < 10:
            bars[i].set_color(COLORS['secondary'])
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='medium')
    ax.set_ylabel('Number of Top-50 Heads', fontsize=12, fontweight='medium')
    ax.set_title('Top Mediating Heads by Layer', fontsize=14, pad=15, fontweight='bold')
    ax.set_xticks(range(0, 32, 4))
    
    ax.grid(axis='y', alpha=0.4, color=COLORS['grid'], linestyle='-', linewidth=0.8)
    
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(COLORS['text'])
        ax.spines[spine].set_linewidth(1.2)
    
    plt.tight_layout()
    save_figure(output_dir, "fig7_head_layer_distribution")
    plt.close()
    print(f"  Created Figure 7: Head Layer Distribution")


# ============================================================================
# Figure 8: Effect Size Comparison (Violin/Box)
# ============================================================================

def create_figure8_effect_distributions(results_dir: Path, output_dir: Path):
    """Box plots showing effect distributions for different methods."""
    
    # Load per-example effects
    ccm_mean = load_json(results_dir / "exp5/evaluation_results.json")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    data_to_plot = []
    labels = []
    
    for alpha_str in ["1.0", "2.0", "4.0", "8.0"]:
        if alpha_str in ccm_mean and "effects" in ccm_mean[alpha_str]:
            data_to_plot.append(ccm_mean[alpha_str]["effects"])
            labels.append(f'α={float(alpha_str):.0f}')
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                        medianprops={'color': COLORS['text'], 'linewidth': 2},
                        whiskerprops={'color': COLORS['text'], 'linewidth': 1.2},
                        capprops={'color': COLORS['text'], 'linewidth': 1.2},
                        flierprops={'markerfacecolor': COLORS['neutral'], 'markeredgecolor': 'white', 
                                   'markersize': 5, 'alpha': 0.6})
        
        # Color boxes
        colors_box = [COLORS['negative'], COLORS['negative'], COLORS['neutral'], COLORS['positive']]
        for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('white')
            patch.set_linewidth(1.5)
        
        ax.axhline(y=0, color=COLORS['text'], linestyle='--', linewidth=1.5, alpha=0.5)
        
        ax.set_xlabel('Steering Strength (α)', fontsize=12, fontweight='medium')
        ax.set_ylabel('Per-Example Effect', fontsize=12, fontweight='medium')
        ax.set_title('Effect Distribution by Alpha', fontsize=14, pad=15, fontweight='bold')
        ax.grid(axis='y', alpha=0.4, color=COLORS['grid'], linestyle='-', linewidth=0.8)
        
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(COLORS['text'])
            ax.spines[spine].set_linewidth(1.2)
    
    plt.tight_layout()
    save_figure(output_dir, "fig8_effect_distributions")
    plt.close()
    print(f"  Created Figure 8: Effect Distributions")


# ============================================================================
# Figure 9: Base vs Instruct (Patching Comparison)
# ============================================================================

def create_figure9_base_vs_instruct_patching(results_dir: Path, output_dir: Path):
    """Grouped bars comparing Base vs Instruct for replace-mode patching."""

    instruct_dir = results_dir / "exp5"
    base_dir = results_dir / "exp5_llama31_base"

    def load_point(eval_path: Path, alpha_key: str = "1.0"):
        if not eval_path.exists():
            return None
        data = load_json(eval_path)
        if alpha_key not in data:
            return None
        rec = data[alpha_key]
        return rec["mean_effect"], rec["std_effect"] / np.sqrt(rec["n"])

    instruct_selected = load_point(instruct_dir / "evaluation_results_patching_selected_replace.json")
    instruct_all = load_point(instruct_dir / "evaluation_results_patching_all_replace.json")
    base_selected = load_point(base_dir / "evaluation_results_patching_selected_replace.json")
    base_all = load_point(base_dir / "evaluation_results_patching_all_replace.json")

    if not all([instruct_selected, instruct_all, base_selected, base_all]):
        print("  Skipping Figure 9: missing Base/Instruct patching files")
        return

    categories = ["Patch Top-50", "Patch All (1024)"]
    x = np.arange(len(categories))
    width = 0.35

    instruct_means = [instruct_selected[0], instruct_all[0]]
    instruct_ses = [instruct_selected[1], instruct_all[1]]
    base_means = [base_selected[0], base_all[0]]
    base_ses = [base_selected[1], base_all[1]]

    fig, ax = plt.subplots(figsize=(10, 6))

    instruct_colors = [COLORS['negative'] if m < 0 else COLORS['positive'] for m in instruct_means]
    base_colors = [COLORS['negative'] if m < 0 else COLORS['positive'] for m in base_means]

    bars_instruct = ax.bar(
        x - width / 2,
        instruct_means,
        width,
        yerr=instruct_ses,
        capsize=5,
        color=instruct_colors,
        alpha=0.85,
        edgecolor='white',
        linewidth=2,
        error_kw={'elinewidth': 2, 'capthick': 2, 'color': COLORS['text']},
        label='Instruct',
    )

    bars_base = ax.bar(
        x + width / 2,
        base_means,
        width,
        yerr=base_ses,
        capsize=5,
        color=base_colors,
        alpha=0.85,
        edgecolor='white',
        linewidth=2,
        hatch='///',
        error_kw={'elinewidth': 2, 'capthick': 2, 'color': COLORS['text']},
        label='Base',
    )

    ax.axhline(y=0, color=COLORS['text'], linestyle='-', linewidth=1.5, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Effect on D_syc (Δ)', fontsize=12, fontweight='medium')
    ax.set_title('Replace-Mode Patching: Instruct vs Base', fontsize=14, pad=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.4, color=COLORS['grid'], linestyle='-', linewidth=0.8)
    ax.set_ylim(-0.14, 0.08)
    ax.legend(loc='lower left', framealpha=0.95, edgecolor=COLORS['grid'])

    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(COLORS['text'])
        ax.spines[spine].set_linewidth(1.2)

    plt.tight_layout()
    save_figure(output_dir, "fig9_base_vs_instruct_patching")
    plt.close()
    print("  Created Figure 9: Base vs Instruct Patching")


# ============================================================================
# Main
# ============================================================================

def main():
    results_dir = Path("llama-results")
    output_dir = results_dir / "figures_mats"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Creating MATS Figures (Okabe-Ito Palette)")
    print("="*60 + "\n")
    
    create_figure1_caa_effects(results_dir, output_dir)
    create_figure2_layer_alpha_heatmap(results_dir, output_dir)
    create_figure3_head_effects(results_dir, output_dir)
    create_figure4_method_comparison(results_dir, output_dir)
    create_figure5_summary_infographic(results_dir, output_dir)
    create_figure5_alpha_sweep_detail(results_dir, output_dir)
    create_figure6_ccm_alpha_sweep(results_dir, output_dir)
    create_figure7_head_layer_distribution(results_dir, output_dir)
    create_figure8_effect_distributions(results_dir, output_dir)
    create_figure9_base_vs_instruct_patching(results_dir, output_dir)
    
    print("\n" + "="*60)
    print(f"All figures saved to: {output_dir}")
    print("="*60 + "\n")
    
    # List created files
    print("Created files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
