"""
Create publication-quality figures for MATS application.

Visualizes the systematic failure of steering interventions on Llama-3.1-8B-Instruct.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# Style Configuration - Clean, modern aesthetic
# ============================================================================

# Color palette - muted, professional
COLORS = {
    'negative': '#E74C3C',      # Red for wrong direction
    'positive': '#27AE60',      # Green for correct direction  
    'neutral': '#7F8C8D',       # Gray for noise/no effect
    'primary': '#2C3E50',       # Dark blue-gray for primary elements
    'secondary': '#3498DB',     # Blue for secondary elements
    'background': '#FAFBFC',    # Light background
    'grid': '#E8E8E8',          # Subtle grid
    'text': '#2C3E50',          # Dark text
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
    'axes.facecolor': COLORS['background'],
    'figure.facecolor': 'white',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})


def load_json(path: Path) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


# ============================================================================
# Figure 1: CAA Steering Effects (Exp4)
# ============================================================================

def create_figure1_caa_effects(results_dir: Path, output_dir: Path):
    """Bar chart showing CAA steering effects are in wrong direction."""
    
    data = load_json(results_dir / "exp4/exp4/alpha_sweep_results.json")
    
    alphas = []
    baseline_effects = []
    baseline_stds = []
    attn_effects = []
    
    for alpha_str, result in data["per_alpha_results"].items():
        alpha = float(alpha_str)
        alphas.append(alpha)
        baseline_effects.append(result["results"]["baseline"]["mean_effect"])
        baseline_stds.append(result["results"]["baseline"]["std_effect"] / np.sqrt(result["results"]["baseline"]["n"]))
        attn_effects.append(result["results"]["all_attn_ablated"]["mean_effect"])
    
    # Sort by alpha
    order = np.argsort(alphas)
    alphas = np.array(alphas)[order]
    baseline_effects = np.array(baseline_effects)[order]
    baseline_stds = np.array(baseline_stds)[order]
    attn_effects = np.array(attn_effects)[order]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(alphas))
    width = 0.35
    
    # Color bars based on direction
    baseline_colors = [COLORS['negative'] if e < 0 else COLORS['positive'] for e in baseline_effects]
    attn_colors = [COLORS['negative'] if e < 0 else COLORS['positive'] for e in attn_effects]
    
    bars1 = ax.bar(x - width/2, baseline_effects, width, 
                   color=baseline_colors, alpha=0.85,
                   yerr=baseline_stds, capsize=4, 
                   error_kw={'elinewidth': 1.5, 'capthick': 1.5},
                   label='Baseline (Steering Active)', edgecolor='white', linewidth=1)
    
    bars2 = ax.bar(x + width/2, attn_effects, width,
                   color=[c if c == COLORS['positive'] else COLORS['secondary'] for c in attn_colors], 
                   alpha=0.6,
                   label='Attention Ablated', edgecolor='white', linewidth=1)
    
    # Reference line at zero
    ax.axhline(y=0, color=COLORS['primary'], linestyle='-', linewidth=1.5, alpha=0.5)
    
    # Annotations
    ax.annotate('More Sycophantic', xy=(0.02, 0.15), xycoords='axes fraction',
                fontsize=10, color=COLORS['negative'], style='italic', alpha=0.8)
    ax.annotate('Less Sycophantic', xy=(0.02, 0.85), xycoords='axes fraction',
                fontsize=10, color=COLORS['positive'], style='italic', alpha=0.8)
    
    ax.set_xlabel('Steering Strength (α)', fontsize=12)
    ax.set_ylabel('Effect on D_syc (Δ)', fontsize=12)
    ax.set_title('CAA Steering Makes Model MORE Sycophantic', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'α = {a:.0f}' for a in alphas])
    
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_ylim(-0.15, 0.15)
    ax.grid(axis='y', alpha=0.3, color=COLORS['grid'])
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_caa_steering_effects.png")
    plt.savefig(output_dir / "fig1_caa_steering_effects.pdf")
    plt.close()
    print(f"✓ Created Figure 1: CAA Steering Effects")


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
    
    # Custom colormap: red (negative/wrong) -> white (zero) -> green (positive/correct)
    cmap = LinearSegmentedColormap.from_list('diverging', 
        [COLORS['negative'], 'white', COLORS['positive']])
    
    # Symmetric color scale
    vmax = max(abs(effects.min()), abs(effects.max()))
    vmax = max(vmax, 0.1)  # Ensure some range
    
    im = ax.imshow(effects, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)
    
    # Add text annotations
    for i in range(len(layers)):
        for j in range(len(alphas)):
            val = effects[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else COLORS['text']
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                   fontsize=10, color=color, fontweight='medium')
    
    ax.set_xticks(np.arange(len(alphas)))
    ax.set_yticks(np.arange(len(layers)))
    ax.set_xticklabels([f'α = {a:.0f}' for a in alphas])
    ax.set_yticklabels([f'Layer {l}' for l in layers])
    
    ax.set_xlabel('Steering Strength (α)', fontsize=12)
    ax.set_ylabel('Injection Layer', fontsize=12)
    ax.set_title('Late-Layer CAA Injection: No Consistent Effect', fontsize=14, pad=15)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Effect on D_syc', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig2_layer_alpha_heatmap.png")
    plt.savefig(output_dir / "fig2_layer_alpha_heatmap.pdf")
    plt.close()
    print(f"✓ Created Figure 2: Layer-Alpha Heatmap")


# ============================================================================
# Figure 3: CCM Head Effects Distribution
# ============================================================================

def create_figure3_head_effects(results_dir: Path, output_dir: Path):
    """Scatter plot of head indirect effects showing weak mediation."""
    
    data = load_json(results_dir / "exp5/head_effects.json")
    
    layers = [d["layer"] for d in data]
    heads = [d["head"] for d in data]
    effects = [d["effect"] for d in data]
    stds = [d.get("std", 0) for d in data]
    
    # Sort by absolute effect for top-50 highlighting
    abs_effects = [abs(e) for e in effects]
    top_50_indices = np.argsort(abs_effects)[-50:]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Scatter plot of all heads
    colors = []
    for i, (l, e) in enumerate(zip(layers, effects)):
        if i in top_50_indices:
            if e < 0:
                colors.append(COLORS['negative'])
            else:
                colors.append(COLORS['positive'])
        else:
            colors.append(COLORS['neutral'])
    
    alphas = [0.9 if i in top_50_indices else 0.3 for i in range(len(effects))]
    sizes = [50 if i in top_50_indices else 15 for i in range(len(effects))]
    
    for i in range(len(effects)):
        ax1.scatter(layers[i], effects[i], c=colors[i], alpha=alphas[i], s=sizes[i], edgecolors='white', linewidths=0.5)
    
    ax1.axhline(y=0, color=COLORS['primary'], linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=0.1, color=COLORS['grid'], linestyle=':', linewidth=1, alpha=0.5)
    ax1.axhline(y=-0.1, color=COLORS['grid'], linestyle=':', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Indirect Effect', fontsize=12)
    ax1.set_title('Per-Head Indirect Effects (1024 heads)', fontsize=14, pad=10)
    ax1.grid(alpha=0.3, color=COLORS['grid'])
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['positive'], alpha=0.9, label='Top-50 (positive)'),
        mpatches.Patch(facecolor=COLORS['negative'], alpha=0.9, label='Top-50 (negative)'),
        mpatches.Patch(facecolor=COLORS['neutral'], alpha=0.3, label='Other heads'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    # Right: Histogram of effects
    ax2.hist(effects, bins=50, color=COLORS['secondary'], alpha=0.7, edgecolor='white', linewidth=0.5)
    ax2.axvline(x=0, color=COLORS['primary'], linestyle='--', linewidth=2)
    ax2.axvline(x=np.mean(effects), color=COLORS['negative'], linestyle='-', linewidth=2, 
                label=f'Mean: {np.mean(effects):.4f}')
    
    ax2.set_xlabel('Indirect Effect', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Indirect Effects', fontsize=14, pad=10)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, color=COLORS['grid'])
    
    # Add text box with key stats
    textstr = f'Max |effect|: {max(abs_effects):.3f}\nMean: {np.mean(effects):.4f}\nStd: {np.std(effects):.4f}'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=COLORS['grid'])
    ax2.text(0.98, 0.75, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig3_head_indirect_effects.png")
    plt.savefig(output_dir / "fig3_head_indirect_effects.pdf")
    plt.close()
    print(f"✓ Created Figure 3: Head Indirect Effects")


# ============================================================================
# Figure 4: Method Comparison
# ============================================================================

def create_figure4_method_comparison(results_dir: Path, output_dir: Path):
    """Bar chart comparing all steering methods."""
    
    # Collect all results
    methods = []
    effects = []
    stds = []
    colors = []
    
    # CAA (exp4)
    caa_data = load_json(results_dir / "exp4/exp4/alpha_sweep_results.json")
    best_caa = caa_data["per_alpha_results"]["16.0"]["results"]["baseline"]
    methods.append("CAA\n(residual, α=16)")
    effects.append(best_caa["mean_effect"])
    stds.append(best_caa["std_effect"] / np.sqrt(best_caa["n"]))
    colors.append(COLORS['negative'])
    
    # CAA late-layer (exp4b)
    late_data = load_json(results_dir / "exp4/exp4/layer_alpha_sweep_results.json")
    best_late = late_data["per_config_results"]["L29_a16.0"]["results"]["baseline"]
    methods.append("CAA Late-Layer\n(L29, α=16)")
    effects.append(best_late["mean_effect"])
    stds.append(best_late["std_effect"] / np.sqrt(best_late["n"]))
    colors.append(COLORS['negative'])
    
    # CCM mean-diff (exp5)
    ccm_mean = load_json(results_dir / "exp5/evaluation_results.json")
    methods.append("CCM Mean-Diff\n(top-50, α=1)")
    effects.append(ccm_mean["1.0"]["mean_effect"])
    stds.append(ccm_mean["1.0"]["std_effect"] / np.sqrt(ccm_mean["1.0"]["n"]))
    colors.append(COLORS['negative'])
    
    # CCM patching (exp5c)
    if (results_dir / "exp5/evaluation_results_patching.json").exists():
        ccm_patch = load_json(results_dir / "exp5/evaluation_results_patching.json")
        methods.append("CCM Patching\n(top-50, α=1)")
        effects.append(ccm_patch["1.0"]["mean_effect"])
        stds.append(ccm_patch["1.0"]["std_effect"] / np.sqrt(ccm_patch["1.0"]["n"]))
        colors.append(COLORS['negative'])
    
    # Full patching (exp5d)
    if (results_dir / "exp5/evaluation_results_patching_all_replace.json").exists():
        full_patch = load_json(results_dir / "exp5/evaluation_results_patching_all_replace.json")
        methods.append("Full Patching\n(all 1024 heads)")
        effects.append(full_patch["1.0"]["mean_effect"])
        stds.append(full_patch["1.0"]["std_effect"] / np.sqrt(full_patch["1.0"]["n"]))
        colors.append(COLORS['negative'])
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(methods))
    
    # Determine colors based on effect direction
    bar_colors = [COLORS['negative'] if e < 0 else COLORS['positive'] for e in effects]
    
    bars = ax.bar(x, effects, color=bar_colors, alpha=0.85, 
                  yerr=stds, capsize=5,
                  error_kw={'elinewidth': 2, 'capthick': 2},
                  edgecolor='white', linewidth=2)
    
    # Reference line
    ax.axhline(y=0, color=COLORS['primary'], linestyle='-', linewidth=2, alpha=0.5)
    
    # Add value labels
    for i, (bar, effect) in enumerate(zip(bars, effects)):
        height = bar.get_height()
        offset = 0.015 if height < 0 else -0.015
        va = 'top' if height < 0 else 'bottom'
        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{effect:.3f}', ha='center', va=va, fontsize=11, fontweight='bold',
                color=COLORS['text'])
    
    ax.set_ylabel('Effect on D_syc (Δ)', fontsize=13)
    ax.set_title('All Steering Methods Fail on Llama-3.1-8B-Instruct', fontsize=15, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    
    # Annotations
    ax.annotate('WRONG DIRECTION\n(increases sycophancy)', 
                xy=(0.98, 0.15), xycoords='axes fraction',
                fontsize=11, color=COLORS['negative'], style='italic', 
                ha='right', fontweight='bold')
    
    ax.set_ylim(-0.18, 0.08)
    ax.grid(axis='y', alpha=0.3, color=COLORS['grid'])
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig4_method_comparison.png")
    plt.savefig(output_dir / "fig4_method_comparison.pdf")
    plt.close()
    print(f"✓ Created Figure 4: Method Comparison")


# ============================================================================
# Figure 5: Summary Infographic
# ============================================================================

def create_figure5_summary(results_dir: Path, output_dir: Path):
    """Clean summary showing progression of failed approaches."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Systematic Failure of Activation Steering on Llama-3.1-8B-Instruct',
            ha='center', va='top', fontsize=16, fontweight='bold', color=COLORS['text'])
    
    # Subtitle
    ax.text(5, 8.9, 'Each approach was tested, failed, and informed the next experiment',
            ha='center', va='top', fontsize=12, style='italic', color=COLORS['neutral'])
    
    # Define the flow
    experiments = [
        {
            'name': 'CAA Vectors',
            'desc': 'Mean activation difference\nin residual stream',
            'result': 'Wrong direction\n(effect: -0.065)',
            'hypothesis': '"Wrong layer?"',
        },
        {
            'name': 'Late-Layer CAA',
            'desc': 'Inject at layers 29-31\n(following literature)',
            'result': 'No effect / noise\n(effect: ~0)',
            'hypothesis': '"Wrong component?"',
        },
        {
            'name': 'CCM Head Selection',
            'desc': 'Causal mediation to find\nrelevant attention heads',
            'result': 'Weak mediators\n(max |IE| = 0.10)',
            'hypothesis': '"Wrong steering?"',
        },
        {
            'name': 'CCM Patching',
            'desc': 'Activation patching\non selected heads',
            'result': 'Wrong direction\n(effect: -0.079)',
            'hypothesis': '"Wrong heads?"',
        },
        {
            'name': 'Full Model Patch',
            'desc': 'Patch ALL 1024 heads\nwith neutral activations',
            'result': 'Wrong direction\n(effect: -0.107)',
            'hypothesis': None,
        },
    ]
    
    # Draw boxes
    box_width = 1.6
    box_height = 1.8
    y_pos = 6.5
    start_x = 0.5
    gap = 0.3
    
    for i, exp in enumerate(experiments):
        x = start_x + i * (box_width + gap)
        
        # Main box
        rect = mpatches.FancyBboxPatch((x, y_pos - box_height/2), box_width, box_height,
                                        boxstyle="round,pad=0.05,rounding_size=0.15",
                                        facecolor='white', edgecolor=COLORS['primary'],
                                        linewidth=2)
        ax.add_patch(rect)
        
        # Experiment name
        ax.text(x + box_width/2, y_pos + 0.7, exp['name'],
                ha='center', va='center', fontsize=10, fontweight='bold',
                color=COLORS['primary'])
        
        # Description
        ax.text(x + box_width/2, y_pos + 0.2, exp['desc'],
                ha='center', va='center', fontsize=8, color=COLORS['text'])
        
        # Result (in red box)
        result_rect = mpatches.FancyBboxPatch((x + 0.1, y_pos - 0.75), box_width - 0.2, 0.6,
                                              boxstyle="round,pad=0.02,rounding_size=0.1",
                                              facecolor=COLORS['negative'], alpha=0.15,
                                              edgecolor=COLORS['negative'], linewidth=1)
        ax.add_patch(result_rect)
        ax.text(x + box_width/2, y_pos - 0.45, exp['result'],
                ha='center', va='center', fontsize=8, color=COLORS['negative'],
                fontweight='medium')
        
        # Arrow and hypothesis to next
        if exp['hypothesis']:
            arrow_x = x + box_width + gap/2
            ax.annotate('', xy=(arrow_x + gap/2 - 0.05, y_pos),
                       xytext=(arrow_x - gap/2 + 0.05, y_pos),
                       arrowprops=dict(arrowstyle='->', color=COLORS['secondary'],
                                      lw=2, mutation_scale=15))
            ax.text(arrow_x, y_pos + 0.5, exp['hypothesis'],
                   ha='center', va='bottom', fontsize=8, style='italic',
                   color=COLORS['secondary'])
    
    # Final conclusion box
    conclusion_rect = mpatches.FancyBboxPatch((1, 2.8), 8, 1.5,
                                              boxstyle="round,pad=0.1,rounding_size=0.2",
                                              facecolor=COLORS['negative'], alpha=0.1,
                                              edgecolor=COLORS['negative'], linewidth=3)
    ax.add_patch(conclusion_rect)
    
    ax.text(5, 3.8, 'CONCLUSION', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['negative'])
    ax.text(5, 3.2, 'Llama-3.1-8B-Instruct is resistant to activation-based steering.\n'
                    'Even patching ALL attention heads with neutral activations increases sycophancy.',
            ha='center', va='center', fontsize=11, color=COLORS['text'])
    
    # Key insight box
    insight_rect = mpatches.FancyBboxPatch((1, 0.8), 8, 1.3,
                                           boxstyle="round,pad=0.1,rounding_size=0.2",
                                           facecolor=COLORS['secondary'], alpha=0.1,
                                           edgecolor=COLORS['secondary'], linewidth=2)
    ax.add_patch(insight_rect)
    
    ax.text(5, 1.7, 'KEY INSIGHT', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLORS['secondary'])
    ax.text(5, 1.15, 'RL-optimized vectors (GRPO) work on this model. Mean-difference methods (CAA, CCM) do not.\n'
                     'The difference is optimization: extracted vectors capture correlation, not causation.',
            ha='center', va='center', fontsize=10, color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig5_summary_infographic.png")
    plt.savefig(output_dir / "fig5_summary_infographic.pdf")
    plt.close()
    print(f"✓ Created Figure 5: Summary Infographic")


# ============================================================================
# Main
# ============================================================================

def main():
    results_dir = Path("llama-results")
    output_dir = results_dir / "figures_mats"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Creating MATS Application Figures")
    print("="*60 + "\n")
    
    create_figure1_caa_effects(results_dir, output_dir)
    create_figure2_layer_alpha_heatmap(results_dir, output_dir)
    create_figure3_head_effects(results_dir, output_dir)
    create_figure4_method_comparison(results_dir, output_dir)
    create_figure5_summary(results_dir, output_dir)
    
    print("\n" + "="*60)
    print(f"All figures saved to: {output_dir}")
    print("="*60 + "\n")
    
    # List created files
    print("Created files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  • {f.name}")


if __name__ == "__main__":
    main()

