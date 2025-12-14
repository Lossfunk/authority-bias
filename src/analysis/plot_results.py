#!/usr/bin/env python3
"""
plot_results.py

Generates publication-quality visualizations for the Persona Vectors project.
Covers Exp1 (Steering), Exp2 (Path Patching), and Exp3 (Mediation).

Figures:
  1. Sycophancy Shift (KDE): Baseline vs. Steered distributions.
  2. Steering Grid (Heatmap): Efficacy across Layers x Alpha.
  3. Sycophancy Circuit (Heatmap): Head importance map.
  4. Mediation Scatter (Scatter): Decoupling of vector effect and circuit.

Style: Okabe-Ito palette, minimalist aesthetic.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Any

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

# Set global style for "Publication Quality"
mpl.rcParams.update({
    "figure.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelweight": "normal",
    "legend.frameon": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

def get_diverging_cmap():
    """Custom diverging map: Blue (Good) <-> White <-> Vermillion (Bad/Sycophantic)"""
    return LinearSegmentedColormap.from_list(
        "okabe_div",
        [OKABE_ITO["blue"], "#FFFFFF", OKABE_ITO["vermillion"]]
    )

# ---------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------

def load_jsonl(path: Path) -> pd.DataFrame:
    with path.open("r") as f:
        return pd.DataFrame([json.loads(line) for line in f])

def load_exp1_grid(path: Path) -> pd.DataFrame:
    """Load grid search results from vector_selection.json"""
    with path.open("r") as f:
        data = json.load(f)
    return pd.DataFrame(data["grid"])

# ---------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------

def plot_fig1_shift(df: pd.DataFrame, out_path: Path):
    """
    Figure 1: The Sycophancy Shift.
    KDE of D_syc scores for Baseline vs. Steered (Vector).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Data
    base = df["D_base"]
    vec = df["D_vec"]
    
    # Plot KDEs
    sns.kdeplot(base, fill=True, color=OKABE_ITO["grey"], label="Baseline", alpha=0.4, ax=ax, linewidth=1.5)
    sns.kdeplot(vec, fill=True, color=OKABE_ITO["blue"], label="Steered (L1, α=2)", alpha=0.4, ax=ax, linewidth=1.5)
    
    # Mean lines
    ax.axvline(base.mean(), color=OKABE_ITO["grey"], linestyle="--", alpha=0.8)
    ax.axvline(vec.mean(), color=OKABE_ITO["blue"], linestyle="--", alpha=0.8)
    
    # Annotate shift
    shift = base.mean() - vec.mean()
    ax.text(
        (base.mean() + vec.mean()) / 2, 
        ax.get_ylim()[1] * 0.9, 
        f"Δ = {shift:.3f}", 
        ha="center", 
        color=OKABE_ITO["black"],
        fontweight="bold"
    )

    ax.set_xlabel("Sycophancy Score ($D_{syc}$)")
    ax.set_ylabel("Density")
    ax.set_title("Steering Reduces Sycophancy Distribution")
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    plt.close()

def plot_fig2_grid(df: pd.DataFrame, out_path: Path, selected_layer=1, selected_alpha=2.0):
    """
    Figure 2: Steering Grid Search.
    Heatmap of Mean Effect by Layer and Alpha.
    """
    # Pivot for heatmap: Rows=Layer, Cols=Alpha
    pivot = df.pivot(index="layer", columns="alpha", values="mean_effect")
    pivot = pivot.sort_index(ascending=False) # Layer 0 at bottom typically, but heatmap puts index 0 at top. Let's keep 0 at top for matrix.
    # Actually, standard is 0 at top. Let's sort index ascending so L0 is top.
    pivot = pivot.sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Heatmap
    # Note: mean_effect > 0 means D_base > D_vec (Reduction in sycophancy). 
    # So Positive = Good (Blue). Negative = Bad (Red).
    sns.heatmap(
        pivot, 
        cmap=get_diverging_cmap(), 
        center=0, 
        annot=True, 
        fmt=".2f", 
        cbar_kws={"label": "Mean Reduction ($\Delta D_{syc}$)"},
        linewidths=0.5,
        ax=ax
    )

    # Highlight selection
    # Find integer coordinates for the selected cell
    # columns are alphas, index is layers
    col_idx = list(pivot.columns).index(selected_alpha)
    row_idx = list(pivot.index).index(selected_layer)
    
    from matplotlib.patches import Rectangle
    rect = Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor=OKABE_ITO["black"], lw=2)
    ax.add_patch(rect)

    ax.set_title("Steering Efficacy by Layer & Alpha")
    ax.set_xlabel("Alpha (Steering Strength)")
    ax.set_ylabel("Layer")
    
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    plt.close()

def plot_fig3_circuit(df: pd.DataFrame, out_path: Path):
    """
    Figure 3: Sycophancy Circuit Map.
    Heatmap of 32 Layers x 32 Heads showing delta_sum.
    """
    # Create full 32x32 grid (fill missing with 0)
    grid = np.zeros((32, 32))
    
    for _, row in df.iterrows():
        l, h = int(row["layer"]), int(row["head"])
        if 0 <= l < 32 and 0 <= h < 32:
            grid[l, h] = row["delta_sum"]

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Diverging map: Positive delta_sum means removing head reduced sycophancy? 
    # Wait, check metric definition. 
    # Usually: delta = D_orig - D_patched. 
    # If head CAUSES sycophancy, patching it (neutralizing) reduces sycophancy -> D_patched < D_orig -> delta > 0.
    # So Positive = Sycophancy Head.
    # Let's map Positive to Red (Bad/Active Head) and near-zero to White.
    
    # Custom map for this: White -> Red
    cmap = LinearSegmentedColormap.from_list("white_red", ["#FFFFFF", OKABE_ITO["vermillion"]])
    
    # We allow bidirectional just in case (some heads might oppose sycophancy)
    # So use diverging: Blue (Anti-Syc) <-> White <-> Red (Syc)
    sns.heatmap(
        grid, 
        cmap=get_diverging_cmap(), 
        center=0,
        square=True,
        cbar_kws={"label": "Cumulative Impact ($\Sigma \Delta D_{syc}$)"},
        vmax=np.max(np.abs(grid)), # Symmetric range
        vmin=-np.max(np.abs(grid)),
        ax=ax
    )

    ax.set_title("The Sycophancy Circuit (Llama-3.1-8B)")
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    ax.invert_yaxis() # Layer 0 at bottom
    
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    plt.close()

def plot_fig4_mediation(df: pd.DataFrame, out_path: Path):
    """
    Figure 4: Mediation Scatter.
    X: Vector Effect (Intact). Y: Vector Effect (Syc-Ablated).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Calculate effects
    # Effect = D_base - D_vec (How much steering reduced sycophancy)
    eff_intact = df["D_base"] - df["D_vec"]
    
    # Effect in ablated model = D_syc_abl - D_vec_syc
    eff_ablated = df["D_syc_abl"] - df["D_vec_syc"]
    
    ax.scatter(eff_intact, eff_ablated, alpha=0.6, color=OKABE_ITO["sky_blue"], edgecolor="white", s=60)
    
    # Limits
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    
    # Diagonal y=x (No Mediation)
    ax.plot(lims, lims, color=OKABE_ITO["grey"], linestyle="--", label="Independence (y=x)")
    
    # Horizontal y=0 (Full Mediation)
    ax.axhline(0, color=OKABE_ITO["vermillion"], linestyle=":", label="Full Mediation (y=0)")
    
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    ax.set_xlabel("Vector Effect (Intact Model)")
    ax.set_ylabel("Vector Effect (Syc-Ablated Model)")
    ax.set_title("Decoupling of Steering & Circuit")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    plt.close()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    base_dir = Path("llama-results")
    
    # 1. Mediation / Shift
    path_exp3 = base_dir / "exp3" / "mediation_results.jsonl"
    if path_exp3.exists():
        df_med = load_jsonl(path_exp3)
        plot_fig1_shift(df_med, base_dir / "Fig1_Shift.png")
        plot_fig4_mediation(df_med, base_dir / "Fig4_Mediation.png")
    else:
        print(f"Skipping Figs 1&4: {path_exp3} not found")

    # 2. Grid
    path_exp1 = base_dir / "exp1" / "vector_selection.json"
    if path_exp1.exists():
        df_grid = load_exp1_grid(path_exp1)
        plot_fig2_grid(df_grid, base_dir / "Fig2_Grid.png")
    else:
        print(f"Skipping Fig 2: {path_exp1} not found")

    # 3. Circuit
    path_exp2 = base_dir / "exp2" / "head_scores.jsonl"
    if path_exp2.exists():
        df_heads = load_jsonl(path_exp2)
        plot_fig3_circuit(df_heads, base_dir / "Fig3_Circuit.png")
    else:
        print(f"Skipping Fig 3: {path_exp2} not found")

if __name__ == "__main__":
    main()
