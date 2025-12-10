#!/usr/bin/env python3
"""
plot_results.py

Publication-quality plots for:
  1) Steering grid (layer × alpha).
  2) Top sycophancy heads (path patching).
  3) Mediation bar plot (vector effect under ablations).
  4) Mediation scatter (intact vs ablated vector effect).

Assumes results are in the JSON/JSONL formats defined in experiment.md.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------
# Global style: Okabe–Ito palette + rcParams
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

mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "font.size": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.titlepad": 8,
        "figure.autolayout": True,
    }
)


def okabe_diverging() -> LinearSegmentedColormap:
    """
    Diverging colormap: 'good' (negative ΔD_syc) in blue/green,
    'bad' (positive) in vermillion, neutral near light grey.
    """
    return LinearSegmentedColormap.from_list(
        "okabe_div",
        [
            OKABE_ITO["blue"],
            "#f5f5f5",
            OKABE_ITO["vermillion"],
        ],
    )


# ---------------------------------------------------------------------
# Utilities to load results
# ---------------------------------------------------------------------

def load_steering_grid(path: Path) -> pd.DataFrame:
    """
    Expect JSONL or CSV with columns:
      - layer (int)
      - alpha (float)
      - delta_D_syc_mean (float)  # mean D_base - D_vec
    """
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        rows = []
        with path.open() as f:
            for line in f:
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)

    required = {"layer", "alpha", "delta_D_syc_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Steering grid file missing columns: {missing}")

    return df


def load_head_scores(path: Path) -> pd.DataFrame:
    """
    Expect JSONL or CSV with columns:
      - layer (int)
      - head (int)
      - delta_sum (float)
      - count (int)
    """
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        rows = []
        with path.open() as f:
            for line in f:
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)

    required = {"layer", "head", "delta_sum", "count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Head score file missing columns: {missing}")

    return df


def load_mediation_results(path: Path) -> pd.DataFrame:
    """
    Expect JSONL or CSV with columns per example:
      - example_id
      - D_base
      - D_vec
      - D_syc_abl
      - D_rand_abl
      - D_vec_syc
      - D_vec_rand
    """
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        rows = []
        with path.open() as f:
            for line in f:
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)

    required = {
        "example_id",
        "D_base",
        "D_vec",
        "D_syc_abl",
        "D_rand_abl",
        "D_vec_syc",
        "D_vec_rand",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mediation file missing columns: {missing}")

    return df


def load_vector_selection_grid(path: Path) -> pd.DataFrame:
    """
    Fallback for Exp1: vector_selection.json with a 'grid' list of dicts
    containing layer/alpha/mean_effect.
    """
    data = json.loads(path.read_text())
    grid = data.get("grid", [])
    if not grid:
        raise ValueError("vector_selection.json has no 'grid' entries")

    df = pd.DataFrame(grid)
    # Standardize column name expected by downstream plotting
    if "mean_effect" in df.columns:
        df = df.rename(columns={"mean_effect": "delta_D_syc_mean"})

    required = {"layer", "alpha", "delta_D_syc_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Vector selection grid missing columns: {missing}")

    return df


# ---------------------------------------------------------------------
# Locate results root
# ---------------------------------------------------------------------


def find_results_root() -> Path:
    """
    Prefer a populated results directory; fall back to updated-results.
    """
    candidates = [Path("results"), Path("updated-results")]
    for p in candidates:
        if any((p / f"exp{i}").exists() for i in (1, 2, 3)):
            return p
    return candidates[0]


# ---------------------------------------------------------------------
# 1. Steering grid
# ---------------------------------------------------------------------

def plot_steering_grid(
    df: pd.DataFrame,
    selected_layer: int,
    selected_alpha: float,
    out_path: Path,
) -> None:
    """
    Heatmap of mean ΔD_syc (D_base - D_vec) by (layer, alpha).
    Negative = reduction in sycophancy (good).
    """
    # Pivot into matrix
    grid = (
        df.pivot_table(
            index="layer",
            columns="alpha",
            values="delta_D_syc_mean",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    layers = grid.index.values
    alphas = grid.columns.values
    data = grid.values

    fig, ax = plt.subplots(figsize=(6.5, 7.0))

    vmax = np.nanmax(np.abs(data))
    im = ax.imshow(
        data,
        cmap=okabe_diverging(),
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        origin="lower",
    )

    # Axis ticks
    ax.set_xticks(np.arange(len(alphas)))
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_xlabel("Steering strength α")

    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_ylabel("Layer index")

    # Outline the selected (layer, alpha)
    if selected_layer in layers and selected_alpha in alphas:
        li = np.where(layers == selected_layer)[0][0]
        ai = np.where(alphas == selected_alpha)[0][0]
        rect = plt.Rectangle(
            (ai - 0.5, li - 0.5),
            1.0,
            1.0,
            fill=False,
            linewidth=2.0,
            edgecolor=OKABE_ITO["orange"],
        )
        ax.add_patch(rect)
        ax.text(
            ai,
            li + 0.35,
            "chosen",
            ha="center",
            va="center",
            fontsize=9,
            color=OKABE_ITO["black"],
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                boxstyle="round,pad=0.15",
                alpha=0.8,
            ),
        )

    # Only annotate cells with |Δ| above small threshold to avoid clutter
    thresh = 0.03
    for i, l in enumerate(layers):
        for j, a in enumerate(alphas):
            val = data[i, j]
            if np.isnan(val) or abs(val) < thresh:
                continue
            txt_color = "white" if abs(val) > 0.5 * vmax else OKABE_ITO["black"]
            ax.text(
                j,
                i,
                f"{val:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=txt_color,
            )

    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("ΔD$_{syc}$ (D$_{base}$ − D$_{steered}$)")

    ax.set_title("Steering efficacy by layer and strength")

    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------
# 2. Top sycophancy heads
# ---------------------------------------------------------------------

def plot_top_heads(
    df: pd.DataFrame,
    top_k: int,
    out_path: Path,
) -> None:
    """
    Horizontal bar plot of top-k heads by cumulative ΔD_syc,
    plus inset showing cumulative impact per layer.
    """
    # Only heads with positive contribution
    df_pos = df[df["delta_sum"] > 0].copy()
    if df_pos.empty:
        raise ValueError("No heads with positive delta_sum found")

    # Top-k by delta_sum
    df_top = df_pos.sort_values("delta_sum", ascending=False).head(top_k)
    df_top["head_label"] = [
        f"L{l}.H{h}" for l, h in zip(df_top["layer"], df_top["head"])
    ]

    # Layer-wise totals (for inset)
    layer_totals = (
        df_pos.groupby("layer")["delta_sum"].sum().reset_index(name="total_delta")
    )

    fig = plt.figure(figsize=(7.5, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.3], wspace=0.35)

    # Main bar plot (horizontal for readability)
    ax = fig.add_subplot(gs[0, 0])

    y_pos = np.arange(len(df_top))[::-1]  # top at top
    ax.barh(
        y_pos,
        df_top["delta_sum"],
        color=OKABE_ITO["vermillion"],
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_top["head_label"])
    ax.set_xlabel("Cumulative impact ΣΔD$_{syc}$")
    ax.set_title(f"Top {top_k} heads driving sycophancy")

    for yi, val in zip(y_pos, df_top["delta_sum"]):
        ax.text(
            val,
            yi,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color=OKABE_ITO["black"],
        )

    # Inset: per-layer impact
    ax_in = fig.add_subplot(gs[0, 1])
    ax_in.bar(
        layer_totals["layer"],
        layer_totals["total_delta"],
        color=OKABE_ITO["sky_blue"],
    )
    ax_in.set_xlabel("Layer")
    ax_in.set_ylabel("ΣΔD$_{syc}$")
    ax_in.set_title("Total impact per layer", fontsize=11)

    fig.suptitle("Circuit identification via path patching", y=1.02, fontsize=15)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 3. Mediation bar plot
# ---------------------------------------------------------------------

def summarise_mediation(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Return mean and standard error of vector effect (D_base - D_condition)
    for three relevant conditions:
      - 'vec'       : D_base - D_vec
      - 'vec_rand'  : D_rand_abl - D_vec_rand
      - 'vec_syc'   : D_syc_abl - D_vec_syc
    """
    effects = {}
    # Clean steering effect in intact model
    e_clean = df["D_base"] - df["D_vec"]
    effects["Vector only"] = (e_clean.mean(), e_clean.std(ddof=1) / math.sqrt(len(e_clean)))

    e_rand = df["D_rand_abl"] - df["D_vec_rand"]
    effects["Vector + rand ablation"] = (
        e_rand.mean(),
        e_rand.std(ddof=1) / math.sqrt(len(e_rand)),
    )

    e_syc = df["D_syc_abl"] - df["D_vec_syc"]
    effects["Vector + syc ablation"] = (
        e_syc.mean(),
        e_syc.std(ddof=1) / math.sqrt(len(e_syc)),
    )

    return effects


def plot_mediation_bars(
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Bar plot comparing mean vector effect across conditions with 95% CIs.
    """
    effects = summarise_mediation(df)

    labels = list(effects.keys())
    means = np.array([effects[k][0] for k in labels])
    ses = np.array([effects[k][1] for k in labels])
    cis = 1.96 * ses

    x = np.arange(len(labels))

    colors = [
        OKABE_ITO["bluish_green"],      # Vector only
        OKABE_ITO["grey"],             # Vec + rand ablation
        OKABE_ITO["vermillion"],       # Vec + syc ablation
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(
        x,
        means,
        yerr=cis,
        capsize=4,
        color=colors,
        edgecolor=OKABE_ITO["black"],
        linewidth=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Reduction in sycophancy (ΔD$_{syc}$)")
    ax.set_title("Does ablating sycophancy heads weaken the steering vector?")

    # Annotate relative change syc vs rand
    mean_rand = means[1]
    mean_syc = means[2]
    if mean_rand != 0:
        rel_change = 100 * (mean_syc - mean_rand) / abs(mean_rand)
        ax.text(
            1.5,
            max(means[1], means[2]) * 1.10,
            f"Relative change syc vs rand: {rel_change:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.axhline(0.0, color=OKABE_ITO["black"], linewidth=0.7)

    n = len(df)
    ax.text(
        0.01,
        0.98,
        f"n = {n} examples",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        color=OKABE_ITO["grey"],
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 4. Mediation scatter
# ---------------------------------------------------------------------

def plot_mediation_scatter(
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Scatter of vector effect in intact vs syc-ablated models.
      x = E_clean = D_base - D_vec
      y = E_syc   = D_syc_abl - D_vec_syc
    """
    e_clean = df["D_base"] - df["D_vec"]
    e_syc = df["D_syc_abl"] - df["D_vec_syc"]

    x = e_clean.values
    y = e_syc.values

    # Filter to finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # Basic correlation
    r = np.corrcoef(x, y)[0, 1]

    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    ax.scatter(
        x,
        y,
        s=26,
        alpha=0.85,
        linewidth=0.4,
        edgecolor="white",
        color=OKABE_ITO["sky_blue"],
    )

    lim_min = min(x.min(), y.min())
    lim_max = max(x.max(), y.max())
    pad = 0.1 * (lim_max - lim_min)
    ax.set_xlim(lim_min - pad, lim_max + pad)
    ax.set_ylim(lim_min - pad, lim_max + pad)

    # y = x (ablation has no effect)
    ax.plot(
        [lim_min - pad, lim_max + pad],
        [lim_min - pad, lim_max + pad],
        linestyle="--",
        linewidth=1.0,
        color=OKABE_ITO["grey"],
        label="Independence (no effect of ablation)",
    )

    # y = 0 (full mediation)
    ax.axhline(
        0.0,
        linestyle=(0, (3, 3)),
        linewidth=1.0,
        color=OKABE_ITO["vermillion"],
        label="Full mediation (vector killed)",
    )

    ax.set_xlabel("Vector effect in intact model ΔD$_{syc}$")
    ax.set_ylabel("Vector effect in syc-ablated model ΔD$_{syc}$")
    ax.set_title("Vector efficacy: intact vs. syc-ablated")

    ax.legend(frameon=False, loc="upper left")

    n = len(x)
    ax.text(
        0.98,
        0.02,
        f"n = {n}\nPearson r = {r:.2f}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(
            facecolor="white",
            edgecolor=OKABE_ITO["grey"],
            boxstyle="round,pad=0.2",
            alpha=0.9,
        ),
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------

def main():
    root = find_results_root()
    root.mkdir(parents=True, exist_ok=True)

    # 1. Steering grid
    steering_path = root / "exp1" / "steering_grid.jsonl"
    vector_selection_path = root / "exp1" / "vector_selection.json"

    grid_df = None
    if steering_path.exists():
        grid_df = load_steering_grid(steering_path)
    elif vector_selection_path.exists():
        grid_df = load_vector_selection_grid(vector_selection_path)

    if grid_df is not None:
        # These should match your chosen L* and alpha*
        L_star = int(grid_df.loc[grid_df["delta_D_syc_mean"].idxmin(), "layer"])
        alpha_star = float(grid_df.loc[grid_df["delta_D_syc_mean"].idxmin(), "alpha"])
        plot_steering_grid(
            grid_df,
            selected_layer=L_star,
            selected_alpha=alpha_star,
            out_path=root / "fig_01_steering_grid.png",
        )

    # 2. Top heads
    heads_path = root / "exp2" / "head_scores.jsonl"
    if heads_path.exists():
        head_df = load_head_scores(heads_path)
        plot_top_heads(
            head_df,
            top_k=15,
            out_path=root / "fig_02_top_heads.png",
        )

    # 3 + 4. Mediation
    med_path = root / "exp3" / "mediation_results.jsonl"
    if med_path.exists():
        med_df = load_mediation_results(med_path)
        plot_mediation_bars(
            med_df,
            out_path=root / "fig_03_mediation_bars.png",
        )
        plot_mediation_scatter(
            med_df,
            out_path=root / "fig_04_mediation_scatter.png",
        )


if __name__ == "__main__":
    main()
