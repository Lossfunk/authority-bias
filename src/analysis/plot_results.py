"""
Utility to visualize current experiment outputs with research-style figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PALETTE = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]


def _set_theme():
    sns.set_theme(style="whitegrid", context="talk")
    sns.set_palette(PALETTE)
    plt.rcParams.update(
        {
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.figsize": (10, 6),
        }
    )


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_baseline(fig_dir: Path):
    records_path = Path("results/exp0/baseline_records.jsonl")
    if not records_path.exists():
        return
    df = pd.read_json(records_path, lines=True)
    fig, ax = plt.subplots()
    sns.histplot(df, x="D_syc", bins=15, kde=True, color=PALETTE[0], alpha=0.85, ax=ax)
    ax.set_title("Baseline $D_{syc}$ Distribution (TinyLlama)")
    ax.set_xlabel("$D_{syc}$ (log prob wrong − log prob right)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(fig_dir / "baseline_dsyc_hist.png", dpi=300)
    plt.close(fig)


def plot_caa_grid(fig_dir: Path):
    selection_path = Path("results/exp1/vector_selection.json")
    if not selection_path.exists():
        return
    with selection_path.open("r") as f:
        selection = json.load(f)
    df = pd.DataFrame(selection["grid"])
    if df.empty:
        return
    top_layers = (
        df.groupby("layer")["mean_effect"].max().sort_values(ascending=False).head(6).index.tolist()
    )
    df_top = df[df["layer"].isin(top_layers)]
    fig, ax = plt.subplots()
    sns.pointplot(
        data=df_top,
        x="alpha",
        y="mean_effect",
        hue="layer",
        dodge=0.2,
        markers="o",
        linestyles="-",
        ax=ax,
    )
    ax.set_title("CAA Grid Search (mean $E_{clean}$ by layer, TinyLlama subset)")
    ax.set_xlabel("Steering strength $\\alpha$")
    ax.set_ylabel("Mean $E_{clean}$ (Δ$D_{syc}$)")
    ax.legend(title="Layer", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "exp1_caa_grid.png", dpi=300)
    plt.close(fig)


def plot_head_deltas(fig_dir: Path):
    head_scores_path = Path("results/exp2/head_scores.jsonl")
    if not head_scores_path.exists():
        return
    df = pd.read_json(head_scores_path, lines=True)
    if df.empty:
        return
    df_top = df.nlargest(10, "delta_sum").copy()
    df_top["label"] = df_top.apply(lambda r: f"L{r.layer} · H{r.head}", axis=1)
    fig, ax = plt.subplots()
    colors = sns.color_palette(PALETTE, n_colors=len(df_top))
    sns.barplot(
        data=df_top,
        x="delta_sum",
        y="label",
        hue="label",
        dodge=False,
        palette=colors,
        ax=ax,
        legend=False,
    )
    ax.set_title("Top attention heads by Δ$D_{syc}$ (TinyLlama path patching)")
    ax.set_xlabel("Cumulative Δ$D_{syc}$ when patched from neutral")
    ax.set_ylabel("Layer · Head")
    fig.tight_layout()
    fig.savefig(fig_dir / "exp2_head_deltas.png", dpi=300)
    plt.close(fig)


def plot_mediation_grid(fig_dir: Path):
    results_path = Path("results/exp3/mediation_results.jsonl")
    if not results_path.exists():
        return
    df = pd.read_json(results_path, lines=True)
    if df.empty:
        return
    cond_map = {
        "D_base": "Base",
        "D_vec": "Vec",
        "D_syc_abl": "Syc Abl",
        "D_rand_abl": "Rand Abl",
        "D_vec_syc": "Vec + Syc",
        "D_vec_rand": "Vec + Rand",
    }
    melted = df.melt(
        id_vars="example_id",
        value_vars=[
            "D_base",
            "D_vec",
            "D_syc_abl",
            "D_rand_abl",
            "D_vec_syc",
            "D_vec_rand",
        ],
        var_name="condition",
        value_name="D_syc",
    )
    melted["condition_label"] = melted["condition"].map(cond_map)
    condition_order = [cond_map[k] for k in cond_map]
    fig, ax = plt.subplots(figsize=(12, 6))
    plot = sns.barplot(
        data=melted,
        x="condition_label",
        y="D_syc",
        hue="example_id",
        order=condition_order,
        ax=ax,
    )
    ax.set_title("Exp3 Mediation Grid (Δ per condition, Tiny subset)")
    ax.set_xlabel("Condition")
    ax.set_ylabel("$D_{syc}$")
    ax.legend(title="Example", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "exp3_mediation_grid.png", dpi=300)
    plt.close(fig)


def main():
    _set_theme()
    fig_dir = Path("results/figures")
    _ensure_dir(fig_dir)
    plot_baseline(fig_dir)
    plot_caa_grid(fig_dir)
    plot_head_deltas(fig_dir)
    plot_mediation_grid(fig_dir)
    print(f"Saved figures to {fig_dir.resolve()}")


if __name__ == "__main__":
    main()


