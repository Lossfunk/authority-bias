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
    if df.empty:
        return

    # Clip extreme outliers for readability; report them separately.
    p99 = df["D_syc"].quantile(0.99)
    clipped = df["D_syc"].clip(lower=df["D_syc"].quantile(0.01), upper=p99)
    outlier_count = (df["D_syc"] > p99).sum()

    fig, ax = plt.subplots()
    sns.histplot(clipped, bins=20, color=PALETTE[0], alpha=0.85, ax=ax)
    syc_rate = (df["D_syc"] > 0).mean()
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_title("Baseline $D_{syc}$ (length-normalized)")
    ax.set_xlabel("$D_{syc}$ (mean log-prob wrong − mean log-prob right)")
    ax.set_ylabel("Count")
    ax.text(
        0.98,
        0.95,
        f"N={len(df)}\nSyc rate: {syc_rate:.2f}\nOutliers >99p: {outlier_count}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    # Optionally constrain x-axis for readability when outliers exist.
    if outlier_count > 0:
        xmin = clipped.quantile(0.01)
        xmax = min(clipped.quantile(0.99), xmin + 15)
        ax.set_xlim(xmin, xmax)
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
    L_star = selection.get("L_star")
    alpha_star = selection.get("alpha_star")
    n_samples = selection.get("num_syc_examples", 0) + selection.get("num_truth_examples", 0)
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
        errorbar=("se", 1.0) if "std_effect" in df_top.columns else None,
        ax=ax,
    )
    if L_star is not None and alpha_star is not None:
        ax.scatter(
            [alpha_star],
            df[(df["layer"] == L_star) & (df["alpha"] == alpha_star)]["mean_effect"],
            color="red",
            zorder=5,
            s=80,
            marker="*",
            label=f"Chosen L{L_star}, α={alpha_star}",
        )
    ax.set_title("CAA Grid (mean $E_{clean}$ by layer)")
    ax.set_xlabel("Steering strength $\\alpha$")
    ax.set_ylabel("Mean $E_{clean}$ (Δ$D_{syc}$)")
    ax.legend(title="Layer", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.text(
        0.02,
        0.98,
        f"N={n_samples} (syc+truth)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
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
    df_pos = df[df["delta_sum"] > 0].copy()
    df_top = df_pos.nlargest(20, "delta_sum").copy()
    if df_top.empty:
        return
    df_top["label"] = df_top.apply(lambda r: f"L{r.layer}-H{r.head}", axis=1)
    df_top["pos_fraction"] = df_top.get("pos_fraction", 0.0)
    fig, ax = plt.subplots()
    sns.barplot(
        data=df_top,
        x="delta_sum",
        y="label",
        color=PALETTE[2],
        ax=ax,
        orient="h",
    )
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_title("Top heads by cumulative Δ$D_{syc}$ (patched from neutral)")
    ax.set_xlabel("Cumulative Δ$D_{syc}$")
    ax.set_ylabel("Layer–Head")
    # Annotate consistency (fraction of examples with positive delta)
    for i, (_, row) in enumerate(df_top.iterrows()):
        ax.text(
            row["delta_sum"],
            i,
            f"{row['pos_fraction']:.2f}",
            va="center",
            ha="left",
            fontsize=9,
        )
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
    # Plot change-from-base aggregated across examples with error bars
    conds = [
        ("Vec", "D_vec"),
        ("Syc Abl", "D_syc_abl"),
        ("Rand Abl", "D_rand_abl"),
        ("Vec + Syc", "D_vec_syc"),
        ("Vec + Rand", "D_vec_rand"),
    ]
    rows = []
    for label, col in conds:
        delta = df[col] - df["D_base"]
        rows.append(
            {
                "condition": label,
                "mean": delta.mean(),
                "sem": delta.sem() if len(delta) > 1 else 0.0,
                "n": len(delta),
            }
        )
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x="condition", y="mean", color=PALETTE[1], ax=ax)
    ax.errorbar(
        x=range(len(plot_df)),
        y=plot_df["mean"],
        yerr=plot_df["sem"],
        fmt="none",
        ecolor="k",
        capsize=4,
        linewidth=1.2,
    )
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_title("Mediation: Δ$D_{syc}$ vs Base (mean ± s.e.)")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Δ$D_{syc}$ (condition − Base)")
    fig.tight_layout()
    fig.savefig(fig_dir / "exp3_mediation_grid.png", dpi=300)
    plt.close(fig)


def plot_mediation_scatter(fig_dir: Path):
    results_path = Path("results/exp3/mediation_results.jsonl")
    if not results_path.exists():
        return
    df = pd.read_json(results_path, lines=True)
    if df.empty:
        return
    df = df.copy()
    df["E_clean"] = df["D_base"] - df["D_vec"]
    df["E_syc"] = df["D_syc_abl"] - df["D_vec_syc"]
    df["E_rand"] = df["D_rand_abl"] - df["D_vec_rand"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    sns.scatterplot(data=df, x="E_clean", y="E_syc", hue="example_id", ax=axes[0])
    axes[0].axhline(0, color="k", linestyle="--", linewidth=1)
    axes[0].axvline(0, color="k", linestyle="--", linewidth=1)
    axes[0].set_title("E_syc vs E_clean")
    axes[0].set_xlabel("E_clean = D_base − D_vec")
    axes[0].set_ylabel("E_syc = D_syc_abl − D_vec_syc")

    sns.scatterplot(data=df, x="E_clean", y="E_rand", hue="example_id", ax=axes[1], legend=False)
    axes[1].axhline(0, color="k", linestyle="--", linewidth=1)
    axes[1].axvline(0, color="k", linestyle="--", linewidth=1)
    axes[1].set_title("E_rand vs E_clean")
    axes[1].set_xlabel("E_clean = D_base − D_vec")
    axes[1].set_ylabel("E_rand = D_rand_abl − D_vec_rand")

    fig.tight_layout()
    fig.savefig(fig_dir / "exp3_mediation_scatter.png", dpi=300)
    plt.close(fig)


def main():
    _set_theme()
    fig_dir = Path("results/figures")
    _ensure_dir(fig_dir)
    plot_baseline(fig_dir)
    plot_caa_grid(fig_dir)
    plot_head_deltas(fig_dir)
    plot_mediation_grid(fig_dir)
    plot_mediation_scatter(fig_dir)
    print(f"Saved figures to {fig_dir.resolve()}")


if __name__ == "__main__":
    main()
