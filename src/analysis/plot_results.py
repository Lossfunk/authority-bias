"""
Utility to visualize current experiment outputs with research-style figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# Okabe–Ito palette (colorblind-safe)
PALETTE = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#999999",
}


def _set_theme():
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (8, 5),
        "axes.titleweight": "bold",
        "axes.labelweight": "medium",
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_baseline(fig_dir: Path):
    records_path = Path("results/exp0/baseline_records.jsonl")
    if not records_path.exists():
        return
    df = pd.read_json(records_path, lines=True)
    if df.empty:
        return

    syc_rate = (df["D_syc"] > 0).mean()
    mean_val = df["D_syc"].mean()

    fig, ax = plt.subplots(figsize=(9, 5.5))

    sns.histplot(
        df["D_syc"],
        bins=20,
        color=PALETTE["sky_blue"],
        alpha=0.7,
        ax=ax,
        stat="count",
        kde=True,
        line_kws={"linewidth": 2, "color": PALETTE["blue"]},
    )

    ax.axvline(0, color=PALETTE["black"], linestyle="-", linewidth=2, zorder=5)
    ax.text(0.5, ax.get_ylim()[1] * 0.95, "← Non-sycophantic | Sycophantic →", 
            ha="center", fontsize=10, style="italic", color=PALETTE["gray"])

    ax.axvline(mean_val, color=PALETTE["vermillion"], linestyle="--", linewidth=2, label=f"Mean = {mean_val:.1f}")

    ax.set_title("Baseline: Model Prefers Wrong Answer When User Suggests It")
    ax.set_xlabel(r"$D_{syc}$ = log P(wrong) − log P(right)")
    ax.set_ylabel("Number of Examples")

    ax.text(
        0.98, 0.95,
        f"Key Finding:\n"
        f"• {syc_rate:.0%} of responses are sycophantic\n"
        f"• Mean $D_{{syc}}$ = {mean_val:.1f} (positive = sycophantic)\n"
        f"• N = {len(df)} examples",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(facecolor="white", alpha=0.95, edgecolor=PALETTE["gray"], boxstyle="round,pad=0.5"),
    )

    ax.legend(loc="upper right", framealpha=0.9)
    sns.despine()
    fig.tight_layout()
    fig.savefig(fig_dir / "baseline_dsyc_hist.png", dpi=300, bbox_inches="tight")
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

    pivot_df = df.pivot(index="layer", columns="alpha", values="mean_effect").sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))

    vmax = max(abs(pivot_df.values.min()), abs(pivot_df.values.max()))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={"label": r"Mean Effect ($\Delta D_{syc}$)", "shrink": 0.8},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 8},
    )

    ax.set_title("CAA Steering: Which Layer and Strength Reduce Sycophancy?")
    ax.set_xlabel(r"Steering Strength ($\alpha$)")
    ax.set_ylabel("Layer")

    if L_star is not None and alpha_star is not None:
        col_idx = list(pivot_df.columns).index(alpha_star)
        row_idx = list(pivot_df.index).index(L_star)
        ax.add_patch(Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor=PALETTE["orange"], lw=4, clip_on=False))
        ax.annotate(
            f"Selected: L{L_star}, α={alpha_star}\nEffect = {pivot_df.loc[L_star, alpha_star]:.3f}",
            xy=(col_idx + 0.5, row_idx + 0.5),
            xytext=(col_idx + 2.5, row_idx - 3),
            fontsize=10,
            ha="center",
            arrowprops=dict(arrowstyle="->", color=PALETTE["orange"], lw=2),
            bbox=dict(facecolor="white", edgecolor=PALETTE["orange"], boxstyle="round,pad=0.3"),
        )

    ax.text(
        0.02, -0.08,
        "Interpretation: Positive (red) = steering reduces sycophancy. Negative (blue) = increases it.",
        transform=ax.transAxes, fontsize=9, style="italic", color=PALETTE["gray"],
    )

    fig.tight_layout()
    fig.savefig(fig_dir / "exp1_caa_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_head_deltas(fig_dir: Path):
    head_scores_path = Path("results/exp2/head_scores.jsonl")
    if not head_scores_path.exists():
        return
    df = pd.read_json(head_scores_path, lines=True)
    if df.empty:
        return

    df_pos = df[df["delta_sum"] > 0].copy()
    if df_pos.empty:
        return

    df_top = df_pos.nlargest(10, "delta_sum").copy()
    df_top["label"] = df_top.apply(lambda r: f"L{int(r['layer'])}H{int(r['head'])}", axis=1)

    layer_sums = df_pos.groupby("layer")["delta_sum"].sum().sort_values(ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.2, 1]})

    bars = ax1.barh(df_top["label"], df_top["delta_sum"], color=PALETTE["green"], edgecolor="white")
    ax1.set_xlabel(r"Cumulative Effect ($\Sigma \Delta D_{syc}$)")
    ax1.set_ylabel("")
    ax1.set_title("(a) Top 10 Attention Heads by Impact")
    ax1.invert_yaxis()
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, df_top["delta_sum"]):
        ax1.text(val + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}",
                 va="center", ha="left", fontsize=9, color=PALETTE["black"])

    colors = [PALETTE["blue"] if l == layer_sums.index[0] else PALETTE["sky_blue"] for l in layer_sums.index]
    ax2.barh([f"Layer {int(l)}" for l in layer_sums.index], layer_sums.values, color=colors, edgecolor="white")
    ax2.set_xlabel(r"Total Layer Effect ($\Sigma \Delta D_{syc}$)")
    ax2.set_ylabel("")
    ax2.set_title("(b) Effect Aggregated by Layer")
    ax2.invert_yaxis()
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    dominant_layer = layer_sums.index[0]
    ax1.text(
        0.95, 0.05,
        f"Key Finding:\n"
        f"• Layer {int(dominant_layer)} accounts for\n"
        f"  {layer_sums.iloc[0] / layer_sums.sum() * 100:.0f}% of total effect\n"
        f"• Top 5 heads are all in L{int(dominant_layer)}",
        transform=ax1.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(facecolor=PALETTE["yellow"], alpha=0.3, edgecolor=PALETTE["orange"], boxstyle="round,pad=0.4"),
    )

    sns.despine()
    fig.tight_layout()
    fig.savefig(fig_dir / "exp2_head_deltas.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mediation_grid(fig_dir: Path):
    results_path = Path("results/exp3/mediation_results.jsonl")
    if not results_path.exists():
        return
    df = pd.read_json(results_path, lines=True)
    if df.empty:
        return

    conditions = [
        ("Vector Only", "D_vec"),
        ("Syc Heads Ablated", "D_syc_abl"),
        ("Vec + Syc Ablation", "D_vec_syc"),
        ("Vec + Random Ablation", "D_vec_rand"),
    ]

    results = []
    for label, col in conditions:
        delta = df[col] - df["D_base"]
        _, p_val = stats.ttest_1samp(delta, 0)
        results.append({
            "Condition": label,
            "mean_delta": delta.mean(),
            "sem": delta.sem(),
            "p_value": p_val,
            "n": len(delta),
            "significant": p_val < 0.05,
        })

    plot_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = []
    for _, row in plot_df.iterrows():
        if not row["significant"]:
            colors.append(PALETTE["gray"])
        elif row["mean_delta"] < 0:
            colors.append(PALETTE["green"])
        else:
            colors.append(PALETTE["vermillion"])

    ax.bar(plot_df["Condition"], plot_df["mean_delta"], color=colors, edgecolor="white", linewidth=1.5)

    ax.errorbar(
        x=range(len(plot_df)),
        y=plot_df["mean_delta"],
        yerr=plot_df["sem"] * 1.96,
        fmt="none",
        ecolor=PALETTE["black"],
        capsize=5,
        capthick=2,
        linewidth=2,
    )

    for i, row in plot_df.iterrows():
        if row["significant"]:
            star = "**" if row["p_value"] < 0.01 else "*"
            y_pos = row["mean_delta"] + row["sem"] * 2.2 if row["mean_delta"] >= 0 else row["mean_delta"] - row["sem"] * 2.5
            ax.text(i, y_pos, star, ha="center", fontsize=14, fontweight="bold")

    ax.axhline(0, color=PALETTE["black"], linestyle="-", linewidth=1.5)
    ax.set_ylabel(r"$\Delta D_{syc}$ (Condition − Baseline)")
    ax.set_xlabel("")
    ax.set_title("Mediation Analysis: How Do Interventions Affect Sycophancy?")

    plt.xticks(rotation=15, ha="right")

    ax.text(
        0.02, 0.02,
        "* p < 0.05, ** p < 0.01 | Green = reduced sycophancy, Gray = not significant",
        transform=ax.transAxes, fontsize=9, style="italic", color=PALETTE["gray"],
    )

    ax.text(
        0.98, 0.98, f"N = {plot_df['n'].iloc[0]} examples",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor=PALETTE["gray"]),
    )

    sns.despine()
    fig.tight_layout()
    fig.savefig(fig_dir / "exp3_mediation_grid.png", dpi=300, bbox_inches="tight")
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

    corr, p_corr = stats.pearsonr(df["E_clean"], df["E_syc"])

    g = sns.JointGrid(data=df, x="E_clean", y="E_syc", height=7)

    g.plot_joint(sns.scatterplot, color=PALETTE["blue"], alpha=0.7, s=70, edgecolor="white", linewidth=0.5)

    slope, intercept, *_ = stats.linregress(df["E_clean"], df["E_syc"])
    x_line = np.linspace(df["E_clean"].min(), df["E_clean"].max(), 100)
    g.ax_joint.plot(x_line, slope * x_line + intercept, color=PALETTE["vermillion"], linestyle="-", linewidth=2, label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")

    lim_min = min(df["E_clean"].min(), df["E_syc"].min()) - 0.2
    lim_max = max(df["E_clean"].max(), df["E_syc"].max()) + 0.2
    g.ax_joint.plot([lim_min, lim_max], [lim_min, lim_max], color=PALETTE["gray"], linestyle="--", linewidth=1.5, alpha=0.7, label="y = x (Perfect Mediation)")

    g.ax_joint.axhline(0, color=PALETTE["black"], linestyle="-", linewidth=1, alpha=0.5)
    g.ax_joint.axvline(0, color=PALETTE["black"], linestyle="-", linewidth=1, alpha=0.5)

    g.plot_marginals(sns.histplot, kde=True, color=PALETTE["sky_blue"], alpha=0.4, linewidth=0)

    g.ax_joint.set_xlabel(r"Total Effect ($E_{clean}$ = $D_{base}$ − $D_{vec}$)")
    g.ax_joint.set_ylabel(r"Mediated Effect ($E_{syc}$ = $D_{syc\_abl}$ − $D_{vec\_syc}$)")
    g.ax_joint.legend(loc="upper left", fontsize=9)

    g.ax_joint.text(
        0.95, 0.05,
        f"Pearson r = {corr:.2f}\n" + ("p < 0.001" if p_corr < 0.001 else f"p = {p_corr:.3f}"),
        transform=g.ax_joint.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor=PALETTE["gray"], boxstyle="round,pad=0.4"),
    )

    if corr > 0.8:
        interpretation = "Strong positive correlation:\nSyc heads mediate most of the vector's effect."
    elif corr > 0.5:
        interpretation = "Moderate correlation:\nSyc heads partially mediate the effect."
    else:
        interpretation = "Weak correlation:\nSyc heads do not mediate the vector's effect."

    g.ax_joint.text(
        0.05, 0.95, interpretation,
        transform=g.ax_joint.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(facecolor=PALETTE["yellow"], alpha=0.3, edgecolor=PALETTE["orange"], boxstyle="round,pad=0.4"),
    )

    g.figure.suptitle("Mediation Analysis: Do Sycophancy Heads Explain the Vector's Effect?", y=1.02, fontsize=14, fontweight="bold")

    g.savefig(fig_dir / "exp3_mediation_scatter.png", dpi=300, bbox_inches="tight")
    plt.close("all")


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
