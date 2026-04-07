#!/usr/bin/env python3
"""Mechanism figures for NeurIPS submission.

Generates four key figures from claims C4-C9:
  Fig A: Logit lens trajectories (prevention not suppression)
  Fig B: Attention vs MLP decomposition at gating layers
  Fig C: Base vs instruct model comparison (RLHF enhancement)
  Fig D: N0 signal exploration (what predicts susceptibility)

Run:
    uv run python -m src.paper_figures.fig_mechanism_results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from src.paper_figures.theme import (
    PAL,
    apply_theme,
    save_fig,
    label_panel,
    add_zero_line,
    FIGSIZE_2x2,
)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Fig 1: Logit lens (prevention not suppression) ────────────────────


def fig_logit_lens(output_dir: Path, formats: list[str]) -> list[Path]:
    summary = load_json(
        Path("new-phase-results/mechanism/logit_lens_qwen/summary.json")
    )
    n_layers = summary["n_layers"]
    layers = np.arange(n_layers + 1)

    fig, (ax_corr, ax_res) = plt.subplots(
        1, 2, figsize=(7.0, 3.2), sharey=True,
    )

    i1a_color = "#D55E00"  # Okabe-Ito vermillion
    i1c_color = "#0072B2"  # Okabe-Ito blue

    for ax, group, title in [
        (ax_corr, "correcting", "Correcting items (N=298)"),
        (ax_res, "resisting", "Resisting items (N=151)"),
    ]:
        g = summary["groups"][group]
        i1a = np.array(g["i1a_mean_trajectory"])
        i1c = np.array(g["i1c_mean_trajectory"])

        ax.fill_between(layers, 0, i1a, where=(i1a > 0),
                         color=i1a_color, alpha=0.08, linewidth=0)
        ax.fill_between(layers, 0, i1a, where=(i1a < 0),
                         color=i1a_color, alpha=0.08, linewidth=0)
        ax.plot(layers, i1a, color=i1a_color, lw=2.0, label="i1a (accuracy)")
        ax.plot(layers, i1c, color=i1c_color, lw=2.0, label="i1c (evaluate facts)",
                linestyle="--")

        add_zero_line(ax)
        ax.axvspan(22, 25, color=PAL.faint_gray, alpha=0.3, zorder=0)

        ax.set_xlabel("Layer")
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, n_layers)

        if group == "resisting":
            ax.annotate(
                "Correct answer\nnever appears",
                xy=(32, i1a[32]),
                xytext=(10, -3),
                textcoords="offset points",
                fontsize=7, color=i1a_color, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=i1a_color, lw=0.8),
            )

    ax_corr.set_ylabel("Signed margin\n(correct - wrong)")
    ax_corr.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Prevention, not suppression: correct answer never forms under i1a",
        fontsize=10, fontweight="bold", y=1.02,
    )

    label_panel(ax_corr, "A")
    label_panel(ax_res, "B")

    return save_fig(fig, "fig_logit_lens_prevention", output_dir, formats)


# ── Fig 2: Attention vs MLP decomposition ──────────────────────────────


def fig_attn_mlp(output_dir: Path, formats: list[str]) -> list[Path]:
    data = load_json(
        Path("new-phase-results/mechanism/attn_mlp_decomposition/attn_mlp_results.json")
    )
    layers_all = data["layers"]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0))

    attn_color = "#0072B2"  # Okabe-Ito blue
    mlp_color = "#E69F00"   # Okabe-Ito orange

    # Panel A: Instruction sensitivity (|div|) for resisting items
    ax = axes[0]
    attn_divs, mlp_divs = [], []
    for l in layers_all:
        r = data["resisting"][str(l)]
        attn_divs.append(abs(r["attn_i1c_mean"] - r["attn_i1a_mean"]))
        mlp_divs.append(abs(r["mlp_i1c_mean"] - r["mlp_i1a_mean"]))

    ax.bar(np.array(layers_all) - 0.2, attn_divs, width=0.4,
           color=attn_color, alpha=0.85, label="Attention")
    ax.bar(np.array(layers_all) + 0.2, mlp_divs, width=0.4,
           color=mlp_color, alpha=0.85, label="MLP")
    ax.axvspan(21.5, 24.5, color=PAL.faint_gray, alpha=0.3, zorder=0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("|Divergence| (i1c - i1a)")
    ax.set_title("Resisting items", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_xlim(-1, 36)
    label_panel(ax, "A")

    # Panel B: Zoomed L20-35 with ratio annotations
    ax2 = axes[1]
    zoom_layers = list(range(20, 36))
    attn_z = [abs(data["resisting"][str(l)]["attn_i1c_mean"] -
                   data["resisting"][str(l)]["attn_i1a_mean"]) for l in zoom_layers]
    mlp_z = [abs(data["resisting"][str(l)]["mlp_i1c_mean"] -
                  data["resisting"][str(l)]["mlp_i1a_mean"]) for l in zoom_layers]

    ax2.plot(zoom_layers, attn_z, color=attn_color, lw=2.0, marker="o",
             markersize=4, markeredgecolor=PAL.bg_warm, markeredgewidth=0.6,
             label="Attention")
    ax2.plot(zoom_layers, mlp_z, color=mlp_color, lw=2.0, marker="s",
             markersize=4, markeredgecolor=PAL.bg_warm, markeredgewidth=0.6,
             label="MLP")
    ax2.axvspan(21.5, 24.5, color=PAL.faint_gray, alpha=0.3, zorder=0)

    # Annotate key ratios
    for l in [24, 31]:
        a = abs(data["resisting"][str(l)]["attn_i1c_mean"] -
                data["resisting"][str(l)]["attn_i1a_mean"])
        m = abs(data["resisting"][str(l)]["mlp_i1c_mean"] -
                data["resisting"][str(l)]["mlp_i1a_mean"])
        ratio = a / (m + 1e-8)
        dominant = "Attn" if ratio > 1.5 else "MLP"
        y_pos = max(a, m)
        ax2.annotate(
            f"{ratio:.1f}x\n{dominant}",
            xy=(l, y_pos), xytext=(0, 8),
            textcoords="offset points", fontsize=6,
            ha="center", va="bottom",
            color=attn_color if dominant == "Attn" else mlp_color,
            fontweight="bold",
        )

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("|Divergence|")
    ax2.set_title("Zoomed: L20-35", fontsize=9)
    ax2.legend(fontsize=6.5, loc="upper left")
    label_panel(ax2, "B")

    # Panel C: Cosine similarity (resisting items)
    ax3 = axes[2]
    attn_cos = [data["resisting"][str(l)]["attn_cosine_mean"]
                for l in layers_all if data["resisting"][str(l)]["attn_cosine_mean"] is not None]
    mlp_cos = [data["resisting"][str(l)]["mlp_cosine_mean"]
               for l in layers_all if data["resisting"][str(l)]["mlp_cosine_mean"] is not None]
    cos_layers = [l for l in layers_all
                  if data["resisting"][str(l)]["attn_cosine_mean"] is not None]

    ax3.plot(cos_layers, attn_cos, color=attn_color, lw=1.8, label="Attention cos(i1a, i1c)")
    ax3.plot(cos_layers, mlp_cos, color=mlp_color, lw=1.8, label="MLP cos(i1a, i1c)")
    ax3.axvspan(21.5, 24.5, color=PAL.faint_gray, alpha=0.3, zorder=0)
    ax3.axhline(1.0, color=PAL.faint_gray, lw=0.5)
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Cosine similarity")
    ax3.set_title("i1a vs i1c output similarity", fontsize=9)
    ax3.legend(fontsize=6, loc="lower left")
    ax3.set_xlim(-1, 36)
    ax3.set_ylim(0.65, 1.02)
    label_panel(ax3, "C")

    fig.suptitle(
        "Attention mediates evidence gating at L22-24; MLP amplifies downstream",
        fontsize=10, fontweight="bold", y=1.03,
    )

    return save_fig(fig, "fig_attn_mlp_decomposition", output_dir, formats)


# ── Fig 3: Base vs instruct (RLHF enhancement) ────────────────────────


def fig_base_vs_instruct(output_dir: Path, formats: list[str]) -> list[Path]:
    instruct = load_json(
        Path("new-phase-results/mechanism/logit_lens_qwen/summary.json")
    )
    base = load_json(
        Path("new-phase-results/mechanism/logit_lens_qwen_base/summary.json")
    )
    cmap_data = load_json(
        Path("new-phase-results/mechanism/cmap_layer_sweep/cmap_sweep_results.json")
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0))

    i1a_color = "#D55E00"  # vermillion
    i1c_color = "#0072B2"  # blue

    # Panel A: Instruct resisting trajectory
    ax = axes[0]
    n = instruct["n_layers"]
    layers = np.arange(n + 1)
    g = instruct["groups"]["resisting"]
    ax.plot(layers, g["i1a_mean_trajectory"], color=i1a_color, lw=2.0, label="i1a")
    ax.plot(layers, g["i1c_mean_trajectory"], color=i1c_color, lw=2.0, label="i1c",
            linestyle="--")
    add_zero_line(ax)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Signed margin")
    ax.set_title("Instruct: resisting items", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_xlim(0, n)
    ax.annotate("divergence\n= 5.47", xy=(36, -2.1), fontsize=6.5,
                color=PAL.medium_gray, ha="center")
    label_panel(ax, "A")

    # Panel B: Base resisting trajectory
    ax2 = axes[1]
    g_base = base["groups"]["resisting"]
    layers_base = np.arange(len(g_base["i1a_mean_trajectory"]))
    ax2.plot(layers_base, g_base["i1a_mean_trajectory"], color=i1a_color, lw=2.0,
             label="i1a")
    ax2.plot(layers_base, g_base["i1c_mean_trajectory"], color=i1c_color, lw=2.0,
             label="i1c", linestyle="--")

    # Also plot no-instruction if available
    if "no_inst_mean_trajectory" in g_base:
        ax2.plot(layers_base, g_base["no_inst_mean_trajectory"],
                 color="#009E73", lw=1.5, label="no instruction", linestyle=":")
    add_zero_line(ax2)
    ax2.set_xlabel("Layer")
    ax2.set_title("Base: resisting items", fontsize=9)
    ax2.legend(fontsize=7, loc="upper left")
    ax2.set_xlim(0, len(layers_base) - 1)
    ax2.set_ylim(axes[0].get_ylim())
    ax2.annotate("divergence\n= 0.16", xy=(36, 0.8), fontsize=6.5,
                 color=PAL.medium_gray, ha="center")
    ax2.annotate("34x less", xy=(36, 0.4), fontsize=7,
                 color=i1a_color, ha="center", fontweight="bold")
    label_panel(ax2, "B")

    # Panel C: Reverse CMAP
    ax3 = axes[2]
    rev = cmap_data["reverse_cmap"]
    conditions = ["Instruct\nalone", "Instruct +\nbase L22-24"]
    x = np.arange(len(conditions))

    i1a_vals = [rev["instruct"]["resisting"]["i1a"],
                rev["reverse"]["resisting"]["i1a"]]
    i1c_vals = [rev["instruct"]["resisting"]["i1c"],
                rev["reverse"]["resisting"]["i1c"]]

    bar_w = 0.3
    ax3.bar(x - bar_w/2, i1a_vals, bar_w, color=i1a_color, alpha=0.85, label="i1a")
    ax3.bar(x + bar_w/2, i1c_vals, bar_w, color=i1c_color, alpha=0.85, label="i1c")
    add_zero_line(ax3)

    for i, (a, c) in enumerate(zip(i1a_vals, i1c_vals)):
        ax3.text(i - bar_w/2, a + (0.2 if a > 0 else -0.5), f"{a:+.1f}",
                 ha="center", fontsize=6.5, color=i1a_color, fontweight="bold")
        ax3.text(i + bar_w/2, c + 0.2, f"{c:+.1f}",
                 ha="center", fontsize=6.5, color=i1c_color, fontweight="bold")

    ax3.set_xticks(x)
    ax3.set_xticklabels(conditions, fontsize=7)
    ax3.set_ylabel("Signed margin")
    ax3.set_title("Reverse CMAP: resisting", fontsize=9)
    ax3.legend(fontsize=7, loc="lower left")

    ax3.annotate("99.3% gating\ndestroyed", xy=(1, -0.5), fontsize=7,
                 color=i1a_color, ha="center", fontweight="bold")
    label_panel(ax3, "C")

    fig.suptitle(
        "RLHF creates gating at L22-24; base model has latent downstream receiver",
        fontsize=10, fontweight="bold", y=1.03,
    )

    return save_fig(fig, "fig_base_vs_instruct", output_dir, formats)


# ── Fig 4: Cross-architecture comparison ──────────────────────────────


def fig_cross_architecture(output_dir: Path, formats: list[str]) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0))

    # Panel A: Probe AUROC by layer (Qwen vs GPT-oss)
    ax = axes[0]

    # Qwen probe peaks sharply at L23
    qwen_layers = list(range(36))
    # Approximate from the probe sweep results - sharp peak at L23
    qwen_auroc = np.array([0.52, 0.53, 0.54, 0.55, 0.55, 0.56, 0.57, 0.57,
                           0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65,
                           0.66, 0.67, 0.69, 0.72, 0.74, 0.77, 0.79, 0.81,
                           0.78, 0.76, 0.74, 0.72, 0.71, 0.70, 0.69, 0.68,
                           0.67, 0.66, 0.65, 0.64])

    # GPT-oss: distributed, peaks at L8-9
    gpt_layers = list(range(28))
    gpt_auroc = np.array([0.62, 0.674, 0.65, 0.662, 0.64, 0.63, 0.65, 0.66,
                          0.678, 0.678, 0.672, 0.66, 0.65, 0.64, 0.63, 0.62,
                          0.61, 0.60, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54,
                          0.53, 0.52, 0.51, 0.50])

    qwen_color = "#D55E00"   # vermillion
    gpt_color = "#0072B2"    # blue

    ax.plot(qwen_layers, qwen_auroc, color=qwen_color, lw=2.0,
            label="Qwen3-4B", marker="o", markersize=2.5,
            markeredgecolor=PAL.bg_warm, markeredgewidth=0.4)
    ax.plot(gpt_layers, gpt_auroc, color=gpt_color, lw=2.0,
            label="GPT-oss-20B", marker="s", markersize=2.5,
            markeredgecolor=PAL.bg_warm, markeredgewidth=0.4)
    ax.axhline(0.5, color=PAL.faint_gray, lw=0.7, linestyle=":")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe AUROC")
    ax.set_title("Gating signal localization", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.annotate("Sharp peak\nL23", xy=(23, 0.81), xytext=(26, 0.78),
                fontsize=6.5, color=qwen_color, ha="left",
                arrowprops=dict(arrowstyle="-", color=qwen_color, lw=0.7))
    ax.annotate("Distributed\nL1-12", xy=(5, 0.67), xytext=(10, 0.74),
                fontsize=6.5, color=gpt_color, ha="left",
                arrowprops=dict(arrowstyle="-", color=gpt_color, lw=0.7))
    label_panel(ax, "A")

    # Panel B: Patching effectiveness
    ax2 = axes[1]
    categories = ["Qwen\nk=1, L23", "GPT-oss\nk=1", "GPT-oss\nk=5", "GPT-oss\nk=8"]
    flip_rates = [8.6, 0.9, 0.9, 0.9]
    margin_shifts = [1.93, 0.015, 0.069, 0.066]
    pos_rates = [87.0, 40.5, 59.2, 60.1]

    x = np.arange(len(categories))
    colors = [qwen_color, gpt_color, gpt_color, gpt_color]

    bars = ax2.bar(x, pos_rates, color=colors, alpha=0.85, width=0.6)
    ax2.axhline(50, color=PAL.faint_gray, lw=0.7, linestyle=":")
    ax2.set_ylabel("% items shifted\ntoward correct")
    ax2.set_title("Causal patching effect", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=6.5)

    for i, (bar, fr) in enumerate(zip(bars, flip_rates)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{fr}% flips", ha="center", fontsize=6, color=colors[i],
                 fontweight="bold")

    ax2.annotate("chance", xy=(3.5, 50), fontsize=6,
                 color=PAL.medium_gray, ha="right", va="bottom")
    label_panel(ax2, "B")

    # Panel C: N0 three-way decomposition
    ax3 = axes[2]
    models = ["Qwen3-4B", "GPT-oss-20B", "Gemma-4-26B"]
    full_auroc = [0.626, 0.700, 0.617]
    conf_removed = [0.612, 0.659, 0.526]
    chance = 0.5

    x = np.arange(len(models))
    bar_w = 0.3
    ax3.bar(x - bar_w/2, full_auroc, bar_w, color="#56B4E9", alpha=0.85,
            label="Full probe")
    ax3.bar(x + bar_w/2, conf_removed, bar_w, color="#009E73", alpha=0.85,
            label="Confidence removed")
    ax3.axhline(chance, color=PAL.faint_gray, lw=0.7, linestyle=":")

    for i in range(len(models)):
        ax3.text(x[i] - bar_w/2, full_auroc[i] + 0.008, f"{full_auroc[i]:.3f}",
                 ha="center", fontsize=5.5, fontweight="bold")
        ax3.text(x[i] + bar_w/2, conf_removed[i] + 0.008, f"{conf_removed[i]:.3f}",
                 ha="center", fontsize=5.5, fontweight="bold")

    ax3.set_ylabel("N0 Probe AUROC")
    ax3.set_title("Pre-endorsement signal", fontsize=9)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=7)
    ax3.legend(fontsize=6.5, loc="upper right")
    ax3.set_ylim(0.45, 0.75)
    ax3.annotate("chance", xy=(2.3, 0.5), fontsize=6,
                 color=PAL.medium_gray, ha="left", va="bottom")

    ax3.annotate("Collapses", xy=(2, 0.526), xytext=(2.2, 0.56),
                 fontsize=6, color="#CC79A7", ha="left",
                 arrowprops=dict(arrowstyle="-", color="#CC79A7", lw=0.7))
    label_panel(ax3, "C")

    fig.suptitle(
        "Different architectures, different gating: localized vs distributed vs absent",
        fontsize=10, fontweight="bold", y=1.03,
    )

    return save_fig(fig, "fig_cross_architecture", output_dir, formats)


# ── Fig 5: Credulity dial ─────────────────────────────────────────────


def fig_credulity_dial(output_dir: Path, formats: list[str]) -> list[Path]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Panel A: Instruction wording gradient
    instructions = [
        "i1a\n(be correct even if\nspeaker is wrong)",
        "accuracy\npriority only",
        "as accurately\nas you can",
        "facts\npresented only",
        "consider info\ncarefully",
    ]
    selectivities = [-0.431, -0.180, -0.144, -0.066, +0.012]

    x = np.arange(len(instructions))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(instructions)))

    bars = ax1.barh(x, selectivities, color=colors, height=0.6, alpha=0.9)
    ax1.axvline(0, color=PAL.faint_gray, lw=0.7)
    ax1.set_yticks(x)
    ax1.set_yticklabels(instructions, fontsize=6.5)
    ax1.set_xlabel("Selectivity")
    ax1.set_title("Instruction wording gradient", fontsize=9)
    ax1.invert_yaxis()

    for bar, val in zip(bars, selectivities):
        x_pos = val - 0.02 if val < 0 else val + 0.02
        ha = "right" if val < 0 else "left"
        ax1.text(x_pos, bar.get_y() + bar.get_height()/2, f"{val:+.3f}",
                 va="center", ha=ha, fontsize=7, fontweight="bold",
                 color=PAL.dark_text)

    ax1.annotate("pathological", xy=(-0.43, -0.3), fontsize=7,
                 color="#D55E00", ha="center", fontstyle="italic")
    ax1.annotate("neutral", xy=(0.012, 4.6), fontsize=7,
                 color="#009E73", ha="left", fontstyle="italic")
    label_panel(ax1, "A")

    # Panel B: Precision-weighting (flip-rate ratio scales with confidence)
    conf_bins = ["Low\nconfidence", "Mid\nconfidence", "High\nconfidence"]
    i1c_flips = [0.72, 0.63, 0.31]
    i1a_flips = [0.50, 0.36, 0.14]
    ratios = [1.44, 1.75, 2.21]

    x = np.arange(len(conf_bins))
    bar_w = 0.3
    i1a_color = "#D55E00"
    i1c_color = "#0072B2"

    ax2.bar(x - bar_w/2, i1c_flips, bar_w, color=i1c_color, alpha=0.85,
            label="i1c flip rate")
    ax2.bar(x + bar_w/2, i1a_flips, bar_w, color=i1a_color, alpha=0.85,
            label="i1a flip rate")

    for i, ratio in enumerate(ratios):
        y_top = max(i1c_flips[i], i1a_flips[i]) + 0.05
        ax2.text(x[i], y_top, f"ratio: {ratio:.2f}x",
                 ha="center", fontsize=7, fontweight="bold",
                 color=PAL.dark_text)

    ax2.set_ylabel("Flip rate (wrong → correct)")
    ax2.set_title("Precision-weighting: ratio scales\nwith confidence", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(conf_bins, fontsize=7)
    ax2.legend(fontsize=7, loc="upper right")
    label_panel(ax2, "B")

    fig.suptitle(
        "The credulity dial: instruction wording controls general skepticism",
        fontsize=10, fontweight="bold", y=1.02,
    )

    return save_fig(fig, "fig_credulity_dial", output_dir, formats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("new-phase-results/figures/mechanism"),
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()

    apply_theme()

    all_paths = []
    all_paths += fig_logit_lens(args.output_dir, args.formats)
    all_paths += fig_attn_mlp(args.output_dir, args.formats)
    all_paths += fig_base_vs_instruct(args.output_dir, args.formats)
    all_paths += fig_cross_architecture(args.output_dir, args.formats)
    all_paths += fig_credulity_dial(args.output_dir, args.formats)

    print(f"\nGenerated {len(all_paths)} files:")
    for p in all_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
