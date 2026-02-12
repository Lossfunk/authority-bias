#!/usr/bin/env python3
"""Figure 2: Authority-weighted compliance hierarchy.

Horizontal dot chart showing Expert > Note > User > Online for both
Llama and Qwen, with gradient-coloured dots encoding effect magnitude.

Run:
    uv run python -m src.paper_figures.fig_authority
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from src.paper_figures.theme import (
    PAL, FIGSIZE_2x1, CMAP_SEQUENTIAL, apply_theme, save_fig,
    label_panel, add_y_grid, ci95,
)


def load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def plot_panel(ax, inst, base_data, title, show_y=True):
    tags = ["Expert", "Note", "User", "Online"]
    tag_keys = {"Expert": "Expert", "Note": "Note", "User": "User",
                "Online": "Someone online"}

    inst_eff = inst.get("per_tag_endorse_effect", {})
    base_eff = base_data.get("per_tag_endorse_effect", {})

    def get(effects, tag):
        for k in [tag, tag_keys.get(tag, tag)]:
            if k in effects:
                return effects[k]
        return None

    y = np.arange(len(tags))[::-1]

    # Gradient norm based on max effect for colour encoding
    all_means = []
    for tag in tags:
        d = get(inst_eff, tag)
        if d:
            all_means.append(d["mean"])
    vmax = max(all_means) if all_means else 3
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    for model_eff, marker, label, y_off, is_inst in [
        (inst_eff, "o", "Instruct", 0.12, True),
        (base_eff, "D", "Base", -0.12, False),
    ]:
        means, ci_lo, ci_hi, colours = [], [], [], []
        for tag in tags:
            d = get(model_eff, tag)
            if d:
                m = d["mean"]
                n = d.get("n_positive", 0) + d.get("n_negative", 0)
                lo, hi = ci95(m, d["std"], n)
            else:
                m, lo, hi = 0, 0, 0
            means.append(m)
            ci_lo.append(lo)
            ci_hi.append(hi)
            colours.append(CMAP_SEQUENTIAL(norm(m)) if is_inst else PAL.faint_gray)

        means_a = np.array(means)
        yi = y + y_off

        # Stems
        for i in range(len(tags)):
            ax.plot([0, means_a[i]], [yi[i], yi[i]],
                    color=colours[i] if is_inst else PAL.faint_gray,
                    alpha=0.4, linewidth=1.5, zorder=1)

        # Error bars
        ax.errorbar(means_a, yi,
                     xerr=[means_a - np.array(ci_lo), np.array(ci_hi) - means_a],
                     fmt="none", ecolor=PAL.medium_gray, elinewidth=0.7,
                     capsize=2.5, capthick=0.5, alpha=0.4, zorder=2)

        # Dots
        ax.scatter(means_a, yi, s=50, c=colours, marker=marker,
                   edgecolors=PAL.bg_warm, linewidths=0.8, zorder=3, label=label)

        # Value annotations
        for i, (m, yy) in enumerate(zip(means_a, yi)):
            col = colours[i] if is_inst else PAL.medium_gray
            ax.annotate(f"{m:.2f}", xy=(m, yy), xytext=(6, 0),
                        textcoords="offset points", fontsize=7,
                        fontweight="bold", color=col, va="center")

    ax.axvline(0, color=PAL.faint_gray, lw=0.6, ls="-", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(tags, fontsize=9, color=PAL.dark_text)
    ax.set_xlabel("Endorsement effect (delta log-odds)")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", fontsize=7, markerscale=0.8)
    ax.grid(axis="x", alpha=0.15, linewidth=0.4, color=PAL.faint_gray)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("new-phase-results/figures/paper"))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()
    apply_theme()

    b = Path("new-phase-results")
    li = load_json(b / "llama/exp8_speakers/meta-llama__Llama-3.1-8B-Instruct_summary.json")
    lb = load_json(b / "llama/exp8_speakers/meta-llama__Llama-3.1-8B_summary.json")
    qi = load_json(b / "qwen/exp8_speakers/Qwen__Qwen3-4B-Instruct-2507_summary.json")
    qb = load_json(b / "qwen/exp8_speakers/Qwen__Qwen3-4B_summary.json")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIGSIZE_2x1, sharey=True)
    plot_panel(a1, li, lb, "Llama-3.1-8B")
    plot_panel(a2, qi, qb, "Qwen3-4B", show_y=False)
    label_panel(a1, "A")
    label_panel(a2, "B")

    # No suptitle — hierarchy differs between models; let panels speak for themselves

    for p in save_fig(fig, "fig2_authority_hierarchy", args.output_dir, args.formats):
        print(f"  {p}")


if __name__ == "__main__":
    main()
