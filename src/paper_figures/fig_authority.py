#!/usr/bin/env python3
"""Figure 2: Authority-weighted compliance hierarchy.

Horizontal dot chart showing Expert > Note > User > Online across
Llama-Instruct, Qwen-Instruct, and Qwen-Thinking (3 panels, instruct only).

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
    PAL,
    CMAP_SEQUENTIAL,
    apply_theme,
    save_fig,
    label_panel,
    add_y_grid,
    ci95,
    model_result_path,
)


def load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def plot_panel(ax, inst, title, model_key):
    tags = ["Expert", "Note", "User", "Online"]
    tag_keys = {
        "Expert": "Expert",
        "Note": "Note",
        "User": "User",
        "Online": "Someone online",
    }

    inst_eff = inst.get("per_tag_endorse_effect", {})
    color = PAL.MODEL_STYLES[model_key]["color"]

    def get(effects, tag):
        for k in [tag, tag_keys.get(tag, tag)]:
            if k in effects:
                return effects[k]
        return None

    y = np.arange(len(tags))[::-1]

    # Gradient norm based on max instruct effect for colour encoding
    all_means = []
    for tag in tags:
        d = get(inst_eff, tag)
        if d:
            all_means.append(d["mean"])
    vmax = max(all_means) if all_means else 3
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    means, ci_lo, ci_hi, colours = [], [], [], []
    for tag in tags:
        d = get(inst_eff, tag)
        if d:
            m = d["mean"]
            n = d.get("n_positive", 0) + d.get("n_negative", 0)
            lo, hi = ci95(m, d["std"], n)
        else:
            m, lo, hi = 0, 0, 0
        means.append(m)
        ci_lo.append(lo)
        ci_hi.append(hi)
        colours.append(CMAP_SEQUENTIAL(norm(m)))

    means_a = np.array(means)

    # Stems
    for i in range(len(tags)):
        ax.plot(
            [0, means_a[i]],
            [y[i], y[i]],
            color=colours[i],
            alpha=0.4,
            linewidth=1.5,
            zorder=1,
        )

    # Error bars
    ax.errorbar(
        means_a,
        y,
        xerr=[means_a - np.array(ci_lo), np.array(ci_hi) - means_a],
        fmt="none",
        ecolor=PAL.medium_gray,
        elinewidth=0.7,
        capsize=2.5,
        capthick=0.5,
        alpha=0.4,
        zorder=2,
    )

    # Dots
    ax.scatter(
        means_a,
        y,
        s=50,
        c=colours,
        marker=PAL.MODEL_STYLES[model_key]["marker"],
        edgecolors=PAL.bg_warm,
        linewidths=0.8,
        zorder=3,
    )

    # Value annotations
    for i, (m, yy) in enumerate(zip(means_a, y)):
        ax.annotate(
            f"{m:.2f}",
            xy=(m, yy),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            color=colours[i],
            va="center",
        )

    ax.axvline(0, color=PAL.faint_gray, lw=0.6, ls="-", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(tags, fontsize=9, color=PAL.dark_text)
    ax.set_xlabel("Endorsement effect (delta log-odds)")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", alpha=0.15, linewidth=0.4, color=PAL.faint_gray)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("new-phase-results/figures/paper")
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()
    apply_theme()

    b = Path("new-phase-results")
    li = load_json(
        model_result_path(
            b, "llama", "exp8_speakers/meta-llama__Llama-3.1-8B-Instruct_summary.json"
        )
    )
    qi = load_json(
        model_result_path(
            b, "qwen", "exp8_speakers/Qwen__Qwen3-4B-Instruct-2507_summary.json"
        )
    )
    qt = load_json(
        model_result_path(
            b,
            "qwen_thinking",
            "exp8_speakers/Qwen__Qwen3-4B-Thinking-2507_summary.json",
        )
    )

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)
    plot_panel(a1, li, "Llama-3.1-8B-Instruct", "llama")
    plot_panel(a2, qi, "Qwen3-4B-Instruct", "qwen")
    plot_panel(a3, qt, "Qwen3-4B-Thinking", "qwen_thinking")
    label_panel(a1, "A")
    label_panel(a2, "B")
    label_panel(a3, "C")

    fig.tight_layout()

    for p in save_fig(fig, "fig2_authority_hierarchy", args.output_dir, args.formats):
        print(f"  {p}")


if __name__ == "__main__":
    main()
