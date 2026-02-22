#!/usr/bin/env python3
"""Figure 1: Endorsement effects are real and dominate order effects.

Two-panel figure: (A) endorsement magnitude across 3 instruct models,
(B) order vs endorsement decomposition for all 3 models.

Run:
    uv run python -m src.paper_figures.fig_endorsement
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.theme import (
    PAL,
    FIGSIZE_2x1,
    apply_theme,
    save_fig,
    label_panel,
    add_zero_line,
    add_y_grid,
    draw_bar,
    draw_error_bar,
    ci95,
    model_result_path,
)


def load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def plot_effect(ax, llama_i, qwen_i, qwen_t):
    models = [
        ("Llama\nInstruct", llama_i, PAL.MODEL_STYLES["llama"]["color"]),
        ("Qwen\nInstruct", qwen_i, PAL.MODEL_STYLES["qwen"]["color"]),
        ("Qwen\nThinking", qwen_t, PAL.MODEL_STYLES["qwen_thinking"]["color"]),
    ]
    x = np.arange(len(models))
    for i, (lbl, data, c) in enumerate(models):
        d = data["delta_logit_wrong"]
        m = d["mean"]
        n = d.get("n_positive", 0) + d.get("n_negative", 0)
        lo, hi = ci95(m, d["std"], n)
        draw_bar(ax, x[i], m, width=0.62, color=c, alpha=0.9)
        draw_error_bar(ax, x[i], m, lo, hi, cap_width=0.1)
        ax.text(
            x[i],
            max(m, hi) + 0.06,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color=c,
        )

    add_zero_line(ax)
    add_y_grid(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models])
    ax.set_ylabel("Mean delta log-odds (wrong)")
    ax.set_title("Endorsement survives lexical control", loc="left")


def plot_decomp(ax, llama_e8, qwen_e8, qwen_t_e8):
    comps = ["order", "endorse", "contrast"]
    labels = ["Order\neffect", "Endorsement\neffect", "Contrast\nnegation"]
    x = np.arange(len(comps))
    bw = 0.22

    for mi, (data, key, lbl) in enumerate(
        [
            (llama_e8, "llama", "Llama-Instruct"),
            (qwen_e8, "qwen", "Qwen-Instruct"),
            (qwen_t_e8, "qwen_thinking", "Qwen-Thinking"),
        ]
    ):
        c = PAL.MODEL_STYLES[key]["color"]
        off = -bw - 0.03 + mi * (bw + 0.03)
        dec = data["decomposition"]
        for ci_idx, comp in enumerate(comps):
            d = dec[comp]
            m = d["mean"]
            n = d.get("n_positive", 0) + d.get("n_negative", 0)
            lo, hi = ci95(m, d["std"], n)
            draw_bar(
                ax,
                x[ci_idx] + off,
                m,
                width=bw,
                color=c,
                label=lbl if ci_idx == 0 else None,
            )
            draw_error_bar(ax, x[ci_idx] + off, m, lo, hi, cap_width=0.06)
            if comp == "endorse":
                ax.text(
                    x[ci_idx] + off,
                    max(m, hi) + 0.05,
                    f"{m:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                    color=c,
                )

    add_zero_line(ax)
    add_y_grid(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean log-odds shift")
    ax.set_title("Endorsement >> Order", loc="left")
    ax.legend(loc="upper right", fontsize=7)


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
            b, "llama", "exp7/meta-llama__Llama-3.1-8B-Instruct_summary.json"
        )
    )
    qi = load_json(
        model_result_path(b, "qwen", "exp7/Qwen__Qwen3-4B-Instruct-2507_summary.json")
    )
    qt = load_json(
        model_result_path(
            b, "qwen_thinking", "exp7/Qwen__Qwen3-4B-Thinking-2507_summary.json"
        )
    )
    le = load_json(
        model_result_path(
            b, "llama", "exp8/meta-llama__Llama-3.1-8B-Instruct_summary.json"
        )
    )
    qe = load_json(
        model_result_path(b, "qwen", "exp8/Qwen__Qwen3-4B-Instruct-2507_summary.json")
    )
    qte = load_json(
        model_result_path(
            b, "qwen_thinking", "exp8/Qwen__Qwen3-4B-Thinking-2507_summary.json"
        )
    )

    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIGSIZE_2x1)
    plot_effect(a1, li, qi, qt)
    plot_decomp(a2, le, qe, qte)
    label_panel(a1, "A")
    label_panel(a2, "B")

    for p in save_fig(fig, "fig1_endorsement_effect", args.output_dir, args.formats):
        print(f"  {p}")


if __name__ == "__main__":
    main()
