#!/usr/bin/env python3
"""Figure 4: Temporal dynamics — pressure accumulation and persistence.

Two-panel figure from exp12_k, editorial style with warm background and
gradient-tinted confidence bands.

Run:
    uv run python -m src.paper_figures.fig_temporal
    uv run python -m src.paper_figures.fig_temporal --tag note
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.theme import (
    PAL, FIGSIZE_2x1, apply_theme, save_fig,
    label_panel, add_y_grid, plot_with_band, annotate_endpoint, ci95,
    model_result_path,
)


K_VALUES = [1, 2, 5, 10, 20]


def load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def pressure_series(s, tag, ik="instr_0"):
    met = s["metrics"][tag.lower()][ik]
    init = met["initial"]["immediate_wrong_shift"]
    m = [init["mean"]]
    lo = [ci95(init["mean"], init["std"], init["n"])[0]]
    hi = [ci95(init["mean"], init["std"], init["n"])[1]]
    for k in K_VALUES:
        d = met["pressure"][f"K{k}"]["pressure_wrong_shift"]
        m.append(d["mean"])
        l, h = ci95(d["mean"], d["std"], d["n"])
        lo.append(l)
        hi.append(h)
    return m, lo, hi


def washout_series(s, tag, ik="instr_0"):
    pr = s["metrics"][tag.lower()][ik]["probes"]["context"]["same"]
    m, lo, hi = [], [], []
    for k in K_VALUES:
        d = pr[f"K{k}"]["washout_score_wrong"]
        m.append(d["mean"])
        l, h = ci95(d["mean"], d["std"], d["n"])
        lo.append(l)
        hi.append(h)
    return m, lo, hi


def plot_pressure(ax, llama, qwen, tag):
    x = np.arange(6)
    xl = ["0", "1", "2", "5", "10", "20"]

    for sm, sk in [(llama, "llama"), (qwen, "qwen")]:
        sty = PAL.MODEL_STYLES[sk]
        c = sty["color"]

        # Solid: no instruction
        m, lo, hi = pressure_series(sm, tag, "instr_0")
        plot_with_band(ax, x, np.array(m), np.array(lo), np.array(hi),
                       color=c, marker=sty["marker"], label=sty["short"])

        # Dashed: with instruction
        mi, loi, hii = pressure_series(sm, tag, "instr_1")
        ax.plot(x, mi, color=c, marker=sty["marker"], markersize=4,
                markeredgecolor=PAL.bg_warm, markeredgewidth=0.6,
                linewidth=1.2, linestyle="--", alpha=0.55, zorder=3,
                label=f"{sty['short']} + instr.")
        ax.fill_between(x, loi, hii, color=c, alpha=0.06, linewidth=0)

    ax.axhline(0, color=PAL.faint_gray, lw=0.7, ls="-", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(xl)
    ax.set_xlabel("K (number of wrong endorsements)")
    ax.set_ylabel("Pressure shift (log-odds)\nlower = more susceptible")
    ax.set_title(f"Pressure accumulation ({tag.capitalize()})", loc="left")
    # Legend handled at figure level
    add_y_grid(ax)


def plot_washout(ax, llama, qwen, tag):
    x = np.arange(5)
    xl = ["1", "2", "5", "10", "20"]

    for sm, sk in [(llama, "llama"), (qwen, "qwen")]:
        sty = PAL.MODEL_STYLES[sk]
        c = sty["color"]

        m, lo, hi = washout_series(sm, tag, "instr_0")
        plot_with_band(ax, x, np.array(m), np.array(lo), np.array(hi),
                       color=c, marker=sty["marker"], label=sty["short"])

        mi, loi, hii = washout_series(sm, tag, "instr_1")
        ax.plot(x, mi, color=c, marker=sty["marker"], markersize=4,
                markeredgecolor=PAL.bg_warm, markeredgewidth=0.6,
                linewidth=1.2, linestyle="--", alpha=0.55, zorder=3,
                label=f"{sty['short']} + instr.")
        ax.fill_between(x, loi, hii, color=c, alpha=0.06, linewidth=0)

        annotate_endpoint(ax, x[-1], m[-1], f"{m[-1]:.2f}", color=c,
                          offset=(6, 0), fontsize=7)

    # Reference lines
    ax.axhline(1.0, color=PAL.faint_gray, lw=0.6, ls=":", alpha=0.5)
    ax.axhline(0, color=PAL.faint_gray, lw=0.6, ls=":", alpha=0.5)
    ax.text(x[-1] + 0.5, 1.0, "Full washout", fontsize=6.5,
            color=PAL.light_gray, va="bottom")
    ax.text(x[-1] + 0.5, 0.0, "Full persistence", fontsize=6.5,
            color=PAL.light_gray, va="top")

    ax.set_ylim(-0.08, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(xl)
    ax.set_xlabel("K (number of wrong endorsements)")
    ax.set_ylabel("In-context washout score\n(0 = persists, 1 = washes out)")
    ax.set_title(f"Persistence after pressure ({tag.capitalize()})", loc="left")
    # Legend handled at figure level
    add_y_grid(ax)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="expert", choices=["expert", "note"])
    parser.add_argument("--output-dir", type=Path, default=Path("new-phase-results/figures/paper"))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()
    apply_theme()

    b = Path("new-phase-results")
    llama = load_json(model_result_path(
        b, "llama", "exp12_k/meta-llama__Llama-3.1-8B-Instruct_summary.json"
    ))
    qwen = load_json(model_result_path(
        b, "qwen", "exp12_k/Qwen__Qwen3-4B-Instruct-2507_summary.json"
    ))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGSIZE_2x1[0], FIGSIZE_2x1[1] + 0.4),
                                    constrained_layout=False)
    fig.subplots_adjust(left=0.11, right=0.95, top=0.82, bottom=0.18,
                        wspace=0.35)

    plot_pressure(a1, llama, qwen, tag=args.tag)
    plot_washout(a2, llama, qwen, tag=args.tag)
    label_panel(a1, "A")
    label_panel(a2, "B")

    # Shared legend at the bottom center
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=6.5, columnspacing=1.2, handletextpad=0.5,
               bbox_to_anchor=(0.5, 0.04), frameon=False)

    fig.suptitle("Repeated endorsement pressure creates persistent in-context bias",
                 fontsize=10, fontweight="bold", y=0.93, color=PAL.dark_text)

    for p in save_fig(fig, f"fig4_temporal_{args.tag}", args.output_dir, args.formats):
        print(f"  {p}")


if __name__ == "__main__":
    main()
