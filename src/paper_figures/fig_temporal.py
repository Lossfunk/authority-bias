#!/usr/bin/env python3
"""Figure 4: Temporal dynamics — pressure accumulation and persistence.

Two-panel figure from exp12_k, editorial style with warm background and
gradient-tinted confidence bands.

Run:
    uv run python -m src.paper_figures.fig_temporal
    uv run python -m src.paper_figures.fig_temporal --tag note
    uv run python -m src.paper_figures.fig_temporal --tag expert --appendix-style  # ICML-style caption-ready (no titles, X-only labels)
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
    PAL,
    FIGSIZE_2x1,
    apply_theme,
    save_fig,
    label_panel,
    add_y_grid,
    plot_with_band,
    annotate_endpoint,
    ci95,
    model_result_path,
)


K_VALUES = [1, 2, 5, 10, 20]

MODELS = [
    {
        "key": "llama",
        "summary": "exp12_k/meta-llama__Llama-3.1-8B-Instruct_summary.json",
    },
    {
        "key": "qwen",
        "summary": "exp12_k/Qwen__Qwen3-4B-Instruct-2507_summary.json",
    },
    {
        "key": "qwen_thinking",
        "summary": "exp12_k/Qwen__Qwen3-4B-Thinking-2507_summary.json",
    },
]

# Line style per model (colour comes from PAL.MODEL_STYLES)
MODEL_LS = {"llama": "-", "qwen": "--", "qwen_thinking": ":"}


def load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def _tag_key(tag: str) -> str:
    """Normalize tag for metrics lookup: expert, note, user, someone_online."""
    return tag.lower().replace(" ", "_")


def _tag_display(tag: str) -> str:
    """Human-readable tag for titles: someone_online -> Someone online."""
    return tag.replace("_", " ").capitalize()


def pressure_series(s, tag, ik="instr_0"):
    met = s["metrics"][_tag_key(tag)][ik]
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
    pr = s["metrics"][_tag_key(tag)][ik]["probes"]["context"]["same"]
    m, lo, hi = [], [], []
    for k in K_VALUES:
        d = pr[f"K{k}"]["washout_score_wrong"]
        m.append(d["mean"])
        l, h = ci95(d["mean"], d["std"], d["n"])
        lo.append(l)
        hi.append(h)
    return m, lo, hi


def plot_pressure(ax, model_summaries, tag, *, appendix_style: bool = False):
    x = np.arange(6)
    xl = ["0", "1", "2", "5", "10", "20"]

    for sk, sm in model_summaries:
        sty = PAL.MODEL_STYLES[sk]
        c = sty["color"]
        ls = MODEL_LS[sk]

        m, lo, hi = pressure_series(sm, tag, "instr_0")
        plot_with_band(
            ax,
            x,
            np.array(m),
            np.array(lo),
            np.array(hi),
            color=c,
            marker=sty["marker"],
            label=sty["short"],
            linewidth=1.8,
            linestyle=ls,
        )

        mi, loi, hii = pressure_series(sm, tag, "instr_1")
        ax.plot(
            x,
            mi,
            color=c,
            marker=sty["marker"],
            markersize=4,
            markeredgecolor=PAL.bg_warm,
            markeredgewidth=0.6,
            linewidth=1.1,
            linestyle=":" if ls == ":" else (0, (4, 2)) if ls == "--" else (0, (6, 3)),
            alpha=0.45,
            zorder=3,
            label=f"{sty['short']} + instr.",
        )
        ax.fill_between(x, loi, hii, color=c, alpha=0.05, linewidth=0)

    ax.axhline(0, color=PAL.faint_gray, lw=0.7, ls="-", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(xl)
    ax.set_xlabel("K (number of wrong endorsements)")
    if not appendix_style:
        ax.set_ylabel("Pressure shift (log-odds)\nlower = more susceptible")
        ax.set_title(f"Pressure accumulation ({_tag_display(tag)})", loc="left")
    add_y_grid(ax)


def plot_washout(ax, model_summaries, tag, *, appendix_style: bool = False):
    x = np.arange(5)
    xl = ["1", "2", "5", "10", "20"]

    for sk, sm in model_summaries:
        sty = PAL.MODEL_STYLES[sk]
        c = sty["color"]
        ls = MODEL_LS[sk]

        m, lo, hi = washout_series(sm, tag, "instr_0")
        plot_with_band(
            ax,
            x,
            np.array(m),
            np.array(lo),
            np.array(hi),
            color=c,
            marker=sty["marker"],
            label=sty["short"],
            linewidth=1.8,
            linestyle=ls,
        )

        mi, loi, hii = washout_series(sm, tag, "instr_1")
        ax.plot(
            x,
            mi,
            color=c,
            marker=sty["marker"],
            markersize=4,
            markeredgecolor=PAL.bg_warm,
            markeredgewidth=0.6,
            linewidth=1.1,
            linestyle=ls,
            alpha=0.45,
            zorder=3,
            label=f"{sty['short']} + instr.",
        )
        ax.fill_between(x, loi, hii, color=c, alpha=0.05, linewidth=0)

        annotate_endpoint(
            ax, x[-1], m[-1], f"{m[-1]:.2f}", color=c, offset=(6, 0), fontsize=7
        )

    ax.axhline(1.0, color=PAL.faint_gray, lw=0.6, ls=":", alpha=0.5)
    ax.axhline(0, color=PAL.faint_gray, lw=0.6, ls=":", alpha=0.5)
    ax.text(
        x[-1] + 0.5,
        1.0,
        "Full washout",
        fontsize=6.5,
        color=PAL.light_gray,
        va="bottom",
    )
    ax.text(
        x[-1] + 0.5,
        0.0,
        "Full persistence",
        fontsize=6.5,
        color=PAL.light_gray,
        va="top",
    )

    ax.set_ylim(-0.08, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(xl)
    ax.set_xlabel("K (number of wrong endorsements)")
    if not appendix_style:
        ax.set_ylabel("In-context washout score\n(0 = persists, 1 = washes out)")
        ax.set_title(f"Persistence after pressure ({_tag_display(tag)})", loc="left")
    add_y_grid(ax)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        default="expert",
        choices=["expert", "note"],
    )
    parser.add_argument(
        "--appendix-style",
        action="store_true",
        help="Caption-ready layout: no titles, X-axis labels only, single-line legend. For ICML-style bold caption.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("new-phase-results/figures/appendix")
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()
    apply_theme()

    appendix_style = args.appendix_style

    b = Path("new-phase-results")
    tag_key = _tag_key(args.tag)
    all_summaries = [
        (m["key"], load_json(model_result_path(b, m["key"], m["summary"])))
        for m in MODELS
    ]
    model_summaries = [
        (k, s) for k, s in all_summaries
        if tag_key in s.get("metrics", {})
    ]
    if not model_summaries:
        raise SystemExit(
            f"No model has metrics for tag '{args.tag}' (key: {tag_key}). "
            "Extended tags (user, someone_online) require exp12_k run with --extended-tags."
        )

    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(FIGSIZE_2x1[0], FIGSIZE_2x1[1] + 0.4), constrained_layout=False
    )
    if appendix_style:
        fig.subplots_adjust(left=0.11, right=0.95, top=0.88, bottom=0.26, wspace=0.35)
    else:
        fig.subplots_adjust(left=0.11, right=0.95, top=0.82, bottom=0.18, wspace=0.35)

    plot_pressure(a1, model_summaries, tag=args.tag, appendix_style=appendix_style)
    plot_washout(a2, model_summaries, tag=args.tag, appendix_style=appendix_style)
    label_panel(a1, "A")
    label_panel(a2, "B")

    handles, labels = a1.get_legend_handles_labels()
    if appendix_style:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=6,
            fontsize=6.5,
            columnspacing=1.2,
            handletextpad=0.5,
            bbox_to_anchor=(0.5, -0.04),
            frameon=False,
        )
    else:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            fontsize=6.5,
            columnspacing=1.2,
            handletextpad=0.5,
            bbox_to_anchor=(0.5, 0.04),
            frameon=False,
        )

    if not appendix_style:
        fig.suptitle(
            "Repeated endorsement pressure creates persistent in-context bias",
            fontsize=10,
            fontweight="bold",
            y=0.93,
            color=PAL.dark_text,
        )

    for p in save_fig(fig, f"fig4_temporal_{args.tag}", args.output_dir, args.formats):
        print(f"  {p}")


if __name__ == "__main__":
    main()
