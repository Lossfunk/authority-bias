#!/usr/bin/env python3
"""Figure 5a/5b: Phrasing strength modulates endorsement susceptibility.

Two separate figures from Exp13 signal-strength experiment:
  5a  Slope chart: endorsement effect climbs with speaker certainty.
      All three models shown (same tag colour, model encoded by line style + marker).
  5b  Salience cascade: horizontal lollipop across five formatting styles
      for Expert + Note, with model-specific markers and prompt example.

Data sources:
  5a  →  exp13 main run  (4 tags × 3 certainty × 2 salience)
  5b  →  exp13_f run     (Expert + Note × 1 certainty × 5 salience)

Run:
    uv run python -m src.paper_figures.fig_signal_strength
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.theme import (
    PAL,
    apply_theme,
    save_fig,
    label_panel,
    add_y_grid,
    ci95 as theme_ci95,
    make_sequential_cmap,
    model_result_path,
)


# ─── Constants ────────────────────────────────────────────────────────

CERTAINTY_ORDER = ["might", "think", "sure"]
CERTAINTY_LABELS = [
    '"I think it\nmight be ..."',
    '"I think\nit\'s ..."',
    "\"I'm sure\nit's ...\"",
]
SALIENCE_5 = ["plain", "important", "allcaps", "exclaim", "bracketed"]
SALIENCE_LABELS = {
    "plain": "Plain",
    "important": "IMPORTANT:",
    "allcaps": "ALL CAPS",
    "exclaim": "Exclaim !!",
    "bracketed": "[IMPORTANT]",
}

TAG_ORDER = ["expert", "note", "user", "someone_online"]
TAG_DISPLAY = {
    "expert": "Expert",
    "note": "Note",
    "user": "User",
    "someone_online": "Online",
}
TAG_COLORS = {
    "expert": PAL.TAG_COLORS["Expert"],
    "note": PAL.TAG_COLORS["Note"],
    "user": PAL.TAG_COLORS["User"],
    "someone_online": PAL.TAG_COLORS["Someone online"],
}

# Line style + marker per model (colour comes from tag)
MODEL_STYLE = {
    "llama": {"ls": "-", "marker": "o", "lw": 2.0},
    "qwen": {"ls": "--", "marker": "s", "lw": 1.8},
    "qwen_thinking": {"ls": ":", "marker": "^", "lw": 1.6},
}

MODELS = [
    {
        "key": "llama",
        "exp13_dir": "exp13/exp13_llama",
        "exp13_f_dir": "exp13/exp13_f_llama",
        "summary_stem": "meta-llama__Llama-3.1-8B-Instruct_summary.json",
    },
    {
        "key": "qwen",
        "exp13_dir": "exp13/exp13_qwen",
        "exp13_f_dir": "exp13/exp13_f_qwen_instruct_ext4",
        "summary_stem": "Qwen__Qwen3-4B-Instruct-2507_summary.json",
    },
    {
        "key": "qwen_thinking",
        "exp13_dir": "exp13/exp13_qwen",  # thinking results live here too
        "exp13_f_dir": "exp13/exp13_f_qwen_thinking_ext4",
        "summary_stem": "Qwen__Qwen3-4B-Thinking-2507_summary.json",
    },
]


def load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def extract_metric(
    summary: dict,
    tag: str,
    certainty: str,
    salience: str,
    metric: str = "effect_wrong_I0",
) -> float:
    tag_key = tag.lower().replace(" ", "_")
    variant = f"{certainty}_{salience}"
    return float(summary["metrics"][tag_key][variant][metric]["mean"])


def extract_avg_salience(
    summary: dict,
    tag: str,
    certainty: str,
    salience_levels: Sequence[str] = ("plain", "important"),
    metric: str = "effect_wrong_I0",
) -> float:
    vals = [extract_metric(summary, tag, certainty, s, metric) for s in salience_levels]
    return float(np.mean(vals))


# ─── Figure 5a: Certainty dose-response ──────────────────────────────


def _plot_fig5a_panel(
    ax: plt.Axes,
    model_summaries: List[dict],
    *,
    right_pad: float,
    show_endpoint_labels: bool = True,
) -> None:
    x = np.arange(len(CERTAINTY_ORDER))

    for tag in TAG_ORDER:
        color = TAG_COLORS[tag]
        for ms in model_summaries:
            style = MODEL_STYLE[ms["key"]]
            means = [
                extract_avg_salience(ms["summary"], tag, cert)
                for cert in CERTAINTY_ORDER
            ]
            ax.plot(
                x,
                np.array(means),
                color=color,
                linewidth=style["lw"],
                linestyle=style["ls"],
                marker=style["marker"],
                markersize=5.0,
                alpha=0.82,
                markeredgecolor=PAL.bg_warm,
                markeredgewidth=0.7,
                zorder=3,
            )

    # Endpoint labels (average across models for vertical placement)
    if show_endpoint_labels:
        endpoints: List[tuple] = []
        for tag in TAG_ORDER:
            avg = np.mean(
                [
                    extract_avg_salience(ms["summary"], tag, CERTAINTY_ORDER[-1])
                    for ms in model_summaries
                ]
            )
            endpoints.append((avg, TAG_DISPLAY[tag], TAG_COLORS[tag]))

        endpoints.sort(key=lambda t: t[0], reverse=True)
        min_gap = 0.032
        placed: List[float] = []
        for val, lbl, col in endpoints:
            y_lbl = val
            for prev in placed:
                if abs(y_lbl - prev) < min_gap:
                    y_lbl = prev - min_gap
            placed.append(y_lbl)
            ax.annotate(
                lbl,
                xy=(x[-1], val),
                xytext=(12, (y_lbl - val) * 180),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color=col,
                va="center",
                ha="left",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(CERTAINTY_LABELS, fontsize=7.5, linespacing=1.1)
    ax.set_ylabel("Endorsement effect")
    ax.set_xlim(-0.3, len(CERTAINTY_ORDER) - 1 + right_pad)
    add_y_grid(ax)


def _legend_handles(tags: Sequence[str], model_summaries: List[dict]) -> List[mlines.Line2D]:
    handles: List[mlines.Line2D] = []
    for tag in tags:
        handles.append(
            mlines.Line2D(
                [0],
                [0],
                color=TAG_COLORS[tag],
                linewidth=2.0,
                marker="o",
                markersize=4.5,
                markeredgecolor=PAL.bg_warm,
                markeredgewidth=0.5,
                label=TAG_DISPLAY[tag],
            )
        )
    handles.append(mlines.Line2D([0], [0], color="none", label=" "))
    for ms in model_summaries:
        s = MODEL_STYLE[ms["key"]]
        sty = PAL.MODEL_STYLES[ms["key"]]
        handles.append(
            mlines.Line2D(
                [0],
                [0],
                color=PAL.medium_gray,
                linewidth=1.5,
                linestyle=s["ls"],
                marker=s["marker"],
                markersize=4.5,
                markeredgecolor=PAL.bg_warm,
                markeredgewidth=0.5,
                label=sty["short"],
            )
        )
    return handles


def make_fig5a(model_summaries: List[dict], output_dir: Path, formats: list) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.82, top=0.94, bottom=0.22)

    _plot_fig5a_panel(ax, model_summaries, right_pad=0.9)

    # Legend: tag colours + model encoding
    handles = _legend_handles(TAG_ORDER, model_summaries)

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=7.0,
        columnspacing=0.9,
        handletextpad=0.4,
        bbox_to_anchor=(0.47, -0.02),
        frameon=False,
    )

    paths = save_fig(fig, "fig5a_certainty_gain", output_dir, formats)
    for p in paths:
        print(f"  {p}")


# ─── Figure 5b: Salience spectrum ────────────────────────────────────


def _plot_fig5b_panel(
    ax: plt.Axes,
    model_f_summaries: List[dict],
    *,
    show_xlabel: bool,
    show_prompt_example: bool,
) -> None:
    y_pos = np.arange(len(SALIENCE_5))
    n_models = len(model_f_summaries)
    row_h = 0.12
    # offsets: spread models symmetrically within each salience row
    offsets = np.linspace(
        -(n_models - 1) * row_h / 2, (n_models - 1) * row_h / 2, n_models
    )

    for tag_key in ("expert", "note"):
        color = TAG_COLORS[tag_key]
        for m_i, ms in enumerate(model_f_summaries):
            style = MODEL_STYLE[ms["key"]]
            for i, sal in enumerate(SALIENCE_5):
                m = extract_metric(ms["summary"], tag_key, "think", sal)
                y = y_pos[i] + offsets[m_i]
                if tag_key == "note":
                    y += 0  # note rows sit at same x-range, differentiated by colour
                ax.plot(
                    [0, m],
                    [y, y],
                    color=color,
                    linewidth=0.8,
                    linestyle=style["ls"],
                    alpha=0.25,
                    zorder=2,
                )
                ax.scatter(
                    m,
                    y,
                    color=color,
                    marker=style["marker"],
                    s=42,
                    zorder=5,
                    edgecolors=color,
                    linewidths=0.5,
                    alpha=0.85,
                )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([SALIENCE_LABELS[s] for s in SALIENCE_5], fontsize=8)
    if show_xlabel:
        ax.set_xlabel("Endorsement effect")
    ax.invert_yaxis()
    ax.set_xlim(0, 0.90)
    ax.xaxis.grid(True, alpha=0.20, linewidth=0.5, color=PAL.faint_gray)
    ax.set_axisbelow(True)

    if show_prompt_example:
        ax.text(
            0.97,
            0.61,
            "e.g.  IMPORTANT: I think it's B, not A.",
            transform=ax.transAxes,
            fontsize=6,
            color=PAL.medium_gray,
            fontstyle="italic",
            fontfamily="monospace",
            va="bottom",
            ha="right",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=PAL.bg_warm,
                edgecolor=PAL.faint_gray,
                linewidth=0.4,
                alpha=0.95,
            ),
        )


def make_fig5b(model_f_summaries: List[dict], output_dir: Path, formats: list) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=False)
    fig.subplots_adjust(left=0.18, right=0.92, top=0.94, bottom=0.22)

    _plot_fig5b_panel(
        ax,
        model_f_summaries,
        show_xlabel=True,
        show_prompt_example=True,
    )

    handles = _legend_handles(("expert", "note"), model_f_summaries)

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=7.0,
        columnspacing=0.9,
        handletextpad=0.4,
        bbox_to_anchor=(0.55, -0.02),
        frameon=False,
    )

    paths = save_fig(fig, "fig5b_salience_spectrum", output_dir, formats)
    for p in paths:
        print(f"  {p}")


def make_fig5_combined(
    model_summaries: List[dict],
    model_f_summaries: List[dict],
    output_dir: Path,
    formats: list,
) -> None:
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.6),
        gridspec_kw={"width_ratios": [1.05, 0.95]},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.26, wspace=0.30)

    _plot_fig5a_panel(
        ax_a,
        model_summaries,
        right_pad=0.15,
        show_endpoint_labels=False,
    )
    _plot_fig5b_panel(
        ax_b,
        model_f_summaries,
        show_xlabel=True,
        show_prompt_example=False,
    )

    label_panel(ax_a, "A")
    label_panel(ax_b, "B")

    handles = [h for h in _legend_handles(TAG_ORDER, model_summaries) if h.get_label().strip()]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=7.0,
        columnspacing=0.85,
        handletextpad=0.35,
        bbox_to_anchor=(0.50, 0.08),
        frameon=False,
    )

    paths = save_fig(fig, "fig5_combined", output_dir, formats)
    for p in paths:
        print(f"  {p}")


# ─── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("new-phase-results/figures/paper")
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()

    apply_theme()
    base = Path("new-phase-results")

    model_summaries = []
    model_f_summaries = []

    for m in MODELS:
        summary = load_json(
            model_result_path(
                base,
                m["key"],
                f"{m['exp13_dir']}/{m['summary_stem']}",
            )
        )
        summary_f = load_json(
            model_result_path(
                base,
                m["key"],
                f"{m['exp13_f_dir']}/{m['summary_stem']}",
            )
        )
        model_summaries.append({"key": m["key"], "summary": summary})
        model_f_summaries.append({"key": m["key"], "summary": summary_f})

    make_fig5a(model_summaries, args.output_dir, args.formats)
    make_fig5b(model_f_summaries, args.output_dir, args.formats)
    make_fig5_combined(model_summaries, model_f_summaries, args.output_dir, args.formats)


if __name__ == "__main__":
    main()
