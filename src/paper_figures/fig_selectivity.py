#!/usr/bin/env python3
"""Figure 3: "Be correct" induces prior-consistency, not truth-tracking.

Three-panel figure combining exp10 (selectivity) and exp11 (inverted prior),
showing all three instruct/thinking models (base model dropped).

Panel A: Selectivity bars (exp10) — all 3 models grouped by tag
Panel B: Inverted-prior test (exp11) — selectivity flips when model is wrong
Panel C: Decomposition (exp11) — r_w vs r_c for Note tag, all 3 models

Run:
    uv run python -m src.paper_figures.fig_selectivity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.theme import (
    PAL,
    apply_theme,
    save_fig,
    label_panel,
    add_zero_line,
    add_y_grid,
    add_region_shading,
    draw_bar,
    draw_error_bar,
    ci95,
    model_result_path,
)

MODELS = [
    {
        "key": "llama",
        "exp10_summary": "exp10/meta-llama__Llama-3.1-8B-Instruct_summary.json",
        "exp11_prefix": "meta-llama__Llama-3.1-8B-Instruct",
        "exp11_dir": "exp11_extended",
    },
    {
        "key": "qwen",
        "exp10_summary": "exp10/Qwen__Qwen3-4B-Instruct-2507_summary.json",
        "exp11_prefix": "Qwen__Qwen3-4B-Instruct-2507",
        "exp11_dir": "exp11_extended",
    },
    {
        "key": "qwen_thinking",
        "exp10_summary": "exp10/Qwen__Qwen3-4B-Thinking-2507_summary.json",
        "exp11_prefix": "Qwen__Qwen3-4B-Thinking-2507",
        "exp11_dir": "exp11_extended",
    },
]


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _extract(data, key: str):
    if key not in data:
        return None
    val = data[key]
    return val[0], val[1][0], val[1][1]


# ─── Panel A: Selectivity from exp10 (all 3 models) ──────────────────


def plot_selectivity(
    ax,
    model_data: List[dict],  # list of {key, summary}
) -> None:
    tags = ["Expert", "Note"]
    x = np.arange(len(tags))
    n_m = len(model_data)
    bar_w = 0.20
    total = n_m * bar_w + (n_m - 1) * 0.04
    offsets = np.linspace(-total / 2 + bar_w / 2, total / 2 - bar_w / 2, n_m)

    for m_i, md in enumerate(model_data):
        sty = PAL.MODEL_STYLES[md["key"]]
        color = sty["color"]
        sel = md["summary"].get("selectivity_metrics", {})

        for t_i, tag in enumerate(tags):
            key = f"selectivity_{tag.lower()}"
            d = sel.get(key)
            if d is None:
                continue
            mean = d["mean"]
            n = d.get("n_positive", 0) + d.get("n_negative", 0)
            lo, hi = ci95(mean, d["std"], n)

            draw_bar(
                ax,
                x[t_i] + offsets[m_i],
                mean,
                width=bar_w,
                color=color,
                alpha=0.88,
                label=sty["short"] if t_i == 0 else None,
            )
            draw_error_bar(ax, x[t_i] + offsets[m_i], mean, lo, hi, cap_width=0.06)

            y_label = max(mean, hi) + 0.003 if mean >= 0 else min(mean, lo) - 0.003
            va = "bottom" if mean >= 0 else "top"
            ax.text(
                x[t_i] + offsets[m_i],
                y_label,
                f"{mean:.3f}",
                ha="center",
                va=va,
                fontsize=5.5,
                fontweight="bold",
                color=color,
            )

    add_zero_line(ax)
    add_y_grid(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_ylabel("Selectivity\n(eff_wrong − eff_correct)")
    ax.set_title("Apparent truth-tracking", loc="left")
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.2)


# ─── Panel B: Inverted-prior selectivity (all 3 models) ──────────────


def plot_inverted_prior(
    ax,
    model_data: List[dict],  # list of {key, exp11_results}
) -> None:
    slices = ["full_dataset", "low_conf", "high_conf_top25"]
    slice_labels = ["All items", "Wrong prior\n(bot 50%)", "Wrong prior\n(top 25%)"]
    x = np.arange(len(slices))

    for m_i, md in enumerate(model_data):
        sty = PAL.MODEL_STYLES[md["key"]]
        results = md["exp11_results"]

        for tag, tag_color, x_off in [
            ("expert", PAL.blue, -0.20 + m_i * 0.13),
            ("note", PAL.red, 0.07 + m_i * 0.13),
        ]:
            vals, ci_los, ci_his = [], [], []
            for sl in slices:
                if sl == "full_dataset":
                    metrics = results.get(f"part_a_{tag}", {}).get(
                        "metrics_sign_consistent", {}
                    )
                else:
                    metrics = (
                        results.get(f"part_b_{tag}", {})
                        .get("slices", {})
                        .get(sl, {})
                        .get("metrics", {})
                    )
                dr = _extract(metrics, "dr_median")
                if dr:
                    vals.append(dr[0])
                    ci_los.append(dr[0] - dr[1])
                    ci_his.append(dr[2] - dr[0])
                else:
                    vals.append(np.nan)
                    ci_los.append(0)
                    ci_his.append(0)

            xi = x + x_off
            label = (
                f"{sty['short']} {'Exp' if tag == 'expert' else 'Note'}"
                if m_i == 0
                else None
            )
            ax.errorbar(
                xi,
                vals,
                yerr=[ci_los, ci_his],
                fmt=sty["marker"],
                color=tag_color,
                markersize=5,
                capsize=2,
                capthick=0.6,
                linewidth=1.0,
                markeredgecolor=PAL.bg_warm,
                markeredgewidth=0.7,
                label=label,
                zorder=3,
                alpha=0.9,
                linestyle=["-", "--", ":"][m_i],
            )

    add_zero_line(ax)
    add_region_shading(ax)
    add_y_grid(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(slice_labels, fontsize=7)
    ax.set_ylabel("Selectivity (dr)\nmedian, 95% CI")
    ax.set_title("Selectivity flips when\nmodel prior is wrong", loc="left")

    # Compact legend: tag colours + model linestyles
    handles = [
        mlines.Line2D([0], [0], color=PAL.blue, lw=1.8, label="Expert"),
        mlines.Line2D([0], [0], color=PAL.red, lw=1.8, label="Note"),
    ]
    for m_i, md in enumerate(model_data):
        sty = PAL.MODEL_STYLES[md["key"]]
        handles.append(
            mlines.Line2D(
                [0],
                [0],
                color=PAL.medium_gray,
                linestyle=["-", "--", ":"][m_i],
                linewidth=1.5,
                label=sty["short"],
            )
        )
    ax.legend(handles=handles, loc="upper left", fontsize=6, handlelength=1.2, ncol=2)


# ─── Panel C: r_w vs r_c decomposition (Note, all 3 models) ──────────


def plot_decomposition(
    ax,
    model_data: List[dict],
) -> None:
    slices = ["full_dataset", "low_conf", "high_conf_top25"]
    slice_labels = ["All", "Wrong\n(bot 50%)", "Wrong\n(top 25%)"]
    x = np.arange(len(slices))

    for m_i, md in enumerate(model_data):
        results = md["exp11_results"]
        ls = ["-", "--", ":"][m_i]

        for metric, color, lbl in [
            ("r_w_median", PAL.r_w, "r_w"),
            ("r_c_median", PAL.r_c, "r_c"),
        ]:
            y_vals, lo_vals, hi_vals = [], [], []
            for sl in slices:
                if sl == "full_dataset":
                    metrics = results.get("part_a_note", {}).get(
                        "metrics_sign_consistent", {}
                    )
                else:
                    metrics = (
                        results.get("part_b_note", {})
                        .get("slices", {})
                        .get(sl, {})
                        .get("metrics", {})
                    )
                val = _extract(metrics, metric)
                if val:
                    y_vals.append(val[0])
                    lo_vals.append(val[1])
                    hi_vals.append(val[2])
                else:
                    y_vals.append(np.nan)
                    lo_vals.append(np.nan)
                    hi_vals.append(np.nan)

            y_arr = np.array(y_vals)
            lo_arr = np.array(lo_vals)
            hi_arr = np.array(hi_vals)
            label = f"{lbl}" if m_i == 0 else None

            ax.fill_between(x, lo_arr, hi_arr, color=color, alpha=0.08, linewidth=0)
            ax.plot(
                x,
                y_arr,
                color=color,
                marker=PAL.MODEL_STYLES[md["key"]]["marker"],
                markersize=4.5,
                markeredgecolor=PAL.bg_warm,
                markeredgewidth=0.7,
                linewidth=1.6,
                alpha=0.9 - m_i * 0.18,
                zorder=3,
                linestyle=ls,
                label=label,
            )

    add_zero_line(ax)
    add_y_grid(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(slice_labels, fontsize=7)
    ax.set_ylabel("Relative suppression ratio")
    ax.set_title("Note: r_c overtakes r_w", loc="left")
    ax.set_ylim(-0.15, 1.1)

    handles = [
        mlines.Line2D([0], [0], color=PAL.r_w, lw=1.8, label="r_w (wrong)"),
        mlines.Line2D([0], [0], color=PAL.r_c, lw=1.8, label="r_c (correct)"),
    ]
    for m_i, md in enumerate(model_data):
        sty = PAL.MODEL_STYLES[md["key"]]
        handles.append(
            mlines.Line2D(
                [0],
                [0],
                color=PAL.medium_gray,
                linestyle=["-", "--", ":"][m_i],
                linewidth=1.5,
                label=sty["short"],
            )
        )
    ax.legend(handles=handles, loc="upper left", fontsize=6, handlelength=1.2)


# ─── Load exp11 results ──────────────────────────────────────────────


def load_exp11_results(base_dir: Path, model_prefix: str) -> Dict[str, dict]:
    results = {}
    for tag in ["expert", "note"]:
        pa = base_dir / "part_a" / f"{model_prefix}_{tag}_metrics.json"
        if pa.exists():
            results[f"part_a_{tag}"] = load_json(pa)
        pb = base_dir / "part_b" / f"{model_prefix}_{tag}_inverted_prior.json"
        if pb.exists():
            results[f"part_b_{tag}"] = load_json(pb)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("new-phase-results/figures/paper")
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()

    apply_theme()
    base = Path("new-phase-results")

    model_data = []
    for m in MODELS:
        summary = load_json(model_result_path(base, m["key"], m["exp10_summary"]))
        exp11_dir = model_result_path(base, m["key"], m["exp11_dir"])
        exp11_results = load_exp11_results(exp11_dir, m["exp11_prefix"])
        model_data.append(
            {
                "key": m["key"],
                "summary": summary,
                "exp11_results": exp11_results,
            }
        )

    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        figsize=(8.5, 3.6),
        gridspec_kw={"width_ratios": [0.85, 1.1, 1.0], "wspace": 0.38},
    )

    plot_selectivity(ax1, model_data)
    plot_inverted_prior(ax2, model_data)
    plot_decomposition(ax3, model_data)

    label_panel(ax1, "A")
    label_panel(ax2, "B")
    label_panel(ax3, "C")

    fig.suptitle(
        '"Be correct" induces prior-consistency, not truth-tracking',
        fontsize=10,
        fontweight="bold",
        y=1.04,
    )

    paths = save_fig(fig, "fig3_prior_consistency", args.output_dir, args.formats)
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
