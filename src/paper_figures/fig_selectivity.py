#!/usr/bin/env python3
"""Figure 3: "Be correct" induces prior-consistency, not truth-tracking.

Three-panel figure combining exp10 (selectivity) and exp11 (inverted prior).

Panel A: Selectivity bars (exp10) — Instruct shows apparent truth-tracking
Panel B: Inverted-prior test (exp11) — selectivity flips when model is wrong
Panel C: Decomposition (exp11) — r_w vs r_c for Note tag, explaining the flip

This is the "hero" figure of the paper.

Run:
    uv run python -m src.paper_figures.fig_selectivity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.theme import (
    PAL, apply_theme, save_fig,
    label_panel, add_zero_line, add_y_grid, add_region_shading,
    draw_bar, draw_error_bar,
)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _extract(data, key: str):
    """Extract [point, [lo, hi]] format → (point, lo, hi)."""
    if key not in data:
        return None
    val = data[key]
    return val[0], val[1][0], val[1][1]


# ─── Panel A: Selectivity from exp10 ─────────────────────────────────

def plot_selectivity(
    ax: plt.Axes,
    llama_inst: dict,
    llama_base: dict,
) -> None:
    """Exp10 selectivity: truth-tracking vs gating test."""

    tags = ["Expert", "Note"]
    x = np.arange(len(tags))
    bar_w = 0.3

    for m_i, (data, color, label) in enumerate([
        (llama_inst, PAL.steel_blue, "Instruct"),
        (llama_base, PAL.slate_gold, "Base"),
    ]):
        offset = -bar_w / 2 - 0.02 + m_i * (bar_w + 0.04)
        sel = data["selectivity_metrics"]

        for t_i, tag in enumerate(tags):
            key = f"selectivity_{tag.lower()}"
            d = sel[key]
            mean = d["mean"]
            n = d.get("n_positive", 0) + d.get("n_negative", 0)
            from src.paper_figures.theme import ci95
            lo, hi = ci95(mean, d["std"], n)

            draw_bar(
                ax, x[t_i] + offset, mean,
                width=bar_w, color=color,
                alpha=0.9,
                label=label if t_i == 0 else None,
            )
            draw_error_bar(
                ax, x[t_i] + offset, mean, lo, hi,
                cap_width=0.07,
            )

            # Value label
            y_label = max(mean, hi) + 0.003 if mean >= 0 else min(mean, lo) - 0.003
            va = "bottom" if mean >= 0 else "top"
            ax.text(
                x[t_i] + offset, y_label,
                f"{mean:.3f}",
                ha="center", va=va,
                fontsize=6.5, fontweight="bold", color=color,
            )

    add_zero_line(ax)
    add_y_grid(ax)

    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_ylabel("Selectivity\n(eff_wrong - eff_correct)")
    ax.set_title("Apparent truth-tracking", loc="left")
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.2)


# ─── Panel B: Inverted-prior selectivity ──────────────────────────────

def plot_inverted_prior(
    ax: plt.Axes,
    results: Dict[str, dict],
) -> None:
    """Exp11 inverted-prior test: dr across confidence slices."""

    slices = ["full_dataset", "low_conf", "high_conf_top25"]
    slice_labels = ["All items", "Wrong prior\n(bot 50%)", "Wrong prior\n(top 25%)"]
    x = np.arange(len(slices))
    offset = 0.12

    for tag, color, marker, x_off in [
        ("expert", PAL.steel_blue, "o", -offset),
        ("note", PAL.deep_rose, "s", offset),
    ]:
        vals, ci_los, ci_his = [], [], []

        for sl in slices:
            if sl == "full_dataset":
                # Use part_a
                data = results.get(f"part_a_{tag}", {})
                metrics = data.get("metrics_sign_consistent", {})
            else:
                # Use part_b
                data = results.get(f"part_b_{tag}", {})
                sl_key = sl if sl != "full_dataset" else "full"
                metrics = data.get("slices", {}).get(sl_key, {}).get("metrics", {})

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
        ax.errorbar(
            xi, vals,
            yerr=[ci_los, ci_his],
            fmt=marker, color=color, markersize=6,
            capsize=3, capthick=0.8, linewidth=1.2,
            markeredgecolor="white", markeredgewidth=0.8,
            label=tag.capitalize(), zorder=3,
        )

        # Significance markers
        for i, (v, lo, hi) in enumerate(zip(vals, ci_los, ci_his)):
            if not np.isnan(v):
                lo_val = v - lo
                hi_val = v + hi
                if (lo_val > 0 and hi_val > 0) or (lo_val < 0 and hi_val < 0):
                    ax.annotate(
                        "*", (xi[i], hi_val + 0.02),
                        ha="center", va="bottom",
                        fontsize=10, color=color, fontweight="bold",
                    )

    add_zero_line(ax)
    add_region_shading(ax)
    add_y_grid(ax)

    ax.set_xticks(x)
    ax.set_xticklabels(slice_labels, fontsize=7)
    ax.set_ylabel("Selectivity (dr = r_w - r_c)\nmedian, 95% CI")
    ax.set_title("Selectivity flips when\nmodel prior is wrong", loc="left")
    ax.legend(loc="upper left", fontsize=6.5)


# ─── Panel C: r_w vs r_c decomposition for Note tag ──────────────────

def plot_decomposition(
    ax: plt.Axes,
    results: Dict[str, dict],
) -> None:
    """Why selectivity flips: r_w collapses while r_c rises (Note tag)."""

    slices = ["full_dataset", "low_conf", "high_conf_top25"]
    slice_labels = ["All", "Wrong\n(bot 50%)", "Wrong\n(top 25%)"]
    x = np.arange(len(slices))
    bar_w = 0.3

    tag = "note"

    for m_i, (metric, color, label) in enumerate([
        ("r_w_median", PAL.terracotta, "r_w (wrong)"),
        ("r_c_median", PAL.sage_green, "r_c (correct)"),
    ]):
        offset = -bar_w / 2 - 0.02 + m_i * (bar_w + 0.04)

        for s_i, sl in enumerate(slices):
            if sl == "full_dataset":
                data = results.get(f"part_a_{tag}", {})
                metrics = data.get("metrics_sign_consistent", {})
            else:
                data = results.get(f"part_b_{tag}", {})
                metrics = data.get("slices", {}).get(sl, {}).get("metrics", {})

            val = _extract(metrics, metric)
            if val:
                mean, lo, hi = val
                draw_bar(
                    ax, x[s_i] + offset, mean,
                    width=bar_w, color=color,
                    alpha=0.85,
                    label=label if s_i == 0 else None,
                )
                draw_error_bar(
                    ax, x[s_i] + offset, mean, lo, hi,
                    cap_width=0.06,
                )

    add_zero_line(ax)
    add_y_grid(ax)

    ax.set_xticks(x)
    ax.set_xticklabels(slice_labels, fontsize=7)
    ax.set_ylabel("Relative suppression ratio")
    ax.set_title("Note: r_c overtakes r_w", loc="left")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_ylim(-0.15, 1.1)

    # No additional annotation to keep panel clean


# ─── Load exp11 results ──────────────────────────────────────────────

def load_exp11_results(
    base_dir: Path,
    model_prefix: str,
) -> Dict[str, dict]:
    """Load exp11 part_a and part_b for expert and note tags."""
    results = {}
    for tag in ["expert", "note"]:
        # Part A: full dataset
        pa = base_dir / "part_a" / f"{model_prefix}_{tag}_metrics.json"
        if pa.exists():
            results[f"part_a_{tag}"] = load_json(pa)

        # Part B: inverted prior
        pb = base_dir / "part_b" / f"{model_prefix}_{tag}_inverted_prior.json"
        if pb.exists():
            results[f"part_b_{tag}"] = load_json(pb)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("new-phase-results/figures/paper"))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()

    apply_theme()

    base = Path("new-phase-results")

    # Exp10 data
    llama_inst_10 = load_json(base / "llama/exp10/meta-llama__Llama-3.1-8B-Instruct_summary.json")
    llama_base_10 = load_json(base / "llama/exp10/meta-llama__Llama-3.1-8B_summary.json")

    # Exp11 data (Llama Instruct)
    exp11_results = load_exp11_results(
        base / "llama/exp11_extended",
        "meta-llama__Llama-3.1-8B-Instruct",
    )

    # Create 3-panel figure
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,
        figsize=(7.5, 3.4),
        gridspec_kw={"width_ratios": [0.75, 1.1, 1.0], "wspace": 0.35},
    )

    plot_selectivity(ax1, llama_inst_10, llama_base_10)
    plot_inverted_prior(ax2, exp11_results)
    plot_decomposition(ax3, exp11_results)

    label_panel(ax1, "A")
    label_panel(ax2, "B")
    label_panel(ax3, "C")

    fig.suptitle(
        '"Be correct" induces prior-consistency, not truth-tracking',
        fontsize=10, fontweight="bold", y=1.04,
    )

    paths = save_fig(fig, "fig3_prior_consistency", args.output_dir, args.formats)
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
