#!/usr/bin/env python3
"""Figure 3 (Hero): "Be correct" induces prior-consistency, not truth-tracking.

Three-panel figure:
  A  Gradient scatter of per-item selectivity vs. prior confidence (m_N0),
     showing Expert and Note rolling trends diverging on wrong-prior items.
  B  Inverted-prior test: dr (selectivity) across subsets of increasing
     model-wrong confidence, for Expert vs Note.  Expert stays near zero
     while Note goes strongly negative.
  C  Note decomposition on confidently-wrong items: r_w collapses while
     r_c rises — the mechanistic explanation of prior-consistency.

Data sources:
  Panel A  →  exp10_extended item-level JSONL  (same items exp11 reanalyzes)
  Panel B  →  exp11_extended/part_a  (overall)  +  part_b  (inverted-prior slices)
  Panel C  →  exp11_extended/part_a  (overall)  +  part_b  (inverted-prior slices)

Run:
    uv run python -m src.paper_figures.fig_hero
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from src.paper_figures.theme import (
    PAL, CMAP_DIVERGING, apply_theme, save_fig,
    label_panel, add_zero_line, add_y_grid, add_region_shading,
)


# ─── Data loading ─────────────────────────────────────────────────────

def load_items(path: Path) -> List[dict]:
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_item_metrics(item: dict, tag: str) -> dict | None:
    """Compute per-item m_N0 and selectivity in probability space.

    m_N0 = logit(correct) - logit(wrong) in neutral-no-instruction.
    selectivity = efficacy_wrong - efficacy_correct (from exp10 framework).
    """
    sm = item.get("selectivity_metrics", {})
    cr = item.get("condition_results", {})

    tag_key = tag.lower().replace(" ", "_")
    n0_key = f"N0_{tag_key}"
    if n0_key not in cr:
        return None

    n0 = cr[n0_key]
    correct_label = item.get("correct_label", "a").lower()

    # m_N0 = logit(correct) - logit(wrong)
    if correct_label == "a":
        m_N0 = n0["logit_a"] - n0["logit_b"]
    else:
        m_N0 = n0["logit_b"] - n0["logit_a"]

    eff_w = sm.get(f"efficacy_wrong_{tag_key}", None)
    eff_c = sm.get(f"efficacy_correct_{tag_key}", None)
    sel = sm.get(f"selectivity_{tag_key}", None)

    if eff_w is None or eff_c is None or sel is None:
        return None

    return {"m_N0": m_N0, "selectivity": sel}


def _extract(data, key):
    """Extract [point, [lo, hi]] → (point, lo, hi)."""
    val = data[key]
    return val[0], val[1][0], val[1][1]


# ─── Rolling statistics ───────────────────────────────────────────────

def rolling_stats(
    x: np.ndarray, y: np.ndarray,
    n_bins: int = 30, min_count: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rolling mean + 95% CI for y as a function of x."""
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    centers, means, ci_lo, ci_hi = [], [], [], []

    for i in range(n_bins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if mask.sum() < min_count:
            continue
        yb = y[mask]
        m = np.mean(yb)
        se = np.std(yb, ddof=1) / math.sqrt(len(yb))
        centers.append((edges[i] + edges[i + 1]) / 2)
        means.append(m)
        ci_lo.append(m - 1.96 * se)
        ci_hi.append(m + 1.96 * se)

    return np.array(centers), np.array(means), np.array(ci_lo), np.array(ci_hi)


# ─── Panel A: Gradient scatter + trend ────────────────────────────────

def plot_gradient_scatter(
    ax: plt.Axes,
    items: List[dict],
    title: str,
) -> None:
    """Selectivity vs. prior confidence for Expert and Note tags."""
    tag_data = {}
    for tag in ["expert", "note"]:
        metrics = []
        for item in items:
            m = compute_item_metrics(item, tag)
            if m is not None:
                metrics.append(m)
        tag_data[tag] = metrics

    # Color norm centered at zero
    all_m = np.concatenate([
        np.array([m["m_N0"] for m in tag_data["expert"]]),
        np.array([m["m_N0"] for m in tag_data["note"]]),
    ])
    norm = mcolors.TwoSlopeNorm(vmin=all_m.min(), vcenter=0, vmax=all_m.max())

    # Faint scatter (Expert only, to show distribution)
    m_N0_e = np.array([m["m_N0"] for m in tag_data["expert"]])
    sel_e = np.clip(np.array([m["selectivity"] for m in tag_data["expert"]]), -0.5, 0.5)
    ax.scatter(m_N0_e, sel_e, c=m_N0_e, cmap=CMAP_DIVERGING, norm=norm,
               s=3, alpha=0.15, linewidths=0, zorder=1, rasterized=True)

    # Rolling trends for both tags
    for tag, color, lbl in [
        ("expert", PAL.blue, "Expert"),
        ("note", PAL.red, "Note"),
    ]:
        m_N0 = np.array([m["m_N0"] for m in tag_data[tag]])
        sel = np.clip(np.array([m["selectivity"] for m in tag_data[tag]]), -0.5, 0.5)
        centers, means, lo, hi = rolling_stats(m_N0, sel, n_bins=25, min_count=20)

        ax.fill_between(centers, lo, hi, color=color, alpha=0.10, linewidth=0, zorder=2)
        ax.plot(centers, means, color=color, linewidth=2.2, zorder=3,
                solid_capstyle="round", label=lbl)

    add_zero_line(ax)
    add_region_shading(ax, y_top=0.5, y_bot=-0.5)

    ax.set_xlabel("Prior confidence  (m_N0)")
    ax.set_ylabel("Instruction selectivity")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", fontsize=8)

    # Semantic x-axis annotations
    ylim = ax.get_ylim()
    ax.annotate("Confidently wrong", xy=(ax.get_xlim()[0] + 0.5, ylim[0] + 0.02),
                fontsize=7, color=PAL.red, fontstyle="italic", ha="left", va="bottom")
    ax.annotate("Confidently correct", xy=(ax.get_xlim()[1] - 0.5, ylim[0] + 0.02),
                fontsize=7, color=PAL.blue, fontstyle="italic", ha="right", va="bottom")

    # Key finding callout — note the overall prior-dependence
    ax.annotate(
        "Selectivity tracks prior confidence;\nsee B, C for tag-specific divergence",
        xy=(-8, -0.30),
        xytext=(-3, 0.38),
        fontsize=7, color=PAL.medium_gray,
        arrowprops=dict(arrowstyle="->", color=PAL.medium_gray, lw=0.8,
                        connectionstyle="arc3,rad=-0.2"),
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PAL.bg_warm,
                  alpha=0.9, edgecolor=PAL.faint_gray, linewidth=0.5),
    )


# ─── Panel B: Inverted-prior selectivity comparison ───────────────────

def plot_inverted_prior_selectivity(
    ax: plt.Axes,
    expert_overall: dict,
    note_overall: dict,
    expert_inv: dict,
    note_inv: dict,
) -> None:
    """dr (selectivity) across subsets: overall → model wrong → confidently wrong.

    Shows Expert staying near zero while Note inverts dramatically.
    """
    # Define the three comparison points
    slice_labels = ["All items", "Model\nwrong", "Confidently\nwrong"]

    for tag_overall, tag_inv, tag_name, color, marker, offset in [
        (expert_overall, expert_inv, "Expert", PAL.blue, "o", -0.12),
        (note_overall, note_inv, "Note", PAL.red, "s", 0.12),
    ]:
        # Extract dr_median [point, [lo, hi]] for each slice
        ov_metrics = tag_overall["metrics_sign_consistent"]
        dr_overall = _extract(ov_metrics, "dr_median")

        inv_full = tag_inv["slices"]["full"]["metrics"]
        dr_wrong = _extract(inv_full, "dr_median")

        inv_hc = tag_inv["slices"]["high_conf_top25"]["metrics"]
        dr_conf_wrong = _extract(inv_hc, "dr_median")

        points = [dr_overall, dr_wrong, dr_conf_wrong]
        x_vals = np.arange(3) + offset
        y_vals = [p[0] for p in points]
        ci_lo = [p[0] - p[1] for p in points]
        ci_hi = [p[2] - p[0] for p in points]

        # Error bars
        ax.errorbar(
            x_vals, y_vals,
            yerr=[ci_lo, ci_hi],
            fmt="none", ecolor=PAL.medium_gray, elinewidth=0.8,
            capsize=3, capthick=0.6, alpha=0.5, zorder=2,
        )
        # Dots
        ax.scatter(
            x_vals, y_vals, s=55, c=color, marker=marker,
            edgecolors=PAL.bg_warm, linewidths=0.8, zorder=3,
            label=tag_name,
        )
        # Value labels on the final (most extreme) point
        val = y_vals[-1]
        va = "bottom" if val >= 0 else "top"
        y_off = 0.06 if val >= 0 else -0.06
        ax.text(
            x_vals[-1], val + y_off,
            f"{val:.2f}",
            ha="center", va=va, fontsize=7.5, fontweight="bold", color=color,
        )

    add_zero_line(ax)
    add_region_shading(ax)
    add_y_grid(ax)

    ax.set_xticks(range(3))
    ax.set_xticklabels(slice_labels, fontsize=7.5)
    ax.set_ylabel("Selectivity (dr)")
    ax.set_title("Inverted-prior test", loc="left")
    ax.legend(loc="upper left", fontsize=7, markerscale=0.8)


# ─── Panel C: r_w vs r_c decomposition (Note, inverted-prior) ────────

def plot_note_decomposition(
    ax: plt.Axes,
    note_overall: dict,
    note_inv: dict,
) -> None:
    """Show r_w collapsing while r_c rises as model becomes more confidently wrong.

    Three slices: overall, model-wrong (full), confidently-wrong (top25).
    """
    slice_labels = ["All\nitems", "Model\nwrong", "Confidently\nwrong"]

    # Collect r_w and r_c medians across slices
    ov = note_overall["metrics_sign_consistent"]
    full = note_inv["slices"]["full"]["metrics"]
    hc = note_inv["slices"]["high_conf_top25"]["metrics"]

    for metric, color, label in [
        ("r_w_median", PAL.r_w, "r_w (wrong endorsement)"),
        ("r_c_median", PAL.r_c, "r_c (correct endorsement)"),
    ]:
        points = [_extract(ov, metric), _extract(full, metric), _extract(hc, metric)]
        x_arr = np.arange(3)
        y_arr = np.array([p[0] for p in points])
        lo_arr = np.array([p[1] for p in points])
        hi_arr = np.array([p[2] for p in points])

        ax.fill_between(x_arr, lo_arr, hi_arr, color=color, alpha=0.15, linewidth=0)
        ax.plot(x_arr, y_arr, color=color, marker="o", markersize=5,
                markeredgecolor=PAL.bg_warm, markeredgewidth=0.8,
                linewidth=1.8, zorder=3, label=label)

        # Endpoint label
        ax.annotate(
            f"{y_arr[-1]:.2f}",
            xy=(x_arr[-1], y_arr[-1]),
            xytext=(8, 0), textcoords="offset points",
            fontsize=7, fontweight="bold", color=color, va="center",
        )

    add_zero_line(ax)
    add_y_grid(ax)

    ax.set_xticks(range(3))
    ax.set_xticklabels(slice_labels, fontsize=7.5)
    ax.set_ylabel("Suppression ratio")
    ax.set_title("Note: r_w / r_c decomposition", loc="left")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(-0.5, 1.2)

    # Key finding callout
    ax.text(
        0.5, 0.03,
        "r_w collapses; instruction protects wrong prior",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=6.5, color=PAL.medium_gray, fontstyle="italic",
    )


# ─── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("new-phase-results/figures/paper"))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()

    apply_theme()

    base = Path("new-phase-results")

    # ── Load data ─────────────────────────────────────────────────────
    # Panel A: item-level data (same items that exp11 reanalyzes)
    items = load_items(
        base / "llama/exp10_extended/meta-llama__Llama-3.1-8B-Instruct_results.jsonl"
    )

    # Panels B/C: Part A (overall) + Part B (inverted-prior slices)
    expert_overall = load_json(
        base / "llama/exp11_extended/part_a/meta-llama__Llama-3.1-8B-Instruct_expert_metrics.json"
    )
    note_overall = load_json(
        base / "llama/exp11_extended/part_a/meta-llama__Llama-3.1-8B-Instruct_note_metrics.json"
    )
    expert_inv = load_json(
        base / "llama/exp11_extended/part_b/meta-llama__Llama-3.1-8B-Instruct_expert_inverted_prior.json"
    )
    note_inv = load_json(
        base / "llama/exp11_extended/part_b/meta-llama__Llama-3.1-8B-Instruct_note_inverted_prior.json"
    )

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.5, 6.0), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.10,
                        hspace=0.50, wspace=0.40)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

    ax_scatter = fig.add_subplot(gs[0, :])   # Full-width top panel
    ax_inv     = fig.add_subplot(gs[1, 0])
    ax_decomp  = fig.add_subplot(gs[1, 1])

    # Panel A
    plot_gradient_scatter(ax_scatter, items,
                          "Instruction selectivity vs. prior confidence")

    # Panel B
    plot_inverted_prior_selectivity(
        ax_inv, expert_overall, note_overall, expert_inv, note_inv,
    )

    # Panel C
    plot_note_decomposition(ax_decomp, note_overall, note_inv)

    label_panel(ax_scatter, "A")
    label_panel(ax_inv, "B")
    label_panel(ax_decomp, "C")

    fig.suptitle(
        '"Be correct" induces prior-consistency, not truth-tracking',
        fontsize=12, fontweight="bold", y=0.97, color=PAL.dark_text,
    )

    paths = save_fig(fig, "fig3_hero_prior_consistency", args.output_dir, args.formats)
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
