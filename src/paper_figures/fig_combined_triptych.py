"""Fig 5: story-arc triptych — behaviour · mechanism · transfer.

Three clean panels compressed into one figure:

A.  Behavioural override: W1 flip rate across six models.
B.  Forward-patch mechanism in GPT-oss-20B across three layers.
C.  Out-of-domain transfer: Δ wrong-answer rate on PIQA for three
    open-weights models at their best layer.

Styling is deliberately sparse (white background, distinct marker
shapes per model, smooth PCHIP curves) so panels read at a glance and
the paper caption carries the full narrative.

Run::

    uv run python -m src.paper_figures.fig_combined_triptych
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.axis_theme import (
    AXIS,
    apply_axis_theme,
    line_with_markers,
    rounded_bar,
    save_fig,
    soften_spines,
    y_grid,
)


RUNS_JSON = Path("results/authority/_shared/dynamic_parser_all_runs.json")
OUTPUT_DIR = Path("figures/neurips/v2")


HERO_MODELS = [
    ("OLMo-2-32B",    "#D5A048", "allenai/OLMo-2-0325-32B-Instruct::v2_olmo2_freegen"),
    ("GPT-oss-20B",   "#C36662", "openai/gpt-oss-20b::v2_gpt_oss_freegen"),
    ("Gemma-4-26B",   "#6F9C79", "google/gemma-4-26B-A4B-it::gemma4__no_thinking_freegen_merged_authoritative"),
    ("GPT-5.4",       "#8B6F9C", "openai/gpt-5.4::gpt54_native_pool_staged"),
    ("Grok-4.20",     "#9A6B4C", "x-ai/grok-4.20::grok__grok420_dissociation_freegen"),
    ("Gemini-3.1-Pro","#5B84B5", "google/gemini-3.1-pro-preview::gemini31pro__dissociation_freegen"),
]


MECH_LAYERS = (16, 18, 20)
MECH_PATH = Path("results/authority/gpt-oss/mechanism/forward_patch_test_256/steering_rows.jsonl")

TRANSFER_MODELS: List[Dict] = [
    {
        "name": "GPT-oss-20B",
        "path": Path("results/authority/gpt-oss/mechanism/piqa_forward_patch_200_freegen/piqa_summary.json"),
        "layer": 16,
        "accent": "#C36662",
        "marker": "s",
    },
    {
        "name": "Gemma-4-26B",
        "path": Path("results/authority/gemma4/mechanism/piqa_forward_patch_200_freegen_no_thinking/piqa_summary.json"),
        "layer": 20,
        "accent": "#6F9C79",
        "marker": "D",
    },
    {
        "name": "OLMo-2-32B",
        "path": Path("results/authority/olmo2/mechanism/piqa_forward_patch_200_freegen/piqa_summary.json"),
        "layer": 16,
        "accent": "#D5A048",
        "marker": "^",
    },
]


# --------------------------------------------------------------------------
# Data loaders
# --------------------------------------------------------------------------

def _hero_values() -> List[Tuple[str, str, float]]:
    runs = json.loads(RUNS_JSON.read_text())
    out = []
    for name, color, key in HERO_MODELS:
        fr = runs.get(key, {}).get("flip_rate")
        out.append((name, color, (fr or 0.0) * 100.0))
    return out


def _mech_flip_rates() -> Dict[int, List[Tuple[float, float]]]:
    grouped = defaultdict(lambda: defaultdict(dict))
    for line in MECH_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("condition_code") != "N0_note":
            continue
        layer = int(row["target_layers"][0])
        alpha = float(row["alpha"])
        grouped[layer][alpha][row["uid"]] = row
    out = {}
    for layer in MECH_LAYERS:
        by_alpha = grouped[layer]
        base = by_alpha.get(0.0, {})
        baseline_correct = {uid for uid, r in base.items() if r.get("is_correct") is True}
        points = []
        for alpha in sorted(by_alpha):
            rs = by_alpha[alpha]
            flips = sum(1 for uid in baseline_correct if rs.get(uid, {}).get("chose_wrong") is True)
            points.append((alpha, 100.0 * flips / max(len(baseline_correct), 1)))
        out[layer] = sorted(points)
    return out


def _transfer_wrong_delta(path: Path, layer: int):
    rows = json.loads(path.read_text())
    rows = [r for r in rows
            if r["condition_code"] == "N0_note"
            and int(r["target_layers"][0]) == layer]
    rows.sort(key=lambda r: float(r["alpha"]))
    alphas = np.array([float(r["alpha"]) for r in rows])
    wrong = np.array([100.0 * float(r["wrong_rate_parsed"]) for r in rows])
    return alphas, wrong - wrong[0]


def _smooth(x: np.ndarray, y: np.ndarray):
    try:
        from scipy.interpolate import PchipInterpolator
        xs = np.linspace(float(x.min()), float(x.max()), 240)
        return xs, PchipInterpolator(x, y)(xs)
    except Exception:
        return x, y


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------

def _panel_A(ax: plt.Axes, hero_rows):
    bar_w = 0.56
    xs = np.arange(len(hero_rows))
    for xi, (name, color, flip) in zip(xs, hero_rows):
        rounded_bar(
            ax, xi, flip,
            width=bar_w, color=color, rounding=0.05,
            alpha=0.96, edgewidth=0.0,
        )
        ax.text(
            xi, flip + 2.2, f"{flip:.0f}%",
            ha="center", va="bottom",
            fontsize=8.8, color=color, fontweight="bold",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([row[0] for row in hero_rows], fontsize=8.4, color=AXIS.ink,
                        rotation=22, ha="right")
    ax.set_xlim(-0.6, len(hero_rows) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("W1 flip rate (%)", fontsize=9.4, color=AXIS.ink_soft)
    ax.set_title("A.  Every model we tested can be flipped",
                 fontsize=10.6, fontweight="bold", loc="left", color=AXIS.ink, pad=8)
    y_grid(ax, alpha=0.55)
    soften_spines(ax)
    ax.tick_params(axis="x", length=0, pad=2)


def _panel_B(ax: plt.Axes, mech_rates):
    shades = {16: "#C36662", 18: "#D99994", 20: "#EDC3BF"}
    markers = {16: "s", 18: "o", 20: "D"}

    for layer in MECH_LAYERS:
        pts = mech_rates[layer]
        alphas = np.array([a for a, _ in pts])
        flips = np.array([v for _, v in pts])
        color = shades[layer]

        xs, ys = _smooth(alphas, flips)
        ax.plot(xs, ys, color=color, linewidth=2.0, zorder=3)
        ax.plot(
            alphas, flips, marker=markers[layer], linestyle="None",
            color=color, markerfacecolor=color,
            markeredgecolor=AXIS.bg_card, markeredgewidth=0.9,
            markersize=6.0, zorder=4, label=f"L{layer}",
        )

    ax.axvline(0, color=AXIS.rule, linestyle=(0, (2, 3)), linewidth=0.8, zorder=1)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0, 40)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Patch magnitude (α)", fontsize=9.2, color=AXIS.ink_soft)
    ax.set_ylabel("Right-to-wrong flips (%)", fontsize=9.4, color=AXIS.ink_soft)
    ax.set_title("B.  A single direction recovers the override  (GPT-oss-20B)",
                 fontsize=10.6, fontweight="bold", loc="left", color=AXIS.ink, pad=8)
    y_grid(ax, alpha=0.55)
    soften_spines(ax)

    leg = ax.legend(
        loc="upper left", frameon=False, fontsize=8.6,
        handlelength=1.2, labelspacing=0.3,
    )
    for t in leg.get_texts():
        t.set_color(AXIS.ink_soft)


def _panel_C(ax: plt.Axes):
    for cfg in TRANSFER_MODELS:
        try:
            alphas, wrong_d = _transfer_wrong_delta(cfg["path"], cfg["layer"])
        except Exception:
            continue

        xs, ys = _smooth(alphas, wrong_d)
        ax.plot(xs, ys, color=cfg["accent"], linewidth=2.0, zorder=3)
        ax.plot(
            alphas, wrong_d, marker=cfg["marker"], linestyle="None",
            color=cfg["accent"], markerfacecolor=cfg["accent"],
            markeredgecolor=AXIS.bg_card, markeredgewidth=0.9,
            markersize=6.0, zorder=4,
            label=f"{cfg['name']}  ·  L{cfg['layer']}",
        )

    ax.axhline(0, color=AXIS.rule, linewidth=0.8, zorder=1)
    ax.axvline(0, color=AXIS.rule, linestyle=(0, (2, 3)), linewidth=0.8, zorder=1)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(-4, 12)
    ax.set_yticks([-4, 0, 4, 8, 12])
    ax.set_xlabel("Patch magnitude (α)", fontsize=9.2, color=AXIS.ink_soft)
    ax.set_ylabel("Δ PIQA wrong-answer rate (pp)", fontsize=9.4, color=AXIS.ink_soft)
    ax.set_title("C.  The same direction transfers to PIQA",
                 fontsize=10.6, fontweight="bold", loc="left", color=AXIS.ink, pad=8)
    y_grid(ax, alpha=0.55)
    soften_spines(ax)

    leg = ax.legend(
        loc="upper left", frameon=False, fontsize=8.4,
        handlelength=1.3, labelspacing=0.35,
    )
    for t in leg.get_texts():
        t.set_color(AXIS.ink_soft)


def main() -> None:
    apply_axis_theme()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hero_rows = _hero_values()
    mech_rates = _mech_flip_rates()

    fig = plt.figure(figsize=(13.5, 4.6))
    fig.patch.set_facecolor(AXIS.bg_canvas)
    gs = fig.add_gridspec(
        1, 3,
        left=0.055, right=0.99,
        bottom=0.22, top=0.88,
        wspace=0.27,
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    for ax in (axA, axB, axC):
        ax.set_facecolor(AXIS.bg_card)

    _panel_A(axA, hero_rows)
    _panel_B(axB, mech_rates)
    _panel_C(axC)

    save_fig(fig, "combined_triptych", OUTPUT_DIR)
    print("Saved combined_triptych.*")


if __name__ == "__main__":
    main()
