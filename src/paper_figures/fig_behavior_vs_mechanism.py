"""Candidate Panel A replacement — behaviour vs. mechanism scatter.

One dot per open-weights model.  The axes separate two distinct claims
the paper makes:

* x-axis: behavioural W1 flip rate — "how much does an authoritative
  wrong endorsement move the model's answer?"
* y-axis: best forward-patch flip rate — "how much of that behaviour
  can we recover with a single mid-layer direction?"

The marker fill encodes the best out-of-domain PIQA Δwrong-answer
rate (third claim: transferability).  The dashed y = x line is the
"mechanism fully captures behaviour" frontier.

Closed-weights models are shown on a narrow behaviour-only strip
below the scatter because we cannot probe their internals.

Run::

    uv run python -m src.paper_figures.fig_behavior_vs_mechanism
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from src.paper_figures.axis_theme import (
    AXIS,
    apply_axis_theme,
    save_fig,
    soften_spines,
    y_grid,
)


RUNS_JSON = Path("neurips-results/_shared/dynamic_parser_all_runs.json")
OUTPUT_DIR = Path("figures/neurips/v2/candidates")


OPEN_WEIGHTS: List[Dict] = [
    {
        "name": "GPT-oss-20B",
        "runs_key": "openai/gpt-oss-20b::v2_gpt_oss_freegen",
        "patch_path": Path("neurips-results/gpt-oss/mechanism/forward_patch_test_256/steering_rows.jsonl"),
        "piqa_path": Path("neurips-results/gpt-oss/mechanism/piqa_forward_patch_200_freegen/piqa_summary.json"),
        "color": "#C36662",
        "marker": "s",
        "label_offset": (14, 8),
    },
    {
        "name": "Gemma-4-26B",
        "runs_key": "google/gemma-4-26B-A4B-it::gemma4__no_thinking_freegen_merged_authoritative",
        "patch_path": Path("neurips-results/gemma4/mechanism/forward_patch_test_256_no_thinking/steering_rows.jsonl"),
        "piqa_path": Path("neurips-results/gemma4/mechanism/piqa_forward_patch_200_freegen_no_thinking/piqa_summary.json"),
        "color": "#6F9C79",
        "marker": "D",
        "label_offset": (14, -10),
    },
    {
        "name": "OLMo-2-32B",
        "runs_key": "allenai/OLMo-2-0325-32B-Instruct::v2_olmo2_freegen",
        "patch_path": Path("neurips-results/olmo2/mechanism/forward_patch_test_256/steering_rows.jsonl"),
        "piqa_path": Path("neurips-results/olmo2/mechanism/piqa_forward_patch_200_freegen/piqa_summary.json"),
        "color": "#D5A048",
        "marker": "^",
        "label_offset": (-14, 10),
    },
]


CLOSED_WEIGHTS: List[Dict] = [
    {
        "name": "GPT-5.4",
        "runs_key": "openai/gpt-5.4::gpt54_native_pool_staged",
        "color": "#8B6F9C",
        "marker": "o",
    },
    {
        "name": "Grok-4.20",
        "runs_key": "x-ai/grok-4.20::grok__grok420_dissociation_freegen",
        "color": "#9A6B4C",
        "marker": "o",
    },
    {
        "name": "Gemini-3.1-Pro",
        "runs_key": "google/gemini-3.1-pro-preview::gemini31pro__dissociation_freegen",
        "color": "#5B84B5",
        "marker": "o",
    },
]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _best_patch_flip(path: Path) -> float:
    grouped = defaultdict(lambda: defaultdict(dict))
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("condition_code") != "N0_note":
            continue
        L = int(r["target_layers"][0])
        a = float(r["alpha"])
        grouped[L][a][r["uid"]] = r
    best = 0.0
    for L, by_a in grouped.items():
        base = by_a.get(0.0, {})
        bc = {uid for uid, r in base.items() if r.get("is_correct") is True}
        if not bc:
            continue
        for a, rs in by_a.items():
            if a == 0.0:
                continue
            flips = sum(1 for uid in bc if rs.get(uid, {}).get("chose_wrong") is True)
            rate = 100.0 * flips / len(bc)
            best = max(best, rate)
    return best


def _best_piqa_wrong(path: Path) -> float:
    rows = [r for r in json.loads(path.read_text()) if r.get("condition_code") == "N0_note"]
    by_L = defaultdict(list)
    for r in rows:
        by_L[int(r["target_layers"][0])].append(r)
    best = -np.inf
    for L, rs in by_L.items():
        rs.sort(key=lambda r: float(r["alpha"]))
        base = next((100.0 * float(r["wrong_rate_parsed"]) for r in rs if float(r["alpha"]) == 0.0), None)
        if base is None:
            continue
        for r in rs:
            if float(r["alpha"]) == 0.0:
                continue
            d = 100.0 * float(r["wrong_rate_parsed"]) - base
            best = max(best, d)
    return float(best)


def _collect(runs: Dict[str, dict]) -> List[Dict]:
    out = []
    for cfg in OPEN_WEIGHTS:
        fr = (runs.get(cfg["runs_key"], {}).get("flip_rate") or 0.0) * 100.0
        pf = _best_patch_flip(cfg["patch_path"])
        qd = _best_piqa_wrong(cfg["piqa_path"])
        out.append({**cfg, "behavior": fr, "mechanism": pf, "piqa": qd})
    return out


def _collect_closed(runs: Dict[str, dict]) -> List[Dict]:
    out = []
    for cfg in CLOSED_WEIGHTS:
        fr = (runs.get(cfg["runs_key"], {}).get("flip_rate") or 0.0) * 100.0
        out.append({**cfg, "behavior": fr})
    return out


# --------------------------------------------------------------------------
# Draw
# --------------------------------------------------------------------------

def main() -> None:
    apply_axis_theme()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = json.loads(RUNS_JSON.read_text())
    rows = _collect(runs)
    closed = _collect_closed(runs)

    fig = plt.figure(figsize=(7.8, 5.8))
    fig.patch.set_facecolor(AXIS.bg_canvas)

    # Layout: main scatter up top, closed-weights strip below with own x-label.
    ax = fig.add_axes([0.13, 0.34, 0.70, 0.50])         # main scatter
    ax_strip = fig.add_axes([0.13, 0.13, 0.70, 0.09])   # closed-weights strip
    ax_cbar = fig.add_axes([0.86, 0.34, 0.022, 0.50])   # colorbar
    for a in (ax, ax_strip):
        a.set_facecolor(AXIS.bg_card)

    x_lim = (0, 100)
    y_lim = (0, 40)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

    # Diagonal "mechanism = behaviour" guide
    ax.plot(
        [0, 40], [0, 40],
        color=AXIS.muted, linestyle=(0, (4, 4)),
        linewidth=0.9, alpha=0.55, zorder=1,
    )
    ax.text(
        40.5, 37.5, "y = x  (mechanism fully explains behaviour)",
        fontsize=7.6, style="italic", color=AXIS.muted,
        ha="left", va="center", rotation=0,
    )

    # Colormap for PIQA transfer
    piqa_vals = [r["piqa"] for r in rows]
    v_min, v_max = 0.0, max(10.0, float(np.ceil(max(piqa_vals))))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "_piqa", ["#FFFFFF", "#E9B0AC", "#A33C36"], N=256,
    )
    norm = mcolors.Normalize(vmin=v_min, vmax=v_max)

    for row in rows:
        fill = cmap(norm(row["piqa"]))
        ax.scatter(
            row["behavior"], row["mechanism"],
            s=360, marker=row["marker"],
            facecolor=fill, edgecolor=row["color"],
            linewidth=2.2, zorder=5,
        )
        dx, dy = row["label_offset"]
        ax.annotate(
            row["name"],
            xy=(row["behavior"], row["mechanism"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=9.4, fontweight="bold",
            color=row["color"],
            ha="left" if dx >= 0 else "right",
            va="center",
        )
        ax.annotate(
            f"ΔPIQA +{row['piqa']:.1f} pp",
            xy=(row["behavior"], row["mechanism"]),
            xytext=(dx, dy - 12), textcoords="offset points",
            fontsize=7.8, color=AXIS.ink_soft,
            ha="left" if dx >= 0 else "right",
            va="center",
        )

    ax.set_ylabel("Best forward-patch flip rate (%)", fontsize=10.0, color=AXIS.ink_soft)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_xticklabels([])
    y_grid(ax, alpha=0.55)
    ax.xaxis.grid(True, color=AXIS.hairline, linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    soften_spines(ax)

    # Colorbar for ΔPIQA
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, cax=ax_cbar)
    cbar.outline.set_edgecolor(AXIS.rule)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(
        color=AXIS.rule, labelcolor=AXIS.ink_soft,
        labelsize=8.0, width=0.7, length=3,
    )
    cbar.set_label(
        "ΔPIQA wrong-answer rate  (pp)",
        fontsize=8.8, color=AXIS.ink_soft,
    )

    # --- closed-weights strip ---------------------------------------------
    ax_strip.set_xlim(*x_lim)
    ax_strip.set_ylim(-1, 1)
    ax_strip.set_yticks([])
    ax_strip.set_xticks([0, 25, 50, 75, 100])
    ax_strip.tick_params(axis="x", labelsize=8.4, labelcolor=AXIS.ink_soft,
                          color=AXIS.rule, length=3)
    for side in ("top", "right", "left"):
        ax_strip.spines[side].set_visible(False)
    ax_strip.spines["bottom"].set_color(AXIS.rule)
    ax_strip.spines["bottom"].set_linewidth(0.8)

    ax_strip.axhline(0, color=AXIS.hairline, linewidth=1.0)

    # Lane title above the strip
    ax_strip.text(
        0, 1.05, "Closed-weights API models  ·  mechanism inaccessible",
        transform=ax_strip.transAxes,
        fontsize=7.8, color=AXIS.muted, style="italic",
        ha="left", va="bottom",
    )

    for cfg in closed:
        ax_strip.scatter(
            cfg["behavior"], 0,
            s=120, marker=cfg["marker"],
            facecolor=AXIS.bg_card, edgecolor=cfg["color"],
            linewidth=1.8, zorder=3,
        )
        ax_strip.annotate(
            f"{cfg['name']}  {cfg['behavior']:.0f}%",
            xy=(cfg["behavior"], 0),
            xytext=(0, -12), textcoords="offset points",
            fontsize=8.0, color=cfg["color"], fontweight="bold",
            ha="center", va="top",
        )

    ax_strip.set_xlabel("Behavioural W1 flip rate (%)", fontsize=9.6, color=AXIS.ink_soft, labelpad=5)

    # Title
    fig.text(
        0.04, 0.93,
        "Behaviour vs. mechanism: the causal handle is not guaranteed",
        fontsize=13.2, fontweight="bold", color=AXIS.ink,
        ha="left", va="bottom",
    )

    save_fig(fig, "behavior_vs_mechanism", OUTPUT_DIR)
    print("Saved behavior_vs_mechanism.*")


if __name__ == "__main__":
    main()
