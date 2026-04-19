"""Candidate Panel A replacement — compact results matrix.

One compact cell per (model × signal) with numeric values and a
columnwise colour gradient.  Columns:

1. Behavioural W1 flip rate  (all 6 models)
2. Best forward-patch flip rate — the causal handle (open-weights only)
3. Best PIQA Δwrong-answer rate — out-of-domain transfer (open-weights only)

API-only rows show a diagonal hatch in the mechanism columns to
signal "inaccessible" rather than "missing".

Run::

    uv run python -m src.paper_figures.fig_results_matrix
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
)


RUNS_JSON = Path("neurips-results/_shared/dynamic_parser_all_runs.json")
OUTPUT_DIR = Path("figures/neurips/v2/candidates")


MODELS: List[Dict] = [
    {
        "name": "OLMo-2-32B",
        "color": "#D5A048",
        "runs_key": "allenai/OLMo-2-0325-32B-Instruct::v2_olmo2_freegen",
        "patch_path": Path("neurips-results/olmo2/mechanism/forward_patch_test_256/steering_rows.jsonl"),
        "piqa_path": Path("neurips-results/olmo2/mechanism/piqa_forward_patch_200_freegen/piqa_summary.json"),
        "closed": False,
    },
    {
        "name": "GPT-oss-20B",
        "color": "#C36662",
        "runs_key": "openai/gpt-oss-20b::v2_gpt_oss_freegen",
        "patch_path": Path("neurips-results/gpt-oss/mechanism/forward_patch_test_256/steering_rows.jsonl"),
        "piqa_path": Path("neurips-results/gpt-oss/mechanism/piqa_forward_patch_200_freegen/piqa_summary.json"),
        "closed": False,
    },
    {
        "name": "Gemma-4-26B",
        "color": "#6F9C79",
        "runs_key": "google/gemma-4-26B-A4B-it::gemma4__no_thinking_freegen_merged_authoritative",
        "patch_path": Path("neurips-results/gemma4/mechanism/forward_patch_test_256_no_thinking/steering_rows.jsonl"),
        "piqa_path": Path("neurips-results/gemma4/mechanism/piqa_forward_patch_200_freegen_no_thinking/piqa_summary.json"),
        "closed": False,
    },
    {
        "name": "GPT-5.4",
        "color": "#8B6F9C",
        "runs_key": "openai/gpt-5.4::gpt54_native_pool_staged",
        "closed": True,
    },
    {
        "name": "Grok-4.20",
        "color": "#9A6B4C",
        "runs_key": "x-ai/grok-4.20::grok__grok420_dissociation_freegen",
        "closed": True,
    },
    {
        "name": "Gemini-3.1-Pro",
        "color": "#5B84B5",
        "runs_key": "google/gemini-3.1-pro-preview::gemini31pro__dissociation_freegen",
        "closed": True,
    },
]


COLUMNS = [
    {"key": "behavior", "label": "Behavioural\nW1 flip rate", "unit": "%", "vmax": 100.0, "cmap": "behavior"},
    {"key": "mechanism", "label": "Best forward-patch\nflip rate", "unit": "%", "vmax": 40.0, "cmap": "mechanism"},
    {"key": "piqa", "label": "Best PIQA Δwrong\n(out-of-domain)", "unit": "pp", "vmax": 12.0, "cmap": "piqa"},
]


def _best_patch_flip(path: Path) -> float:
    grouped = defaultdict(lambda: defaultdict(dict))
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("condition_code") != "N0_note":
            continue
        L = int(r["target_layers"][0])
        grouped[L][float(r["alpha"])][r["uid"]] = r
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
            best = max(best, 100.0 * flips / len(bc))
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
            best = max(best, 100.0 * float(r["wrong_rate_parsed"]) - base)
    return float(best)


def _collect(runs: Dict[str, dict]) -> List[Dict]:
    out = []
    for cfg in MODELS:
        fr = (runs.get(cfg["runs_key"], {}).get("flip_rate") or 0.0) * 100.0
        rec = {"name": cfg["name"], "color": cfg["color"], "closed": cfg["closed"], "behavior": fr}
        if not cfg["closed"]:
            rec["mechanism"] = _best_patch_flip(cfg["patch_path"])
            rec["piqa"] = _best_piqa_wrong(cfg["piqa_path"])
        else:
            rec["mechanism"] = None
            rec["piqa"] = None
        out.append(rec)
    return out


def _column_cmaps():
    return {
        "behavior": mcolors.LinearSegmentedColormap.from_list(
            "_col_behavior", ["#FBFBFB", "#F0C3BF", "#C36662"], N=256,
        ),
        "mechanism": mcolors.LinearSegmentedColormap.from_list(
            "_col_mechanism", ["#FBFBFB", "#D2D6E4", "#5B84B5"], N=256,
        ),
        "piqa": mcolors.LinearSegmentedColormap.from_list(
            "_col_piqa", ["#FBFBFB", "#E2CBDC", "#8B6F9C"], N=256,
        ),
    }


def main() -> None:
    apply_axis_theme()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = json.loads(RUNS_JSON.read_text())
    rows = _collect(runs)
    cmaps = _column_cmaps()

    fig = plt.figure(figsize=(9.2, 5.8))
    fig.patch.set_facecolor(AXIS.bg_canvas)
    ax = fig.add_axes([0.26, 0.10, 0.70, 0.70])
    ax.set_facecolor(AXIS.bg_card)

    n_rows = len(rows)
    n_cols = len(COLUMNS)

    cell_w, cell_h = 1.0, 1.0
    ax.set_xlim(0, n_cols * cell_w)
    ax.set_ylim(0, n_rows * cell_h)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    # Column headers
    for c, col in enumerate(COLUMNS):
        ax.text(
            c * cell_w + cell_w / 2, -0.18,
            col["label"],
            ha="center", va="bottom",
            fontsize=9.6, fontweight="bold", color=AXIS.ink,
        )

    # Row labels (model names)
    for r, row in enumerate(rows):
        ax.text(
            -0.08, r * cell_h + cell_h / 2,
            row["name"],
            ha="right", va="center",
            fontsize=9.6, fontweight="bold", color=row["color"],
        )

    # Section divider between open-weights and closed
    first_closed = next((i for i, r in enumerate(rows) if r["closed"]), None)
    if first_closed is not None:
        y = first_closed * cell_h
        ax.plot(
            [0, n_cols * cell_w], [y, y],
            color=AXIS.rule, linewidth=1.0, zorder=4,
        )

    # Section labels positioned well to the LEFT of the model-name column
    # so they don't collide with model names.  We use figure coords.
    bbox = ax.get_position()
    x_section = bbox.x0 - 0.135
    if first_closed is not None:
        # y in figure coords — axis y inverted, so higher data y = lower figure y
        open_mid_data = (first_closed - 1) / 2 * cell_h + cell_h / 2
        closed_mid_data = (first_closed + (n_rows - first_closed - 1) / 2) * cell_h + cell_h / 2
        # Transform data -> display -> figure
        trans = ax.transData.transform
        inv = fig.transFigure.inverted().transform
        y_open = inv(trans((0, open_mid_data)))[1]
        y_closed = inv(trans((0, closed_mid_data)))[1]

        for label, y_fig in [("OPEN\nWEIGHTS", y_open), ("CLOSED\nWEIGHTS\n(API)", y_closed)]:
            fig.text(
                x_section, y_fig, label,
                ha="center", va="center",
                fontsize=7.4, color=AXIS.muted, fontweight="bold",
            )

    # Cells
    pad = 0.06
    for r, row in enumerate(rows):
        for c, col in enumerate(COLUMNS):
            value = row[col["key"]]
            x0, y0 = c * cell_w + pad, r * cell_h + pad
            w, h = cell_w - 2 * pad, cell_h - 2 * pad

            if value is None:
                # Hatched "inaccessible" cell
                ax.add_patch(
                    plt.Rectangle(
                        (x0, y0), w, h,
                        facecolor="#F5F5F5",
                        edgecolor=AXIS.rule,
                        linewidth=0.8,
                        hatch="///",
                        zorder=2,
                    )
                )
                ax.text(
                    x0 + w / 2, y0 + h / 2,
                    "N/A",
                    ha="center", va="center",
                    fontsize=9.0, color=AXIS.muted, style="italic",
                    zorder=3,
                )
                continue

            norm = mcolors.Normalize(vmin=0.0, vmax=col["vmax"])
            cmap = cmaps[col["cmap"]]
            face = cmap(norm(max(0.0, value)))
            ax.add_patch(
                plt.Rectangle(
                    (x0, y0), w, h,
                    facecolor=face,
                    edgecolor=AXIS.rule,
                    linewidth=0.7,
                    zorder=2,
                )
            )
            # Decide text colour: use dark ink on light fills, otherwise white.
            fill_lum = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]
            text_color = AXIS.ink if fill_lum > 0.65 else "#FFFFFF"
            ax.text(
                x0 + w / 2, y0 + h / 2,
                f"{value:.0f}{col['unit']}" if col["unit"] == "%" else f"+{value:.1f} {col['unit']}",
                ha="center", va="center",
                fontsize=11.0, fontweight="bold", color=text_color,
                zorder=3,
            )

    # Title
    fig.text(
        0.04, 0.92,
        "Results matrix:  behaviour, mechanism, and out-of-domain transfer",
        fontsize=13.2, fontweight="bold", color=AXIS.ink,
        ha="left", va="bottom",
    )
    fig.text(
        0.04, 0.885,
        "Each column uses its own colour scale.  "
        "Hatched cells mark signals that require internal access we do not have for closed-weights models.",
        fontsize=8.8, color=AXIS.muted,
        ha="left", va="bottom", style="italic",
    )

    save_fig(fig, "results_matrix", OUTPUT_DIR)
    print("Saved results_matrix.*")


if __name__ == "__main__":
    main()
