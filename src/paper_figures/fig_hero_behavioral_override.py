"""Hero figure 1: behavioural authority-override across seven models.

Styled after Anthropic's *Assistant Axis* Fig. 3 — rounded bar patches,
clean inline legend, no titles, bare white canvas, baseline rule at the
foot of each bar group.

Run::

    uv run python -m src.paper_figures.fig_hero_behavioral_override
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

from src.paper_figures.axis_theme import AXIS, apply_axis_theme, save_fig


RESULTS_JSON = Path("neurips-results/_shared/dynamic_parser_all_runs.json")
OUTPUT_DIR = Path("figures/neurips/v2/final")
FONT_DIR = Path("/Users/majortimberwolf/Library/Fonts")


# Qwen-3.5-27B does not have a dissociation run in the shared index yet —
# pull the numbers from the authoritative-verified H100 re-parse
# (`qwen35/authoritative_verified_shared_h100/dynamic_parser_recomputed_summary.md`):
#   N0 acc (parsed) 74.94%, W1 acc (parsed) 49.70%, flip rate 44.87% (608/1355).
QWEN_FREEGEN = {
    "n0_acc": 0.7494,
    "w1_acc": 0.4970,
    "flip": 0.4487,
}


MODELS: List[Dict] = [
    {
        "label": "OLMo-2-32B",
        "run_key": "allenai/OLMo-2-0325-32B-Instruct::v2_olmo2_freegen",
    },
    {
        "label": "GPT-oss-20B",
        "run_key": "openai/gpt-oss-20b::v2_gpt_oss_freegen",
    },
    {
        "label": "Gemma-4-26B",
        "run_key": "google/gemma-4-26B-A4B-it::gemma4__no_thinking_freegen_merged_authoritative",
    },
    {
        "label": "Qwen-3.5-27B",
        "run_key": "__qwen_manual__",
    },
    {
        "label": "GPT-5.4",
        "run_key": "openai/gpt-5.4::gpt54_native_pool_staged",
    },
    {
        "label": "Grok-4.20",
        "run_key": "x-ai/grok-4.20::grok__grok420_dissociation_freegen",
    },
    {
        "label": "Gemini-3.1-Pro",
        "run_key": "google/gemini-3.1-pro-preview::gemini31pro__dissociation_freegen",
    },
]


# Fig-3 palette: charcoal / warm gray / terracotta.
BAR_N0 = "#3B3734"     # near-black, baseline accuracy
BAR_W1 = "#BFB8AD"     # warm gray, accuracy under W1
BAR_FLIP = "#C76E6A"   # red / terracotta, the headline bar

BASELINE_RULE = "#8F8780"
INK = "#2A2724"
INK_SOFT = "#5F5954"


def _compile(run: dict) -> Dict[str, float]:
    return {
        "n0_acc": float(run.get("n0_acc") or 0.0) * 100.0,
        "w1_acc": float(run.get("w1_acc") or 0.0) * 100.0,
        "flip": float(run.get("flip_rate") or 0.0) * 100.0,
    }


def _collect(models: Sequence[Dict], runs: Dict[str, dict]) -> List[Dict]:
    out: List[Dict] = []
    for m in models:
        if m["run_key"] == "__qwen_manual__":
            out.append({
                **m,
                "n0_acc": QWEN_FREEGEN["n0_acc"] * 100.0,
                "w1_acc": QWEN_FREEGEN["w1_acc"] * 100.0,
                "flip":   QWEN_FREEGEN["flip"]   * 100.0,
            })
        else:
            out.append({**m, **_compile(runs.get(m["run_key"], {}))})
    return out


def _round_bar(
    ax: plt.Axes,
    x: float,
    height: float,
    width: float,
    color: str,
    label_fs: float = 8.2,
    corner_frac: float = 0.05,
) -> None:
    """Fig-3 style bar: flat bottom, rounded top corners only.

    Drawn as a single closed Path so there are no seams or overdraw.
    The rounding radius is `corner_frac * width` (data units), clamped
    to at most half the bar's height.
    """
    if height <= 0:
        return

    x0 = x - width / 2.0
    x1 = x + width / 2.0
    y0 = 0.0
    y1 = height

    # Aspect-aware corner radius: in data units we multiply the x-radius
    # by the axes aspect so the top appears circular on the rendered
    # figure rather than stretched.
    fig = ax.figure
    bbox = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    ax_w_in = bbox.width * fig_w
    ax_h_in = bbox.height * fig_h
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    data_w = xlim[1] - xlim[0]
    data_h = ylim[1] - ylim[0]
    # pixels-per-data-unit
    px_x = ax_w_in / data_w
    px_y = ax_h_in / data_h

    r_px = corner_frac * width * px_x
    r_x = r_px / px_x
    r_y = r_px / px_y
    # Clamp so we don't over-round short bars or exceed half-width.
    r_x = min(r_x, width / 2.0 - 1e-6)
    r_y = min(r_y, height - 1e-6, height / 5.0)

    # Build a path: bottom-left → bottom-right → up right side
    # → cubic to top-right corner → top across → cubic to top-left
    # → down to close.
    k = 0.5523  # cubic Bezier circle approximation constant

    verts = [
        (x0, y0),                                 # BL
        (x1, y0),                                 # BR
        (x1, y1 - r_y),                           # up to start of TR curve
        # cubic to (x1 - r_x, y1)
        (x1, y1 - r_y + r_y * k),
        (x1 - r_x + r_x * k, y1),
        (x1 - r_x, y1),
        # straight across top
        (x0 + r_x, y1),
        # cubic to (x0, y1 - r_y)
        (x0 + r_x - r_x * k, y1),
        (x0, y1 - r_y + r_y * k),
        (x0, y1 - r_y),
        # straight down to BL
        (x0, y0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    path = MplPath(verts, codes)
    patch = PathPatch(
        path,
        facecolor=color,
        edgecolor="none",
        linewidth=0,
        zorder=3,
    )
    ax.add_patch(patch)

    # Value label
    ax.text(
        x,
        height + 1.8,
        f"{height:.0f}",
        ha="center",
        va="bottom",
        fontsize=label_fs,
        color=INK_SOFT,
        fontweight="normal",
    )


def _draw_group(
    ax: plt.Axes, centre: float, row: Dict, bar_width: float, gap: float
) -> None:
    triples = [
        (-1, row["n0_acc"], BAR_N0),
        (0,  row["w1_acc"], BAR_W1),
        (1,  row["flip"],   BAR_FLIP),
    ]
    for slot, value, color in triples:
        x = centre + slot * (bar_width + gap)
        _round_bar(ax, x, value, bar_width, color)


def main() -> None:
    apply_axis_theme()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for font_path in (
        FONT_DIR / "Inter-Regular.otf",
        FONT_DIR / "Inter-Medium.otf",
        FONT_DIR / "Inter-SemiBold.otf",
        FONT_DIR / "Inter-Bold.otf",
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))

    runs = json.loads(RESULTS_JSON.read_text())
    rows = _collect(MODELS, runs)

    # Geometry — tighter than before, Fig-3 vibe.
    bar_width = 0.29
    gap = 0.018
    model_span = 3 * bar_width + 2 * gap
    inter_model = 0.72
    step = model_span + inter_model
    centres = np.arange(len(rows)) * step

    x_min = centres[0] - model_span / 2 - 0.35
    x_max = centres[-1] + model_span / 2 + 0.35

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Inter",
        "Inter Regular",
        "Inter Medium",
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "DejaVu Sans",
    ]
    fig = plt.figure(figsize=(12.4, 4.2))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.045, 0.12, 0.93, 0.73])
    ax.set_facecolor("white")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 106)

    for xi, row in zip(centres, rows):
        _draw_group(ax, xi, row, bar_width, gap)

    # Per-group baseline rule across the bar group (Fig-3 signature).
    half_span = model_span / 2 + 0.16
    for xi in centres:
        ax.plot(
            [xi - half_span, xi + half_span],
            [0, 0],
            color=BASELINE_RULE,
            linewidth=0.7,
            zorder=2,
            solid_capstyle="butt",
        )

    # Model labels
    ax.set_xticks(centres)
    ax.set_xticklabels([row["label"] for row in rows], fontsize=8.8, color="#4A453F", fontweight="medium")
    ax.tick_params(axis="x", length=0, pad=8)

    # Strip y-axis chrome — Fig 3 has no y ticks, no grid, no spines.
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    # Inline legend, top-right, no frame.
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BAR_N0,  edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor=BAR_W1,  edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor=BAR_FLIP, edgecolor="none"),
    ]
    labels = [
        "Baseline accuracy (N0)",
        "Accuracy under wrong endorsement (W1)",
        "Right-to-wrong flip rate",
    ]
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
        fontsize=7.9,
        handlelength=0.95,
        handleheight=0.75,
        handletextpad=0.45,
        labelspacing=0.22,
        columnspacing=1.55,
        borderpad=0.0,
    )
    for txt in leg.get_texts():
        txt.set_color(INK_SOFT)

    save_fig(fig, "hero_behavioral_override", OUTPUT_DIR)
    print("Saved hero_behavioral_override.*")


if __name__ == "__main__":
    main()
