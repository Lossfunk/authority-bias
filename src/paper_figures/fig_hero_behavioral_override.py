"""Hero figure 1: behavioural authority-override across seven models.

Styled after Anthropic's *Assistant Axis* Fig. 3 — rounded bar patches,
clean inline legend, no titles, bare white canvas, baseline rule at the
foot of each bar group.

Run::

    uv run python -m src.paper_figures.fig_hero_behavioral_override
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

from src.paper_figures.axis_theme import AXIS, apply_axis_theme, save_fig


OUTPUT_DIR = Path("figures/neurips/v2/final")
FONT_DIR = Path("/Users/majortimberwolf/Library/Fonts")


# Hard-coded plotting data for the final NeurIPS hero figure.
#
# Units:
#   - n0_acc / w1_acc / flip are percentages, not fractions.
#   - n0_correct is the denominator for the flip-rate Wilson interval.
#   - w1_flips is the numerator for the flip-rate Wilson interval.
#
# The first four entries form the top row; the remaining entries form the
# lower row.  This keeps open-weight models on top and the closed-source
# frontier models on the lower row.
HERO_ROWS: List[Dict] = [
    {
        "label": "OLMo-3.1-32B",
        "n0_acc": 46.83,
        "w1_acc": 9.49,
        "flip": 82.88,
        "n0_correct": 146,
        "w1_flips": 121,
    },
    {
        "label": "GPT-oss-20B",
        "n0_acc": 77.06766917293233,
        "w1_acc": 28.085642317380355,
        "flip": 65.359477124183,
        "n0_correct": 612,
        "w1_flips": 400,
    },
    {
        "label": "Gemma-4-26B",
        "n0_acc": 90.02320185614849,
        "w1_acc": 33.21799307958477,
        "flip": 62.87978863936592,
        "n0_correct": 757,
        "w1_flips": 476,
    },
    {
        "label": "Qwen-3.5-27B",
        "n0_acc": 74.94,
        "w1_acc": 49.70,
        "flip": 44.87,
        "n0_correct": 1355,
        "w1_flips": 608,
    },
    {
        "label": "GPT-5.4",
        "n0_acc": 91.87219730941703,
        "w1_acc": 57.36102626756261,
        "flip": 42.61992619926199,
        "n0_correct": 1626,
        "w1_flips": 693,
    },
    {
        "label": "Grok-4.20",
        "n0_acc": 90.65420560747664,
        "w1_acc": 11.078199052132701,
        "flip": 87.4829001367989,
        "n0_correct": 1462,
        "w1_flips": 1279,
    },
]


# Fig-3 palette: charcoal / warm gray / terracotta.
BAR_N0 = "#3B3734"       # near-black, normalized baseline-correct set
BAR_W1 = "#BFB8AD"       # warm gray, remains correct under W1
ARROW_FLIP = "#C76E6A"   # red / terracotta, right-to-wrong decrease arrow

BASELINE_RULE = "#8F8780"
INK = "#2A2724"
INK_SOFT = "#5F5954"


def _wilson_ci_pct(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return 100.0 * max(0.0, centre - half), 100.0 * min(1.0, centre + half)


def _round_bar(
    ax: plt.Axes,
    x: float,
    height: float,
    width: float,
    color: str,
    label_fs: float = 8.2,
    corner_frac: float = 0.05,
    bottom: float = 0.0,
    label: str | None = None,
    label_y: float | None = None,
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
    y0 = bottom
    y1 = bottom + height

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

    if label is not None:
        ax.text(
            x,
            (y1 + 1.8) if label_y is None else label_y,
            label,
            ha="center",
            va="bottom",
            fontsize=label_fs,
            color=INK_SOFT,
            fontweight="normal",
            linespacing=1.05,
        )


def _draw_group(
    ax: plt.Axes, centre: float, row: Dict, bar_width: float, gap: float
) -> None:
    baseline_x = centre - (bar_width * 0.78 + gap)
    outcome_x = centre + (bar_width * 0.78 + gap)
    flip = float(row["flip"])
    retained = max(0.0, 100.0 - flip)
    flip_ci_lo, flip_ci_hi = _wilson_ci_pct(int(row["w1_flips"]), int(row["n0_correct"]))
    retained_ci_lo = max(0.0, 100.0 - flip_ci_hi)
    retained_ci_hi = min(100.0, 100.0 - flip_ci_lo)

    _round_bar(
        ax,
        baseline_x,
        100.0,
        bar_width,
        BAR_N0,
        label=f"100\n({row['n0_acc']:.0f}% of all)",
        label_fs=9.6,
    )
    _round_bar(
        ax,
        outcome_x,
        retained,
        bar_width,
        BAR_W1,
        label=f"{retained:.0f}",
        label_fs=9.6,
    )
    ax.errorbar(
        outcome_x,
        retained,
        yerr=[[max(0.0, retained - retained_ci_lo)], [max(0.0, retained_ci_hi - retained)]],
        fmt="none",
        ecolor="#7E766E",
        elinewidth=0.75,
        capsize=2.4,
        capthick=0.75,
        zorder=5,
    )
    arrow_x = outcome_x + bar_width * 0.52
    arrow_top = 99.5
    arrow_bottom = max(retained + 1.4, 2.0)
    if flip >= 3.0:
        # Use an editable text glyph rather than a Matplotlib arrow patch.
        # This keeps the SVG easy to select and move in Figma.
        ax.text(
            arrow_x,
            (arrow_top + arrow_bottom) / 2,
            "↓",
            ha="center",
            va="center",
            fontsize=28,
            color=ARROW_FLIP,
            fontweight="medium",
            zorder=5,
        )
        label_y = max((arrow_top + arrow_bottom) / 2, retained + 8)
        ax.text(
            arrow_x + 0.045,
            label_y,
            f"{flip:.0f}% flip",
            ha="left",
            va="center",
            fontsize=9.0,
            color=ARROW_FLIP,
            fontweight="medium",
            zorder=6,
        )
    else:
        ax.text(
            arrow_x,
            99.0,
            "–",
            ha="center",
            va="center",
            fontsize=15,
            color=ARROW_FLIP,
            fontweight="medium",
            zorder=5,
        )
        ax.text(
            arrow_x + 0.045,
            99.0,
            f"{flip:.0f}% flip",
            ha="left",
            va="center",
            fontsize=9.0,
            color=ARROW_FLIP,
            fontweight="medium",
            zorder=6,
        )


def main() -> None:
    apply_axis_theme()
    # Keep SVG text as editable <text> nodes for Figma/Illustrator workflows.
    plt.rcParams["svg.fonttype"] = "none"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for font_path in (
        FONT_DIR / "Inter-Regular.otf",
        FONT_DIR / "Inter-Medium.otf",
        FONT_DIR / "Inter-SemiBold.otf",
        FONT_DIR / "Inter-Bold.otf",
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))

    rows = HERO_ROWS

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
    fig = plt.figure(figsize=(6.8, 4.9))
    fig.patch.set_facecolor("white")
    axes = [
        fig.add_axes([0.105, 0.54, 0.86, 0.31]),
        fig.add_axes([0.105, 0.14, 0.86, 0.31]),
    ]

    row_chunks = [rows[:4], rows[4:]]
    for ax, chunk in zip(axes, row_chunks):
        # Geometry: four models per row so the final TeX figure remains readable.
        bar_width = 0.33
        gap = 0.12
        model_span = 2 * bar_width + gap
        inter_model = 0.74
        step = model_span + inter_model
        centres = np.arange(len(chunk)) * step
        x_min = centres[0] - model_span / 2 - 0.22
        x_max = centres[-1] + model_span / 2 + 0.34

        ax.set_facecolor("white")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 112)

        for xi, row in zip(centres, chunk):
            _draw_group(ax, xi, row, bar_width, gap)

        half_span = model_span / 2 + 0.14
        for xi in centres:
            ax.plot(
                [xi - half_span, xi + half_span],
                [0, 0],
                color=BASELINE_RULE,
                linewidth=0.7,
                zorder=2,
                solid_capstyle="butt",
            )

        ax.set_xticks(centres)
        ax.set_xticklabels([row["label"] for row in chunk], fontsize=9.2, color="#4A453F", fontweight="medium")
        ax.tick_params(axis="x", length=0, pad=5)
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(["0", "50", "100"], fontsize=8.6, color="#7A736C")
        ax.yaxis.grid(True, color="#EEEAE4", linewidth=0.7, zorder=0)
        ax.tick_params(axis="y", length=0, pad=4)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#D8D1C8")
        ax.spines["left"].set_linewidth(0.7)

    fig.text(
        0.035,
        0.48,
        "% of baseline-correct answers",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9.4,
        color=INK_SOFT,
    )

    # Inline legend, top-right, no frame.
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BAR_N0,  edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor=BAR_W1,  edgecolor="none"),
        plt.Line2D([0], [0], color=ARROW_FLIP, marker=r"$\downarrow$", markersize=8, linewidth=1.2),
    ]
    labels = [
        "Baseline-correct = 100",
        "Still correct under wrong cue",
        "Flip to wrong answer",
    ]
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
        fontsize=8.8,
        handlelength=1.05,
        handleheight=0.8,
        handletextpad=0.48,
        labelspacing=0.35,
        columnspacing=1.15,
        borderpad=0.0,
    )
    for txt in leg.get_texts():
        txt.set_color(INK_SOFT)

    paths = save_fig(fig, "hero_behavioral_override", OUTPUT_DIR)
    for path in paths:
        if path.suffix == ".svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print("Saved hero_behavioral_override.*")


if __name__ == "__main__":
    main()
