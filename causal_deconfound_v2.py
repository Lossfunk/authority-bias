"""
Causal deconfound bar chart — Anthropic-style restyle.

Side-by-side panels (Factual QA, PIQA), bars going UP — y-axis is the
reduction in wrong-answer rate after each projection-removal. Small
dips below zero are kept honest (a few interventions slightly
increased wrong-rate, the opposite of help).

Style cues mirror the prior Fig-3-style hero:
  - cream rounded card behind everything
  - rounded "outside" corner only (top for positives, bottom for dips)
  - no gridlines, no y-ticks; value labels and a single zero rule
  - Inter / Styrene fallback, square swatches in a tight top-right legend
"""

from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


# ---------- Style -------------------------------------------------------

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Inter",
    "Styrene B",
    "StyreneB",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "DejaVu Sans",
]
mpl.rcParams["font.size"] = 9.5
mpl.rcParams["axes.unicode_minus"] = True
mpl.rcParams["svg.fonttype"] = "none"   # editable text in Figma/Illustrator
mpl.rcParams["pdf.fonttype"] = 42

INK_PRIMARY   = "#1F1F1E"
INK_SECONDARY = "#5F5E5A"
INK_TERTIARY  = "#888780"

# Coral pair for the two strong interventions, muted gray for the null.
C_ASSISTANT = "#9B9A92"   # muted gray — the null
C_AUTHORITY = "#993C1D"   # dark coral — strong intervention
C_RESID     = "#D85A30"   # lighter coral — paired with authority

GRID_COLOR = "#D8D5CD"
ZERO_LINE  = "#5F5954"


# ---------- Data --------------------------------------------------------

FACTUAL_QA = [
    dict(model="Qwen-3.5",  baseline=39.4, assistant=-1.0, authority= 33.1, resid= 33.6),
    dict(model="OLMo-2",    baseline=83.9, assistant=-1.1, authority= 20.8, resid= 21.1),
    dict(model="OLMo-3.1",  baseline=81.6, assistant=-0.1, authority= 19.8, resid= 20.1),
    dict(model="GPT-OSS",   baseline=56.9, assistant=-0.5, authority= 16.7, resid= 16.4),
    dict(model="Gemma-4",   baseline=57.8, assistant= 2.9, authority=  2.7, resid=  3.4),
]

PIQA = [
    dict(model="Qwen-3.5",  baseline=58.7, assistant=-0.4, authority= 53.3, resid= 54.2),
    dict(model="OLMo-2",    baseline=89.0, assistant= 0.0, authority= 20.1, resid= 20.2),
    dict(model="OLMo-3.1",  baseline=96.1, assistant=-1.4, authority= 18.7, resid= 19.5),
    dict(model="GPT-OSS",   baseline=86.5, assistant=-0.3, authority= 14.8, resid= 14.6),
    dict(model="Gemma-4",   baseline=97.1, assistant=-0.6, authority=  0.0, resid=  0.0),
]


# ---------- Bar (rounded outside corner only) --------------------------

def _round_bar(
    ax: plt.Axes,
    x_center: float,
    width: float,
    value: float,
    color: str,
    *,
    corner_frac: float = 0.18,
    min_visible: float = 0.55,
) -> None:
    """Bar with rounded "outside" corners only.

    Positive bars round at the top (zero edge is flat); negative bars
    round at the bottom (zero edge is flat). The corner radius is
    `corner_frac * width` in data x-units, made aspect-aware so the
    rendered corner reads as roughly circular rather than squashed.
    Tiny values are floored to `min_visible` so near-zero dips remain
    visible without distorting magnitudes.
    """
    if value >= 0:
        y0 = 0.0
        h = max(value, min_visible)
        rounded_top = True
    else:
        y0 = float(value)
        h = max(-float(value), min_visible)
        # If we floored a tiny dip, push y0 up so the bottom of the bar
        # stays anchored at the (sign-correct) value end.
        if -float(value) < min_visible:
            y0 = -min_visible
        rounded_top = False  # round bottom corners

    x0 = x_center - width / 2.0
    x1 = x_center + width / 2.0
    y1 = y0 + h

    # Aspect-aware radius: pixels-per-data-unit on each axis
    fig = ax.figure
    bbox = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    ax_w_in = bbox.width * fig_w
    ax_h_in = bbox.height * fig_h
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    px_x = ax_w_in / (xlim[1] - xlim[0])
    px_y = ax_h_in / (ylim[1] - ylim[0])
    r_px = corner_frac * width * px_x
    r_x = min(r_px / px_x, width / 2.0 - 1e-6)
    r_y = min(r_px / px_y, h - 1e-6, h / 2.0)

    k = 0.5523  # cubic Bezier circle approximation

    if rounded_top:
        # BL → BR → up → curve TR → across → curve TL → down → close
        verts = [
            (x0, y0),
            (x1, y0),
            (x1, y1 - r_y),
            (x1, y1 - r_y + r_y * k),
            (x1 - r_x + r_x * k, y1),
            (x1 - r_x, y1),
            (x0 + r_x, y1),
            (x0 + r_x - r_x * k, y1),
            (x0, y1 - r_y + r_y * k),
            (x0, y1 - r_y),
            (x0, y0),
        ]
    else:
        # TL → TR → down → curve BR → across → curve BL → up → close
        verts = [
            (x0, y1),
            (x1, y1),
            (x1, y0 + r_y),
            (x1, y0 + r_y - r_y * k),
            (x1 - r_x + r_x * k, y0),
            (x1 - r_x, y0),
            (x0 + r_x, y0),
            (x0 + r_x - r_x * k, y0),
            (x0, y0 + r_y - r_y * k),
            (x0, y0 + r_y),
            (x0, y1),
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
    patch = PathPatch(
        MplPath(verts, codes),
        facecolor=color,
        edgecolor="none",
        linewidth=0,
        zorder=3,
    )
    ax.add_patch(patch)


def fmt_pp(v: float) -> str:
    if abs(v) < 0.05:
        return "0"
    return f"{v:.1f}"


# ---------- Panel -------------------------------------------------------

def draw_panel(ax, data, panel_label, *, show_y_labels=True):
    n = len(data)
    bar_w = 0.32
    gap   = 0.025
    # Two bars per group now: assistant axis (null) + residualized
    # authority (headline). The plain "remove authority" was dropped
    # because it produced near-identical results to the residualized
    # version (within ~1 pp across all conditions) and showed no extra
    # information.
    offsets = np.array([-(bar_w / 2 + gap / 2), +(bar_w / 2 + gap / 2)])
    x_centers = np.arange(n)

    ax.set_facecolor("white")
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_ylim(-5, 62)

    # Y-axis: ticks + faint dotted gridlines.
    yticks = [0, 10, 20, 30, 40, 50, 60]
    ax.set_yticks(yticks)
    if show_y_labels:
        ax.set_yticklabels([str(t) for t in yticks])
        ax.tick_params(axis="y", labelsize=9.5, length=0, pad=4,
                       colors=INK_SECONDARY)
        for tick_lab, val in zip(ax.get_yticklabels(), yticks):
            tick_lab.set_color(INK_PRIMARY if val == 0 else INK_SECONDARY)
        ax.set_ylabel(
            "Reduction in wrong-answer rate (pp)   ·   higher is better",
            fontsize=10.5,
            color=INK_PRIMARY,
            labelpad=10,
        )
    else:
        ax.set_yticklabels([""] * len(yticks))
        ax.tick_params(axis="y", length=0)

    ax.set_xticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)

    for y in (10, 20, 30, 40, 50, 60):
        ax.axhline(y, color=GRID_COLOR, linewidth=0.5,
                   linestyle=(0, (1, 2.5)), zorder=0)
    ax.axhline(0, color=ZERO_LINE, linewidth=0.9, zorder=1)

    for i, d in enumerate(data):
        pairs = [
            (d["assistant"], C_ASSISTANT),
            (d["resid"],     C_RESID),
        ]
        for j, (v, c) in enumerate(pairs):
            x = x_centers[i] + offsets[j]
            _round_bar(ax, x, bar_w, v, c)

            label = fmt_pp(v)
            if v >= 0:
                ly = max(v, 0.55) + 1.0
                va = "bottom"
                weight = "medium" if abs(v) >= 10 else "normal"
            else:
                ly = v - 0.9
                va = "top"
                weight = "normal"
            ax.text(
                x, ly, label,
                ha="center", va=va,
                fontsize=8.4,
                color=INK_PRIMARY,
                fontweight=weight,
            )

    # Panel title — bigger, sits above the plot
    ax.text(
        -0.04, 1.04, panel_label,
        transform=ax.transAxes,
        fontsize=16.5, color=INK_PRIMARY, fontweight="medium",
        ha="left", va="bottom",
    )

    # Model labels + baselines — pulled closer to the zero rule
    for i, d in enumerate(data):
        ax.text(
            i, -0.025, d["model"],
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            fontsize=10.5, color=INK_PRIMARY, fontweight="medium",
        )
        ax.text(
            i, -0.085, f'baseline {d["baseline"]:.1f}%',
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            fontsize=8.8, color=INK_TERTIARY,
        )


# ---------- Figure ------------------------------------------------------

def main(out_dir="/home/claude/fig_out"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.2, 5.0))
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(
        1, 2,
        wspace=0.10,
        top=0.78, bottom=0.14,
        left=0.075, right=0.985,
    )
    ax_left  = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    draw_panel(ax_left,  FACTUAL_QA, "Factual QA", show_y_labels=True)
    draw_panel(ax_right, PIQA,        "PIQA",       show_y_labels=False)

    # Legend — top-right, square swatches, single column.
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=C_ASSISTANT, edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_RESID,     edgecolor="none"),
    ]
    labels = [
        "Remove assistant axis",
        "Remove residualized authority",
    ]
    leg = fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.978, 0.96),
        ncol=1,
        frameon=False,
        fontsize=9.0,
        handlelength=0.9,
        handleheight=0.9,
        handletextpad=0.55,
        labelspacing=0.55,
        borderpad=0.0,
    )
    for txt in leg.get_texts():
        txt.set_color(INK_PRIMARY)

    png = out_dir / "causal_deconfound_v7.png"
    pdf = out_dir / "causal_deconfound_v7.pdf"
    svg = out_dir / "causal_deconfound_v7.svg"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    fig.savefig(svg, facecolor="white")
    print(f"Saved {png}")
    print(f"Saved {pdf}")
    print(f"Saved {svg}")
    return fig


if __name__ == "__main__":
    main()
