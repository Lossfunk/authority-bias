"""Assistant-Axis-inspired editorial theme for NeurIPS figures.

Design cues lifted from Anthropic's *Assistant Axis* figures but adapted
to a neutral white canvas for NeurIPS:

* Pure white canvas and card (`#FFFFFF`).
* Muted blue / red pair for semantic direction (assistant vs role-playing).
* Light faint-gray axis ticks, axis spines drawn in a single soft tone.
* Optional vertical blue→red gradient shading for "above/below baseline".
* Bold italic semantic axis labels with arrowheads at each pole.

Usage::

    from src.paper_figures.axis_theme import (
        apply_axis_theme, AXIS, card_axes, semantic_y_gradient,
        semantic_y_poles, annotate_endpoint, save_fig,
    )

    apply_axis_theme()
    fig, ax = card_axes(figsize=(7, 4))
    ...
    save_fig(fig, "my_figure", output_dir)

The module is deliberately standalone and does not mutate `theme.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import FancyBboxPatch


# ═══════════════════════════════════════════════════════════════════
# 1. Palette
# ═══════════════════════════════════════════════════════════════════


class _Palette:
    # Canvas + card (NeurIPS-friendly white)
    bg_canvas = "#FFFFFF"  # page background
    bg_card = "#FFFFFF"    # card surface

    # Semantic axis pair
    blue = "#5B84B5"       # cool, muted steel
    red = "#C36662"        # dusty terracotta
    blue_dim = "#B7CBE1"
    red_dim = "#ECC6C2"
    blue_soft = "#EAF1F8"  # very faint blue tint
    red_soft = "#FBECEA"   # very faint red tint

    # Text
    ink = "#1F1D1C"
    ink_soft = "#4A4744"
    muted = "#8C857E"
    rule = "#BEB9B1"       # faint spine / axis
    hairline = "#E8E5E0"   # gridlines / card outlines

    # Qualitative palette (used for model curves, tags, …)
    #   blue (Llama-ish), red (Qwen-ish), green, amber, purple, brown
    qualitative: List[str] = [
        "#5B84B5",  # steel blue
        "#C36662",  # terracotta
        "#6F9C79",  # sage green
        "#D5A048",  # mustard gold
        "#8B6F9C",  # dusty purple
        "#9A6B4C",  # cocoa brown
    ]

    # Model-specific style dictionary.  Adjusted to map cleanly to the
    # NeurIPS paper lineup while staying visually distinct on the white
    # background.  Markers chosen so the figure reads at 300 DPI.
    MODEL_STYLES = {
        "gpt-oss-20b": {
            "color": "#C36662",
            "marker": "s",
            "label": "GPT-oss-20B",
            "short": "GPT-oss",
        },
        "gemma4": {
            "color": "#6F9C79",
            "marker": "D",
            "label": "Gemma-4-26B",
            "short": "Gemma-4",
        },
        "olmo2": {
            "color": "#D5A048",
            "marker": "^",
            "label": "OLMo-2-32B",
            "short": "OLMo-2",
        },
        "gpt54": {
            "color": "#8B6F9C",
            "marker": "o",
            "label": "GPT-5.4",
            "short": "GPT-5.4",
        },
        "gemini31pro": {
            "color": "#5B84B5",
            "marker": "P",
            "label": "Gemini-3.1-Pro",
            "short": "Gemini",
        },
        "grok420": {
            "color": "#9A6B4C",
            "marker": "X",
            "label": "Grok-4.20",
            "short": "Grok",
        },
    }

    # Authority-framing gradient colour scale (light → strong).  We use
    # a blended ramp rather than a perceptually-uniform sequential map so
    # the earliest level (None) visually reads as "neutral".
    gradient_levels = [
        "#C8C0B6",
        "#E0B89A",
        "#D89877",
        "#C66F5D",
        "#A9412F",
    ]


AXIS = _Palette()


# ═══════════════════════════════════════════════════════════════════
# 2. rcParams
# ═══════════════════════════════════════════════════════════════════


_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 450,
    "figure.facecolor": AXIS.bg_canvas,
    "savefig.facecolor": AXIS.bg_canvas,
    "savefig.edgecolor": "none",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.18,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Inter",
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
    ],
    "font.size": 9.2,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlepad": 9,
    "axes.labelsize": 9.2,
    "axes.labelpad": 5,
    "axes.labelcolor": AXIS.ink,
    "axes.facecolor": AXIS.bg_card,
    "axes.edgecolor": AXIS.rule,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.grid": False,
    "axes.prop_cycle": mpl.cycler(color=AXIS.qualitative),
    "xtick.color": AXIS.rule,
    "ytick.color": AXIS.rule,
    "xtick.labelcolor": AXIS.ink_soft,
    "ytick.labelcolor": AXIS.ink_soft,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "grid.color": AXIS.hairline,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.85,
    "legend.frameon": False,
    "legend.fontsize": 8.4,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.35,
    "lines.linewidth": 1.9,
    "lines.markersize": 5,
    "text.color": AXIS.ink,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_axis_theme() -> None:
    """Install the editorial rcParams globally."""
    plt.rcParams.update(_RC)


# ═══════════════════════════════════════════════════════════════════
# 3. Card / region helpers
# ═══════════════════════════════════════════════════════════════════


def card_axes(
    figsize: Tuple[float, float] = (7.0, 4.2),
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool = False,
    sharey: bool = False,
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
):
    """Create a figure + axes laid out on the white canvas.

    Each axis already uses the card face colour from rcParams.
    """
    apply_axis_theme()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        gridspec_kw=(
            {
                "width_ratios": width_ratios,
                "height_ratios": height_ratios,
            }
            if width_ratios or height_ratios
            else None
        ),
        constrained_layout=True,
    )
    fig.patch.set_facecolor(AXIS.bg_canvas)
    return fig, axes


def soften_spines(ax: plt.Axes, color: str = AXIS.rule) -> None:
    for side in ("left", "bottom"):
        ax.spines[side].set_color(color)
        ax.spines[side].set_linewidth(0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def y_grid(ax: plt.Axes, alpha: float = 0.55) -> None:
    ax.yaxis.grid(True, color=AXIS.hairline, linewidth=0.6, alpha=alpha)
    ax.set_axisbelow(True)


def x_grid(ax: plt.Axes, alpha: float = 0.55) -> None:
    ax.xaxis.grid(True, color=AXIS.hairline, linewidth=0.6, alpha=alpha)
    ax.set_axisbelow(True)


# ═══════════════════════════════════════════════════════════════════
# 4. Semantic shading (blue-top / red-bottom / white-middle)
# ═══════════════════════════════════════════════════════════════════


def semantic_y_gradient(
    ax: plt.Axes,
    top_color: str = AXIS.blue_soft,
    bot_color: str = AXIS.red_soft,
    mid_color: str | None = None,
    alpha: float = 0.9,
    resolution: int = 256,
    zero_at: float | None = 0.0,
) -> None:
    """Paint the plot background as a vertical blue→red gradient.

    ``zero_at`` pins the neutral colour at that y-value (defaults to 0).
    Pass ``None`` to get a linear blend across the whole y range.
    """
    if mid_color is None:
        mid_color = AXIS.bg_card

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    if zero_at is None:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "_axis_vgrad", [bot_color, mid_color, top_color], N=resolution
        )
        grad = np.linspace(0.0, 1.0, resolution).reshape(-1, 1)
    else:
        y_lo, y_hi = ylim
        # Build a two-segment palette pinned at `zero_at`.
        span = y_hi - y_lo
        if span <= 0:
            return
        zero_frac = float(np.clip((zero_at - y_lo) / span, 0.0, 1.0))
        # Construct cmap with mid_color at zero_frac.
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "_axis_vgrad",
            [(0.0, bot_color), (zero_frac, mid_color), (1.0, top_color)],
            N=resolution,
        )
        grad = np.linspace(0.0, 1.0, resolution).reshape(-1, 1)

    ax.imshow(
        grad,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        alpha=alpha,
        zorder=0,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def semantic_y_poles(
    ax: plt.Axes,
    top_text: str = "Assistant-like",
    top_color: str = AXIS.blue,
    bot_text: str = "Role-playing",
    bot_color: str = AXIS.red,
    fontsize: float = 8.0,
    arrow_length: float = 0.22,
    offset: float = -0.085,
) -> None:
    """Draw the signature colored arrow labels flanking the y-axis."""
    ax.annotate(
        top_text,
        xy=(offset, 1.0),
        xytext=(offset, 1.0 - arrow_length),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="right",
        va="center",
        rotation=90,
        fontsize=fontsize,
        fontweight="bold",
        color=top_color,
        arrowprops=dict(arrowstyle="<-", color=top_color, lw=1.2),
    )
    ax.annotate(
        bot_text,
        xy=(offset, 0.0),
        xytext=(offset, arrow_length),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="right",
        va="center",
        rotation=90,
        fontsize=fontsize,
        fontweight="bold",
        color=bot_color,
        arrowprops=dict(arrowstyle="<-", color=bot_color, lw=1.2),
    )


def semantic_x_poles(
    ax: plt.Axes,
    left_text: str = "Role-playing",
    left_color: str = AXIS.red,
    right_text: str = "Assistant-like",
    right_color: str = AXIS.blue,
    fontsize: float = 8.5,
    arrow_length: float = 0.18,
    y: float = -0.16,
) -> None:
    ax.annotate(
        left_text,
        xy=(0.0, y),
        xytext=(arrow_length, y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="right",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=left_color,
        arrowprops=dict(arrowstyle="<-", color=left_color, lw=1.2),
    )
    ax.annotate(
        right_text,
        xy=(1.0, y),
        xytext=(1.0 - arrow_length, y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=right_color,
        arrowprops=dict(arrowstyle="<-", color=right_color, lw=1.2),
    )


# ═══════════════════════════════════════════════════════════════════
# 5. Atomic plotting primitives
# ═══════════════════════════════════════════════════════════════════


def line_with_markers(
    ax: plt.Axes,
    x: Iterable[float],
    y: Iterable[float],
    color: str,
    marker: str = "o",
    label: str | None = None,
    linewidth: float = 2.1,
    markersize: float = 6.5,
    zorder: int = 3,
    alpha: float = 1.0,
) -> None:
    ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder - 1,
        marker=None,
    )
    ax.plot(
        x,
        y,
        marker=marker,
        linestyle="None",
        color=color,
        markerfacecolor=color,
        markeredgecolor=AXIS.bg_card,
        markeredgewidth=0.9,
        markersize=markersize,
        zorder=zorder,
        label=label,
    )


def smooth_fill(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    fill_color: str | None = None,
    linewidth: float = 1.6,
    alpha_fill: float = 0.18,
    zorder: int = 2,
) -> None:
    """PCHIP-interpolated smooth line with shaded area beneath."""
    try:
        from scipy.interpolate import PchipInterpolator

        xs = np.linspace(float(np.min(x)), float(np.max(x)), 300)
        ys = PchipInterpolator(x, y)(xs)
    except Exception:
        xs, ys = np.asarray(x), np.asarray(y)

    base_y = 0.0
    fill = fill_color or color
    ax.fill_between(
        xs,
        base_y,
        ys,
        color=fill,
        alpha=alpha_fill,
        linewidth=0,
        zorder=zorder - 1,
    )
    ax.plot(xs, ys, color=color, linewidth=linewidth, zorder=zorder)


def annotate_endpoint(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    dx: float = 8,
    dy: float = 0,
    fontsize: float = 8.4,
    fontweight: str = "bold",
) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        va="center",
        ha="left" if dx >= 0 else "right",
    )


def rounded_bar(
    ax: plt.Axes,
    x: float,
    height: float,
    width: float = 0.6,
    color: str = AXIS.blue,
    alpha: float = 0.95,
    rounding: float = 0.06,
    edgecolor: str = AXIS.bg_card,
    edgewidth: float = 0.8,
    label: str | None = None,
    zorder: int = 2,
) -> FancyBboxPatch:
    y0 = min(0.0, height)
    h = abs(height)
    x0 = x - width / 2.0
    r = min(rounding, width * 0.4, max(h * 0.4, 1e-6))
    patch = FancyBboxPatch(
        (x0, y0),
        width,
        h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=edgewidth,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def baseline_marker(
    ax: plt.Axes,
    y: float,
    text: str = "Baseline",
    color: str = AXIS.rule,
    fontsize: float = 7.5,
    xpos_frac: float = 0.62,
) -> None:
    ax.axhline(
        y,
        color=color,
        linestyle=(0, (3, 3)),
        linewidth=0.9,
        alpha=0.85,
        zorder=1,
    )
    xlim = ax.get_xlim()
    x = xlim[0] + xpos_frac * (xlim[1] - xlim[0])
    ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        color=AXIS.muted,
        style="italic",
        va="bottom",
        ha="left",
        bbox=dict(
            facecolor=AXIS.bg_card,
            edgecolor="none",
            pad=1.2,
            alpha=0.85,
        ),
    )


# ═══════════════════════════════════════════════════════════════════
# 6. Save
# ═══════════════════════════════════════════════════════════════════


def save_fig(
    fig: plt.Figure,
    stem: str,
    output_dir: Union[str, Path],
    formats: Sequence[str] = ("png", "pdf", "svg"),
    max_raster_px: int = 3000,
) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for fmt in formats:
        out = output_dir / f"{stem}.{fmt}"
        if fmt == "png":
            w, h = fig.get_size_inches()
            dpi = 450
            if max(w, h) * dpi > max_raster_px:
                dpi = int(max_raster_px / max(w, h))
            fig.savefig(out, dpi=dpi, facecolor=AXIS.bg_canvas)
        else:
            fig.savefig(out, facecolor=AXIS.bg_canvas)
        paths.append(out)
    plt.close(fig)
    return paths
