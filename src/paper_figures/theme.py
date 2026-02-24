"""Unified publication theme for endorsement / prior-consistency paper.

Design language: editorial data visualization inspired by Anthropic's
"Assistant Axis" paper — warm backgrounds, semantic gradients, minimal
chrome, narrative-first layout.

Usage::

    from src.paper_figures.theme import apply_theme, PAL, save_fig
    apply_theme()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ...
    save_fig(fig, "my_figure", output_dir)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ═══════════════════════════════════════════════════════════════════════
# 1. COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════


class _Palette:
    """Central colour definitions.

    Editorial palette with warm tones.  Every primary colour passes WCAG
    AA contrast on the warm background and is distinguishable under the
    three main CVD types.
    """

    # ── Warm background (the signature look) ──────────────────────────
    # bg_warm = "#F7F2ED"  # warm cream
    bg_warm = "#FFFFFF"
    bg_card = "#FFFFFF"  # white cards on warm bg
    bg_white = "#FFFFFF"

    # ── Primary semantic pair (blue ↔ red gradient) ───────────────────
    blue = "#4A7FB5"  # "correct" / "assistant-like" end
    red = "#C0625D"  # "wrong" / "role-playing" end
    blue_light = "#A8C8E8"
    red_light = "#E8B4B1"

    # ── Model colours ─────────────────────────────────────────────────
    llama = "#4A7FB5"  # steel blue
    qwen = "#C0625D"  # muted red

    # ── Tag colours (authority hierarchy) ─────────────────────────────
    expert = "#4A7FB5"
    note = "#C0625D"
    user = "#6B9E78"
    online = "#C8A951"

    # ── Instruction conditions ────────────────────────────────────────
    no_instr = "#4A7FB5"
    with_instr = "#4A7FB5"  # same hue, lighter alpha

    # ── Suppression pair ──────────────────────────────────────────────
    r_w = "#D4805F"  # terracotta (wrong endorsement suppression)
    r_c = "#6B9E78"  # sage green (correct endorsement suppression)

    # ── Grays ─────────────────────────────────────────────────────────
    dark_text = "#2D2D2D"
    medium_gray = "#777777"
    light_gray = "#AAAAAA"
    faint_gray = "#D5D0CB"  # warm-tinted gray
    white = "#FFFFFF"
    black = "#000000"

    # ── Categorical (up to 6) ─────────────────────────────────────────
    categorical = [
        "#4A7FB5",  # blue
        "#C0625D",  # red
        "#6B9E78",  # sage green
        "#D4805F",  # terracotta
        "#C8A951",  # gold
        "#7B6D8E",  # dusty purple
    ]

    # ── Semantic shading ──────────────────────────────────────────────
    positive_region = "#E2EFE5"  # soft sage tint
    negative_region = "#F2E0DE"  # soft rose tint

    # ── Model styling dict ────────────────────────────────────────────
    MODEL_STYLES: Dict[str, Dict] = {
        "instruct": {
            "color": "#4A7FB5",
            "marker": "o",
            "label": "Llama-3.1-8B-Instruct",
            "short": "Instruct",
        },
        "base": {
            "color": "#C8A951",
            "marker": "D",
            "label": "Llama-3.1-8B",
            "short": "Base",
        },
        "llama": {
            "color": "#4A7FB5",
            "marker": "o",
            "label": "Llama-3.1-8B-Instruct",
            "short": "Llama",
        },
        "qwen": {
            "color": "#C0625D",
            "marker": "s",
            "label": "Qwen3-4B-Instruct",
            "short": "Qwen",
        },
        "qwen_thinking": {
            "color": "#6B9E78",
            "marker": "^",
            "label": "Qwen3-4B-Thinking",
            "short": "Qwen-T",
        },
    }

    TAG_ORDER = ["Expert", "Note", "User", "Someone online"]
    TAG_COLORS = {
        "Expert": "#4A7FB5",
        "Note": "#C0625D",
        "User": "#6B9E78",
        "Someone online": "#C8A951",
    }
    TAG_SHORT = {
        "Expert": "Expert",
        "Note": "Note",
        "User": "User",
        "Someone online": "Online",
    }


PAL = _Palette()


# ═══════════════════════════════════════════════════════════════════════
# 2. GRADIENTS
# ═══════════════════════════════════════════════════════════════════════


def make_diverging_cmap(
    low: str = PAL.blue,
    mid: str = "#F0EBE3",
    high: str = PAL.red,
    name: str = "paper_diverging",
) -> mcolors.LinearSegmentedColormap:
    """Blue → warm cream → red diverging colourmap.

    The midpoint uses the warm background colour so that items near
    zero blend into the canvas — only the tails draw attention.
    """
    return mcolors.LinearSegmentedColormap.from_list(
        name,
        [low, mid, high],
        N=256,
    )


def make_sequential_cmap(
    low: str = "#E8E2DB",
    high: str = PAL.blue,
    name: str = "paper_sequential",
) -> mcolors.LinearSegmentedColormap:
    """Faint warm → blue sequential colourmap."""
    return mcolors.LinearSegmentedColormap.from_list(
        name,
        [low, high],
        N=256,
    )


# Pre-built cmaps
CMAP_DIVERGING = make_diverging_cmap()
CMAP_SEQUENTIAL = make_sequential_cmap()

# Canonical per-model result directories.
MODEL_RESULT_DIRS = {
    "llama": ("llama-3.1-8b-results",),
    "qwen": ("qwen3-4b-results",),
    "qwen_thinking": ("qwen3-4b-thinking-results",),
}


def model_result_roots(root: Path, model: str) -> List[Path]:
    """Candidate roots for a model's results directory.

    Ordered by preferred canonical names.
    """
    names = MODEL_RESULT_DIRS.get(model, (model,))
    return [root / name for name in names]


def model_result_path(root: Path, model: str, relative: Union[str, Path]) -> Path:
    """Resolve a model-specific file path across canonical + legacy layouts."""
    rel = Path(relative)
    candidates = [base / rel for base in model_result_roots(root, model)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


# ═══════════════════════════════════════════════════════════════════════
# 3. FIGURE SIZING
# ═══════════════════════════════════════════════════════════════════════

FIGSIZE_SINGLE = (5.5, 3.5)
FIGSIZE_WIDE = (7.0, 3.2)
FIGSIZE_TALL = (5.5, 5.0)
FIGSIZE_2x1 = (7.0, 3.5)
FIGSIZE_2x2 = (7.0, 6.0)
FIGSIZE_3PANEL = (7.5, 3.4)
FIGSIZE_HERO = (7.0, 4.0)

SAVE_DPI = 600
DISPLAY_DPI = 150


# ═══════════════════════════════════════════════════════════════════════
# 4. RCPARAMS
# ═══════════════════════════════════════════════════════════════════════

_RC_PARAMS = {
    "figure.dpi": DISPLAY_DPI,
    "figure.facecolor": PAL.bg_warm,
    "figure.edgecolor": "none",
    "figure.constrained_layout.use": True,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
    ],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.linewidth": 0.6,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlepad": 8,
    "axes.labelpad": 4,
    "axes.facecolor": PAL.bg_warm,
    "axes.edgecolor": PAL.faint_gray,
    "axes.labelcolor": PAL.dark_text,
    "axes.prop_cycle": mpl.cycler(color=PAL.categorical),
    "axes.grid": False,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.5,
    "grid.color": PAL.faint_gray,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": PAL.faint_gray,
    "ytick.color": PAL.faint_gray,
    "xtick.labelcolor": PAL.medium_gray,
    "ytick.labelcolor": PAL.medium_gray,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.35,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "patch.edgecolor": "none",
    "patch.linewidth": 0,
    "text.color": PAL.dark_text,
    "savefig.dpi": SAVE_DPI,
    "savefig.facecolor": PAL.bg_warm,
    "savefig.edgecolor": "none",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
}


def apply_theme() -> None:
    """Apply the editorial publication theme globally."""
    plt.rcParams.update(_RC_PARAMS)


# ═══════════════════════════════════════════════════════════════════════
# 5. HELPERS
# ═══════════════════════════════════════════════════════════════════════


def save_fig(
    fig: plt.Figure,
    stem: str,
    output_dir: Union[str, Path],
    formats: Sequence[str] = ("png", "pdf"),
    max_raster_px: int = 2400,
) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for fmt in formats:
        out = output_dir / f"{stem}.{fmt}"
        if fmt == "png":
            w, h = fig.get_size_inches()
            dpi = SAVE_DPI
            if max(w, h) * dpi > max_raster_px:
                dpi = int(max_raster_px / max(w, h))
            fig.savefig(out, dpi=dpi, facecolor=PAL.bg_warm, bbox_inches="tight")
        else:
            fig.savefig(out, facecolor=PAL.bg_warm, bbox_inches="tight")
        paths.append(out)
    plt.close(fig)
    return paths


def label_panel(
    ax: plt.Axes,
    letter: str,
    x: float = -0.06,
    y: float = 1.06,
    fontsize: int = 12,
) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="right",
        color=PAL.dark_text,
    )


def add_zero_line(ax: plt.Axes, orientation: str = "h", **kw) -> None:
    defaults = dict(
        color=PAL.faint_gray, linewidth=0.7, linestyle="-", alpha=0.8, zorder=0
    )
    defaults.update(kw)
    if orientation == "h":
        ax.axhline(0, **defaults)
    else:
        ax.axvline(0, **defaults)


def add_region_shading(
    ax: plt.Axes, y_top=None, y_bot=None, alpha: float = 0.5
) -> None:
    ylim = ax.get_ylim()
    top = y_top if y_top is not None else ylim[1]
    bot = y_bot if y_bot is not None else ylim[0]
    ax.axhspan(0, top, color=PAL.positive_region, alpha=alpha, zorder=0, linewidth=0)
    ax.axhspan(bot, 0, color=PAL.negative_region, alpha=alpha, zorder=0, linewidth=0)


def add_y_grid(ax: plt.Axes, alpha: float = 0.20) -> None:
    ax.yaxis.grid(True, alpha=alpha, linewidth=0.5, color=PAL.faint_gray)
    ax.set_axisbelow(True)


def add_semantic_yaxis(
    ax: plt.Axes,
    top_text: str,
    top_color: str,
    bot_text: str,
    bot_color: str,
    fontsize: float = 7,
) -> None:
    """Add interpretive labels at the top and bottom of the y-axis."""
    ax.text(
        -0.02,
        1.0,
        top_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
        color=top_color,
        fontstyle="italic",
    )
    ax.text(
        -0.02,
        0.0,
        bot_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=fontsize,
        color=bot_color,
        fontstyle="italic",
    )


def plot_with_band(
    ax: plt.Axes,
    x,
    y,
    ci_lo,
    ci_hi,
    color: str,
    label=None,
    marker="o",
    band_alpha: float = 0.15,
    **kw,
) -> None:
    ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=band_alpha, linewidth=0)
    defaults = dict(
        color=color,
        marker=marker,
        markersize=5,
        markeredgecolor=PAL.bg_warm,
        markeredgewidth=0.8,
        linewidth=1.8,
        zorder=3,
        label=label,
    )
    defaults.update(kw)
    ax.plot(x, y, **defaults)


def annotate_endpoint(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    offset=(8, 0),
    fontsize=8,
    **kw,
) -> None:
    defaults = dict(
        fontsize=fontsize, fontweight="bold", color=color, va="center", ha="left"
    )
    defaults.update(kw)
    ax.annotate(text, xy=(x, y), xytext=offset, textcoords="offset points", **defaults)


def draw_bar(
    ax: plt.Axes,
    x: float,
    height: float,
    width: float = 0.6,
    color: str = PAL.blue,
    alpha: float = 0.9,
    rounding: float = 0.08,
    hatch=None,
    label=None,
) -> FancyBboxPatch:
    y0 = min(0.0, height)
    h = abs(height)
    x0 = x - width / 2
    r = min(rounding, width * 0.4, h * 0.4 if h > 0 else rounding)
    patch = FancyBboxPatch(
        (x0, y0),
        width,
        h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=color,
        edgecolor=PAL.bg_warm,
        linewidth=0.8,
        alpha=alpha,
        hatch=hatch,
        label=label,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def draw_error_bar(
    ax: plt.Axes,
    x: float,
    mean: float,
    ci_lo: float,
    ci_hi: float,
    color: str = PAL.dark_text,
    cap_width: float = 0.08,
    linewidth: float = 0.9,
    alpha: float = 0.5,
) -> None:
    ax.plot([x, x], [ci_lo, ci_hi], color=color, lw=linewidth, alpha=alpha, zorder=4)
    for yv in [ci_lo, ci_hi]:
        ax.plot(
            [x - cap_width / 2, x + cap_width / 2],
            [yv, yv],
            color=color,
            lw=linewidth,
            alpha=alpha,
            zorder=4,
        )


def ci95(mean: float, std: float, n: int) -> Tuple[float, float]:
    if n <= 0:
        return (mean, mean)
    se = std / math.sqrt(n)
    return (mean - 1.96 * se, mean + 1.96 * se)


def semantic_arrow(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    direction: str = "right",
    fontsize: float = 8,
) -> None:
    """Draw a semantic axis arrow with label (like "← Role-playing")."""
    arrow = "<-" if direction == "left" else "->"
    ax.annotate(
        text,
        xy=(x, y),
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        ha="left" if direction == "right" else "right",
        va="center",
        arrowprops=dict(arrowstyle=arrow, color=color, lw=1.2),
        xytext=(15 if direction == "right" else -15, 0),
        textcoords="offset points",
    )
