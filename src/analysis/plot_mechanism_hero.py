#!/usr/bin/env python3
"""Anthropic-style hero figure for the correction-gating mechanism.

This figure is meant to read like the Assistant Axis opening figure:
one editorial composition with two coordinated panes rather than a
dashboard of small panels.

- Left card: geometry of the correction-gating direction
- Right card: one integrated causal / evidence-specificity readout

Run:
  uv run python -m src.analysis.plot_mechanism_hero
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
from scipy.stats import gaussian_kde


PALETTE = {
    "bg": "#F5EFE8",
    "card": "#FBF8F3",
    "card_edge": "#E2D9CD",
    "text": "#2F2924",
    "muted": "#7E766E",
    "line": "#8F877E",
    "axis": "#4A443D",
    "rose": "#D56E67",
    "rose_fill": "#EFC4BF",
    "blue": "#5D85B3",
    "blue_fill": "#C9D8EB",
    "sage": "#7EA690",
    "sage_fill": "#D9E7DE",
    "gold": "#D2B16E",
    "sand": "#EEE6DC",
    "left_tint": "#F8E9E6",
    "right_tint": "#EAF0F7",
    "pill": "#F1E8DE",
    "white": "#FFFFFF",
}


plt.rcParams.update(
    {
        "figure.dpi": 240,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica Neue",
            "Helvetica",
            "Arial",
            "DejaVu Sans",
            "Liberation Sans",
        ],
        "font.size": 10,
        "axes.facecolor": "none",
        "axes.edgecolor": "none",
        "savefig.bbox": "tight",
    }
)


AXIS_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "correction_axis",
    [PALETTE["rose"], "#B49AAF", PALETTE["blue"]],
)


def _add_card(fig: plt.Figure, rect: tuple[float, float, float, float]) -> FancyBboxPatch:
    x, y, w, h = rect
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.022",
        transform=fig.transFigure,
        facecolor=PALETTE["card"],
        edgecolor=PALETTE["card_edge"],
        linewidth=0.9,
        zorder=-5,
    )
    fig.patches.append(card)
    return card


def _add_accent(fig: plt.Figure, rect: tuple[float, float, float, float], color: str) -> None:
    x, y, w, h = rect
    accent = FancyBboxPatch(
        (x + 0.012, y + h - 0.020),
        0.09 * w,
        0.010,
        boxstyle="round,pad=0.001,rounding_size=0.004",
        transform=fig.transFigure,
        facecolor=color,
        edgecolor="none",
        zorder=-4,
    )
    fig.patches.append(accent)


def _add_icon(fig: plt.Figure, center: tuple[float, float], color: str, kind: str) -> None:
    cx, cy = center
    circle = Circle(
        (cx, cy),
        0.010,
        transform=fig.transFigure,
        facecolor=PALETTE["card"],
        edgecolor=PALETTE["line"],
        linewidth=0.8,
        zorder=5,
    )
    fig.patches.append(circle)

    overlay = fig.add_axes([0, 0, 1, 1], frameon=False)
    overlay.set_axis_off()
    overlay.set_zorder(6)

    if kind == "axis":
        overlay.plot(
            [cx - 0.0055, cx + 0.0055],
            [cy, cy],
            transform=fig.transFigure,
            color=color,
            lw=1.0,
            solid_capstyle="round",
        )
        overlay.plot(
            [cx, cx],
            [cy - 0.0045, cy + 0.0045],
            transform=fig.transFigure,
            color=color,
            lw=1.0,
            solid_capstyle="round",
        )
    elif kind == "patch":
        overlay.plot(
            [cx - 0.005, cx + 0.002],
            [cy - 0.002, cy + 0.005],
            transform=fig.transFigure,
            color=color,
            lw=1.1,
            solid_capstyle="round",
        )
        overlay.plot(
            [cx - 0.001, cx + 0.005],
            [cy - 0.005, cy + 0.001],
            transform=fig.transFigure,
            color=color,
            lw=1.1,
            solid_capstyle="round",
        )
    elif kind == "evidence":
        overlay.plot(
            [cx - 0.0045, cx + 0.0045],
            [cy - 0.003, cy - 0.003],
            transform=fig.transFigure,
            color=color,
            lw=1.0,
            solid_capstyle="round",
        )
        overlay.plot(
            [cx - 0.003, cx - 0.003],
            [cy - 0.003, cy + 0.004],
            transform=fig.transFigure,
            color=color,
            lw=1.0,
            solid_capstyle="round",
        )
        overlay.plot(
            [cx + 0.003, cx + 0.003],
            [cy - 0.003, cy + 0.004],
            transform=fig.transFigure,
            color=color,
            lw=1.0,
            solid_capstyle="round",
        )


def _add_pill(fig: plt.Figure, x: float, y: float, text: str) -> None:
    fig.text(
        x,
        y,
        text,
        fontsize=8.6,
        color=PALETTE["muted"],
        ha="left",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.25,rounding_size=0.5",
            facecolor=PALETTE["pill"],
            edgecolor="none",
        ),
    )


def _style_plot_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, colors=PALETTE["muted"])


def _draw_geometry_panel(
    ax: plt.Axes,
    entrenching: np.ndarray,
    correcting: np.ndarray,
    i1c_all: np.ndarray,
) -> None:
    all_i1a = np.concatenate([entrenching, correcting])
    i1a_mean = float(np.mean(all_i1a))
    i1c_mean = float(np.mean(i1c_all))
    i1d_mean = 2.4

    xlim = (-18.5, 12.5)
    x_grid = np.linspace(*xlim, 500)

    ax.set_xlim(xlim)
    ax.set_ylim(-0.34, 1.30)
    ax.axvspan(xlim[0], -2.8, color=PALETTE["left_tint"], alpha=0.85, zorder=0)
    ax.axvspan(-2.8, xlim[1], color=PALETTE["right_tint"], alpha=0.85, zorder=0)

    kde_ent = gaussian_kde(entrenching, bw_method=0.33)
    kde_cor = gaussian_kde(correcting, bw_method=0.30)
    kde_i1c = gaussian_kde(i1c_all, bw_method=0.32)

    y_ent = kde_ent(x_grid)
    y_cor = kde_cor(x_grid)
    y_i1c = kde_i1c(x_grid)
    scale = 0.95 / max(y_ent.max(), y_cor.max(), y_i1c.max())
    y_ent *= scale
    y_cor *= scale
    y_i1c *= scale

    ax.fill_between(x_grid, 0, y_ent, color=PALETTE["rose_fill"], alpha=0.90, zorder=1)
    ax.plot(x_grid, y_ent, color=PALETTE["rose"], lw=1.5, alpha=0.95, zorder=2)

    ax.fill_between(x_grid, 0, y_cor, color=PALETTE["blue_fill"], alpha=0.88, zorder=1)
    ax.plot(x_grid, y_cor, color=PALETTE["blue"], lw=1.5, alpha=0.95, zorder=2)

    ax.plot(
        x_grid,
        y_i1c,
        color=PALETTE["sage"],
        lw=1.4,
        alpha=0.9,
        ls=(0, (4, 2)),
        zorder=2,
    )
    ax.fill_between(x_grid, 0, y_i1c, color=PALETTE["sage_fill"], alpha=0.42, zorder=1)

    rng = np.random.default_rng(7)
    dot_y = -0.012 + rng.normal(scale=0.003, size=all_i1a.shape[0])
    norm = mcolors.Normalize(vmin=xlim[0], vmax=xlim[1])
    ax.scatter(
        all_i1a,
        dot_y,
        c=AXIS_CMAP(norm(all_i1a)),
        s=22,
        alpha=0.50,
        edgecolors="none",
        zorder=3,
    )

    ax.axhline(0, color=PALETTE["axis"], lw=1.2, zorder=4)
    for tick in np.arange(-15, 11, 5):
        ax.plot([tick, tick], [-0.018, 0.018], color=PALETTE["axis"], lw=1.0, zorder=4)
        ax.text(
            tick,
            -0.055,
            f"{tick:d}",
            color=PALETTE["muted"],
            fontsize=8,
            ha="center",
            va="top",
        )

    ax.text(
        xlim[0] + 0.4,
        1.08,
        "Same instruction, different internal state",
        color=PALETTE["text"],
        fontsize=11.5,
        fontweight="bold",
        ha="left",
    )
    ax.text(
        xlim[0] + 0.4,
        1.00,
        "Within i1a, entrenching items sit leftward; correcting items sit closer to\n"
        "the evidence-integrating end of the direction. i1c shifts the full distribution right.",
        color=PALETTE["muted"],
        fontsize=9.0,
        ha="left",
        va="top",
        linespacing=1.35,
    )

    marker_specs = [
        ("i1a", i1a_mean, PALETTE["rose"], 0.02, -0.45, "right", "Answer correctly\neven if the speaker is wrong."),
        ("i1c", i1c_mean, PALETTE["sage"], 0.10, 0.15, "left", "Base your answer on the\nfacts, not the speaker's opinion."),
        ("i1d", i1d_mean, PALETTE["gold"], 0.18, 0.35, "left", "Answer correctly by\nevaluating the facts."),
    ]
    for label, xpos, color, y_offset, x_offset, ha, desc in marker_specs:
        ax.plot(
            xpos,
            0,
            marker="D",
            markersize=10,
            markerfacecolor=color,
            markeredgecolor=PALETTE["white"],
            markeredgewidth=1.6,
            zorder=6,
        )
        ax.plot(
            [xpos, xpos],
            [0.03, 0.88 + y_offset],
            color=color,
            alpha=0.35,
            lw=0.9,
            ls=(0, (1, 2)),
            zorder=5,
        )
        ax.text(
            xpos + x_offset,
            0.92 + y_offset,
            label,
            color=color,
            fontsize=12.5,
            fontweight="bold",
            ha=ha,
            va="bottom",
        )
        ax.text(
            xpos + x_offset,
            0.865 + y_offset,
            desc,
            color=color,
            fontsize=7.0,
            ha=ha,
            va="top",
            linespacing=1.25,
        )

    ax.annotate(
        "Think-then-ignore\n"
        "\"Let me confirm with reliable sources in my mind...\"",
        xy=(float(np.mean(entrenching)), 0.12),
        xytext=(-16.6, 0.72),
        textcoords="data",
        color=PALETTE["text"],
        fontsize=8.4,
        ha="left",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.35,rounding_size=0.45",
            facecolor=PALETTE["white"],
            edgecolor=PALETTE["card_edge"],
            linewidth=0.8,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color=PALETTE["line"],
            lw=1.0,
            connectionstyle="angle,angleA=0,angleB=90,rad=8",
        ),
        zorder=7,
    )
    ax.annotate(
        "Evidence gets through\n"
        "\"Gerard... yeah, that sounds right.\"",
        xy=(float(np.mean(correcting)), 0.12),
        xytext=(5.3, 0.30),
        textcoords="data",
        color=PALETTE["text"],
        fontsize=8.4,
        ha="left",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.35,rounding_size=0.45",
            facecolor=PALETTE["white"],
            edgecolor=PALETTE["card_edge"],
            linewidth=0.8,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color=PALETTE["line"],
            lw=1.0,
            connectionstyle="angle,angleA=180,angleB=90,rad=10",
        ),
        zorder=7,
    )

    ax.annotate(
        "",
        xy=(-18.0, -0.22),
        xytext=(-8.8, -0.22),
        arrowprops=dict(arrowstyle="->", color=PALETTE["rose"], lw=1.8),
    )
    ax.text(
        -17.8,
        -0.26,
        "Prior-preserving",
        color=PALETTE["rose"],
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        -17.8,
        -0.33,
        "Defends the model's prior,\neven when the note is corrective",
        color=PALETTE["muted"],
        fontsize=8.2,
        ha="left",
        va="top",
        linespacing=1.25,
    )

    ax.annotate(
        "",
        xy=(12.0, -0.22),
        xytext=(3.0, -0.22),
        arrowprops=dict(arrowstyle="->", color=PALETTE["blue"], lw=1.8),
    )
    ax.text(
        11.8,
        -0.26,
        "Evidence-integrating",
        color=PALETTE["blue"],
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="top",
    )
    ax.text(
        11.8,
        -0.33,
        "Lets endorsement evidence\nchange the answer",
        color=PALETTE["muted"],
        fontsize=8.2,
        ha="right",
        va="top",
        linespacing=1.25,
    )

    ax.text(
        -3.0,
        -0.10,
        "Projection onto correction-gating direction",
        color=PALETTE["muted"],
        fontsize=8.6,
        ha="center",
        va="top",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    _style_plot_axis(ax)


def _draw_effect_panel(ax: plt.Axes) -> None:
    rows = [
        ("Correct endorsement (C1)", 1.93, PALETTE["blue"]),
        ("Wrong endorsement (W1)", 0.30, PALETTE["gold"]),
        ("No endorsement (N0)", -0.07, "#B9B0A7"),
        ("Random vector", -0.50, "#BDB3AA"),
        ("Wrong position", -0.28, "#C8BEB4"),
        ("Opposite direction", -1.33, PALETTE["rose"]),
    ]

    y = np.array([4.95, 4.05, 3.15, 1.75, 0.95, 0.15])
    ax.axvspan(0, 2.4, color=PALETTE["right_tint"], alpha=0.55, zorder=0)
    ax.axvspan(-1.8, 0, color=PALETTE["left_tint"], alpha=0.55, zorder=0)
    ax.axvline(0, color=PALETTE["axis"], lw=1.0, zorder=1)

    ax.text(
        -1.70,
        6.02,
        "Patching is causal, but only when evidence is present",
        fontsize=11.3,
        fontweight="bold",
        color=PALETTE["text"],
        ha="left",
    )
    ax.text(
        -1.70,
        5.66,
        "Correct endorsement gives a large positive shift; controls are null or harmful.",
        fontsize=8.8,
        color=PALETTE["muted"],
        ha="left",
        va="top",
        linespacing=1.30,
    )

    ax.text(
        -1.70,
        5.26,
        "Evidence test",
        fontsize=8.3,
        color=PALETTE["muted"],
        ha="left",
        va="bottom",
        fontweight="bold",
    )
    ax.text(
        -1.70,
        2.00,
        "Causal controls",
        fontsize=8.3,
        color=PALETTE["muted"],
        ha="left",
        va="bottom",
        fontweight="bold",
    )
    ax.plot([-1.70, 2.25], [2.25, 2.25], color=PALETTE["card_edge"], lw=0.9, zorder=1)

    for yi, (label, val, color) in zip(y, rows):
        ax.plot([0, val], [yi, yi], color=color, lw=7.2, solid_capstyle="round", zorder=2)
        ax.scatter([val], [yi], s=62, color=color, edgecolor=PALETTE["white"], linewidth=1.2, zorder=3)
        ax.text(-1.70, yi, label, fontsize=8.9, color=PALETTE["text"], ha="left", va="center")
        ax.text(
            val + (0.08 if val >= 0 else -0.08),
            yi,
            f"{val:+.2f}",
            fontsize=9.0,
            fontweight="bold",
            color=color,
            ha="left" if val >= 0 else "right",
            va="center",
        )

    ax.plot([0.10, 0.10, 0.10, 1.88], [3.15, 3.70, 4.58, 4.58], color=PALETTE["line"], lw=1.0)
    ax.text(
        1.18,
        4.72,
        "30x stronger with\ncorrect endorsement",
        fontsize=9.1,
        fontweight="bold",
        color=PALETTE["blue"],
        ha="center",
        va="bottom",
    )

    ax.text(
        1.18,
        1.72,
        "87% shifted positive\n13/151 flips\n0 harmed",
        fontsize=9.0,
        color=PALETTE["blue"],
        fontweight="bold",
        ha="center",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.35,rounding_size=0.40",
            facecolor=PALETTE["white"],
            edgecolor=PALETTE["blue_fill"],
            linewidth=0.8,
        ),
    )

    ax.text(
        1.92,
        0.58,
        "opposite direction\nmakes things worse",
        fontsize=8.2,
        color=PALETTE["rose"],
        ha="right",
        va="center",
    )

    ax.set_xlim(-1.8, 2.4)
    ax.set_ylim(-0.35, 6.15)
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_xticklabels([f"{tick:d}" for tick in [-1, 0, 1, 2]], fontsize=8, color=PALETTE["muted"])
    ax.set_yticks([])
    ax.text(
        0.32,
        -0.28,
        "Logit margin shift toward correct",
        fontsize=8.5,
        color=PALETTE["muted"],
        ha="center",
        va="top",
    )
    _style_plot_axis(ax)


def main() -> None:
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    data = np.load("figures/mechanism_projections.npz")
    entrenching = data["i1a_entrenching"]
    correcting = data["i1a_correcting"]
    i1c_all = data["i1c_all"]

    fig = plt.figure(figsize=(13.6, 7.7), facecolor=PALETTE["bg"])

    main_rect = (0.035, 0.11, 0.59, 0.66)
    effect_rect = (0.655, 0.11, 0.31, 0.66)

    _add_card(fig, main_rect)
    _add_card(fig, effect_rect)
    _add_accent(fig, main_rect, PALETTE["blue"])
    _add_accent(fig, effect_rect, PALETTE["rose"])
    _add_icon(fig, (main_rect[0] + 0.030, main_rect[1] + main_rect[3] - 0.028), PALETTE["blue"], "axis")
    _add_icon(fig, (effect_rect[0] + 0.028, effect_rect[1] + effect_rect[3] - 0.028), PALETTE["rose"], "patch")

    fig.text(
        0.035,
        0.93,
        "A Direction That Gates Evidence Integration",
        fontsize=34,
        fontweight="bold",
        color=PALETTE["text"],
        ha="left",
        va="top",
    )
    fig.text(
        0.035,
        0.875,
        "At the endorsement position, a single activation direction predicts whether i1a will\n"
        "defend the model's prior or let corrective evidence change the answer.",
        fontsize=12.5,
        color=PALETTE["muted"],
        ha="left",
        va="top",
        linespacing=1.35,
    )
    _add_pill(fig, 0.036, 0.815, "Qwen3-4B-Instruct")
    _add_pill(fig, 0.153, 0.815, "endorsement position")
    _add_pill(fig, 0.303, 0.815, "layer 23")
    _add_pill(fig, 0.382, 0.815, "probe accuracy 81%")
    _add_pill(fig, 0.507, 0.815, "effect size d = 0.91")

    main_ax = fig.add_axes([0.055, 0.155, 0.55, 0.56], facecolor="none")
    effect_ax = fig.add_axes([0.675, 0.155, 0.27, 0.56], facecolor="none")

    _draw_geometry_panel(main_ax, entrenching, correcting, i1c_all)
    _draw_effect_panel(effect_ax)

    for ext in ("png", "pdf"):
        out = output_dir / f"mechanism_hero.{ext}"
        fig.savefig(
            out,
            dpi=300,
            facecolor=PALETTE["bg"],
            bbox_inches="tight",
            pad_inches=0.10,
        )
        print(f"Saved: {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
