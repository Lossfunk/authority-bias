#!/usr/bin/env python3
"""Figure 3 hero split into two separate graphs.

Outputs:
  - fig3a_prior_consistency_geometry
  - fig3b_prior_consistency_decomposition

Design goals:
  - Keep the same warm editorial style and model color mapping.
  - Remove congestion from the old combined two-panel layout.
  - Keep the real item callout (trivia_qa::968) without overlapping the plot.

Run:
    uv run python -m src.paper_figures.fig_hero
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.interpolate import PchipInterpolator

from src.paper_figures.theme import (
    CMAP_DIVERGING,
    PAL,
    apply_theme,
    label_panel,
    model_result_path,
    save_fig,
)


_EXAMPLE_LINES = [
    "Real item: trivia_qa::968",
    "Q: In which US state did Bill Gates",
    "   found Microsoft in April 1975?",
    "",
    "Model prior: California",
    "P(correct=New Mexico) at baseline: 0.01%",
    "",
    "Expert says: New Mexico, not California",
    "No instruction:  P(correct)=99.9%",
    "'Be correct':    P(correct)= 6.0%",
]

_EXAMPLE_LINES_COMPACT = [
    "trivia_qa::968",
    "Prior: California  [wrong]",
    "Expert: New Mexico, not CA",
    "No instr:  P(correct) = 99.9%",
    "Be correct: P(correct) =  6.0%",
]

_EMOJI_BRAIN_PATH = Path("src/paper_figures/assets/emoji_brain.png")
_EMOJI_DOC_PATH = Path("src/paper_figures/assets/emoji_doc.png")

_MODELS = [
    {
        "key": "llama",
        "prefix": "meta-llama__Llama-3.1-8B-Instruct",
        "exp11_dir": "exp11_extended",
    },
    {
        "key": "qwen",
        "prefix": "Qwen__Qwen3-4B-Instruct-2507",
        "exp11_dir": "exp11_extended",
    },
    {
        "key": "qwen_thinking",
        "prefix": "Qwen__Qwen3-4B-Thinking-2507",
        "exp11_dir": "exp11_extended",
    },
]

_PANEL_A_MODEL = "qwen"
_PANEL_A_EXP10 = "exp10_extended/Qwen__Qwen3-4B-Instruct-2507_results.jsonl"

# Model line styles used in panel B (color is metric color)
_MODEL_LS = {"llama": "-", "qwen": "--", "qwen_thinking": ":"}


def _load_jsonl(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _item_metrics(item: dict, tag: str) -> dict | None:
    sm = item.get("selectivity_metrics", {})
    cr = item.get("condition_results", {})

    n0_key = f"N0_{tag}"
    if n0_key not in cr:
        return None

    n0 = cr[n0_key]
    correct = item.get("correct_label", "a").lower()
    m_n0 = (
        (n0["logit_a"] - n0["logit_b"])
        if correct == "a"
        else (n0["logit_b"] - n0["logit_a"])
    )

    sel = sm.get(f"selectivity_{tag}")
    if sel is None:
        return None

    return {"m_n0": float(m_n0), "sel": float(sel)}


def _rolling(x: np.ndarray, y: np.ndarray, n_bins: int = 30, min_count: int = 18):
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    centers, means, lo, hi = [], [], [], []
    for i in range(n_bins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if mask.sum() < min_count:
            continue
        y_bin = y[mask]
        m = float(np.mean(y_bin))
        se = float(np.std(y_bin, ddof=1) / math.sqrt(len(y_bin)))
        centers.append((edges[i] + edges[i + 1]) / 2)
        means.append(m)
        lo.append(m - 1.96 * se)
        hi.append(m + 1.96 * se)
    return np.array(centers), np.array(means), np.array(lo), np.array(hi)


def _smooth(
    c: np.ndarray, m: np.ndarray, lo: np.ndarray, hi: np.ndarray, n_pts: int = 320
):
    x = np.linspace(c[0], c[-1], n_pts)
    return (
        x,
        PchipInterpolator(c, m)(x),
        PchipInterpolator(c, lo)(x),
        PchipInterpolator(c, hi)(x),
    )


def _unpack(metrics: dict, key: str):
    val = metrics[key]  # [median, [lo, hi]]
    return float(val[0]), float(val[1][0]), float(val[1][1])


def _draw_emoji_icon(
    ax: plt.Axes, x: float, y: float, image_path: Path, zoom: float = 0.20
) -> None:
    """Place a PNG emoji icon at axes-fraction coordinates."""
    if not image_path.exists():
        return
    img = plt.imread(image_path)
    icon = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(
        icon,
        (x, y),
        xycoords=ax.transAxes,
        frameon=False,
        box_alignment=(0.5, 0.5),
        zorder=12,
    )
    ax.add_artist(ab)


def _plot_panel_a(
    ax: plt.Axes,
    items: list,
    *,
    compact_callout: bool = False,
    show_callout: bool = True,
) -> None:
    sel_clip = 0.52
    expert = [m for it in items for m in [_item_metrics(it, "expert")] if m is not None]
    note = [m for it in items for m in [_item_metrics(it, "note")] if m is not None]

    x_exp = np.array([p["m_n0"] for p in expert])
    y_exp = np.clip(np.array([p["sel"] for p in expert]), -sel_clip, sel_clip)

    norm = mcolors.TwoSlopeNorm(vmin=x_exp.min(), vcenter=0.0, vmax=x_exp.max())
    ax.scatter(
        x_exp,
        y_exp,
        c=x_exp,
        cmap=CMAP_DIVERGING,
        norm=norm,
        s=2.5,
        alpha=0.07,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    ax.set_xlim(-27, 27)
    ax.set_ylim(-sel_clip, sel_clip)
    ax.axhspan(
        0.0, sel_clip, color=PAL.positive_region, alpha=0.45, linewidth=0, zorder=0
    )
    ax.axhspan(
        -sel_clip, 0.0, color=PAL.negative_region, alpha=0.45, linewidth=0, zorder=0
    )
    ax.axhline(0.0, color=PAL.faint_gray, lw=0.7, alpha=0.8, zorder=0)
    ax.axvline(0.0, color=PAL.faint_gray, lw=0.8, ls="--", alpha=0.55, zorder=1)

    endpoints = []
    note_curve_x = None
    note_curve_y = None
    for tag_name, points, color in [
        ("Expert", expert, PAL.blue),
        ("Note", note, PAL.red),
    ]:
        xx = np.array([p["m_n0"] for p in points])
        yy = np.clip(np.array([p["sel"] for p in points]), -sel_clip, sel_clip)
        c, m, lo, hi = _rolling(xx, yy, n_bins=28, min_count=18)
        if len(c) < 4:
            continue
        sx, sy, slo, shi = _smooth(c, m, lo, hi)
        ax.fill_between(sx, slo, shi, color=color, alpha=0.13, linewidth=0, zorder=2)
        ax.plot(sx, sy, color=color, lw=2.2, zorder=3, solid_capstyle="round")
        endpoints.append((tag_name, color, float(sx[-1]), float(sy[-1])))
        if tag_name == "Note":
            note_curve_x = sx
            note_curve_y = sy

    for tag_name, color, x_end, y_end in endpoints:
        ax.text(
            x_end - 2.5,
            y_end + (0.05 if tag_name == "Expert" else -0.05),
            tag_name,
            color=color,
            fontsize=7.5,
            fontweight="bold",
            ha="right",
            va="center",
        )

    ax.text(
        -26.5,
        -sel_clip + 0.01,
        "<- model wrong",
        fontsize=6.2,
        color=PAL.medium_gray,
        fontstyle="italic",
        va="bottom",
        ha="left",
    )
    ax.text(
        26.5,
        -sel_clip + 0.01,
        "model correct ->",
        fontsize=6.2,
        color=PAL.medium_gray,
        fontstyle="italic",
        va="bottom",
        ha="right",
    )

    ax.set_xlabel("Prior confidence ($m_{N_0}$)", fontsize=8, labelpad=3)
    ax.set_ylabel("Instruction selectivity", fontsize=8, labelpad=3)
    ax.set_title(
        "Qwen3-4B-Instruct", loc="center", fontsize=9, color=PAL.medium_gray, pad=4
    )

    # Embedded example callout — full or compact depending on context.
    if show_callout and compact_callout:
        box_x, box_y = 0.56, 0.16
        ax.text(
            box_x,
            box_y,
            "\n".join(_EXAMPLE_LINES_COMPACT),
            transform=ax.transAxes,
            fontsize=5.6,
            color=PAL.dark_text,
            va="bottom",
            ha="left",
            fontfamily="monospace",
            linespacing=1.28,
            zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor=PAL.bg_card,
                edgecolor=PAL.faint_gray,
                linewidth=0.6,
                alpha=0.96,
            ),
        )
        if note_curve_x is not None and note_curve_y is not None:
            idx = int(np.argmin(np.abs(note_curve_x - (-7.5))))
            ax.annotate(
                "",
                xy=(float(note_curve_x[idx]), float(note_curve_y[idx])),
                xycoords="data",
                xytext=(box_x + 0.010, box_y + 0.10),
                textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-|>,head_width=0.15,head_length=0.2",
                    color=PAL.red,
                    lw=0.7,
                    mutation_scale=6,
                    connectionstyle="arc3,rad=0.16",
                ),
                zorder=11,
            )
    elif show_callout:
        box_x, box_y = 0.565, 0.074
        text_x = box_x + 0.038
        ax.text(
            text_x,
            box_y,
            "\n".join(_EXAMPLE_LINES),
            transform=ax.transAxes,
            fontsize=6.15,
            color=PAL.dark_text,
            va="bottom",
            ha="left",
            fontfamily="monospace",
            linespacing=1.34,
            zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.46",
                facecolor=PAL.bg_card,
                edgecolor=PAL.faint_gray,
                linewidth=0.7,
                alpha=0.96,
            ),
        )
        icon_x = box_x + 0.015
        _draw_emoji_icon(ax, icon_x, box_y + 0.169, _EMOJI_BRAIN_PATH, zoom=0.19)
        _draw_emoji_icon(ax, icon_x, box_y + 0.088, _EMOJI_DOC_PATH, zoom=0.19)
        if note_curve_x is not None and note_curve_y is not None:
            idx = int(np.argmin(np.abs(note_curve_x - (-7.5))))
            ax.annotate(
                "",
                xy=(float(note_curve_x[idx]), float(note_curve_y[idx])),
                xycoords="data",
                xytext=(box_x + 0.020, box_y + 0.274),
                textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=PAL.red,
                    lw=0.8,
                    connectionstyle="arc3,rad=0.16",
                ),
                zorder=11,
            )


def _plot_panel_b(ax: plt.Axes, base: Path, *, show_title: bool = True) -> None:
    x = np.arange(3)
    labels = ["All\nitems", "Model\nwrong", "Confidently\nwrong"]

    for model in _MODELS:
        key = model["key"]
        style = PAL.MODEL_STYLES[key]

        pa_path = model_result_path(
            base,
            key,
            f"{model['exp11_dir']}/part_a/{model['prefix']}_note_metrics.json",
        )
        pb_path = model_result_path(
            base,
            key,
            f"{model['exp11_dir']}/part_b/{model['prefix']}_note_inverted_prior.json",
        )

        part_a = _load_json(pa_path)["metrics_sign_consistent"]
        slices = _load_json(pb_path)["slices"]
        seq = [part_a, slices["full"]["metrics"], slices["high_conf_top25"]["metrics"]]

        for metric, metric_color in [("r_w_median", PAL.r_w), ("r_c_median", PAL.r_c)]:
            vals, lo, hi = [], [], []
            for block in seq:
                try:
                    m, l, h = _unpack(block, metric)
                except (KeyError, TypeError, IndexError):
                    m, l, h = np.nan, np.nan, np.nan
                vals.append(m)
                lo.append(l)
                hi.append(h)

            y = np.array(vals, dtype=float)

            if key == "qwen":
                ax.fill_between(
                    x,
                    np.array(lo),
                    np.array(hi),
                    color=metric_color,
                    alpha=0.10,
                    linewidth=0,
                    zorder=1,
                )

            ax.plot(
                x,
                y,
                color=metric_color,
                linestyle=_MODEL_LS[key],
                linewidth=2.0 if key == "qwen" else 1.7,
                marker=style["marker"],
                markersize=5.3,
                markerfacecolor=style["color"],
                markeredgecolor=PAL.bg_warm,
                markeredgewidth=0.7,
                alpha=0.95 if key == "qwen" else 0.88,
                zorder=3,
                solid_capstyle="round",
            )

    qwen_metrics = _load_json(
        model_result_path(
            base,
            "qwen",
            "exp11_extended/part_b/Qwen__Qwen3-4B-Instruct-2507_note_inverted_prior.json",
        )
    )["slices"]["high_conf_top25"]["metrics"]
    rw_q = _unpack(qwen_metrics, "r_w_median")[0]
    rc_q = _unpack(qwen_metrics, "r_c_median")[0]

    ax.annotate(
        rf"$r_w \sim {rw_q:.2f}$",
        xy=(2, rw_q),
        xytext=(8, -2),
        textcoords="offset points",
        fontsize=7.0,
        color=PAL.r_w,
        va="center",
        ha="left",
        fontweight="bold",
    )
    ax.annotate(
        rf"$r_c \sim {rc_q:.2f}$",
        xy=(2, rc_q),
        xytext=(8, 2),
        textcoords="offset points",
        fontsize=7.0,
        color=PAL.r_c,
        va="center",
        ha="left",
        fontweight="bold",
    )

    qt_metrics = _load_json(
        model_result_path(
            base,
            "qwen_thinking",
            "exp11_extended/part_b/Qwen__Qwen3-4B-Thinking-2507_note_inverted_prior.json",
        )
    )["slices"]["high_conf_top25"]["metrics"]
    qt_dr = _unpack(qt_metrics, "dr_median")[0]
    qt_rc = _unpack(qt_metrics, "r_c_median")[0]
    qt_rw = _unpack(qt_metrics, "r_w_median")[0]
    qt_mid = 0.5 * (qt_rc + qt_rw)

    ax.annotate(
        rf"Qwen-T $\Delta r \sim {qt_dr:.2f}$",
        xy=(2, qt_mid),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=6.5,
        color=PAL.MODEL_STYLES["qwen_thinking"]["color"],
        ha="left",
        va="center",
        fontstyle="italic",
    )

    ax.axhline(0.0, color=PAL.faint_gray, lw=0.7, alpha=0.8, zorder=0)
    ax.yaxis.grid(True, alpha=0.18, linewidth=0.5, color=PAL.faint_gray)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.32, 2.70)
    ax.set_ylim(-0.62, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Suppression ratio", fontsize=8)
    if show_title:
        ax.set_title(r"Note tag: $r_w$ vs $r_c$ across slices", loc="center", fontsize=8.5)


def _legend_panel_b(fig: plt.Figure) -> None:
    metric_handles = [
        mlines.Line2D(
            [], [], color=PAL.r_w, lw=2.0, label=r"$r_w$ (wrong-endorse suppression)"
        ),
        mlines.Line2D(
            [], [], color=PAL.r_c, lw=2.0, label=r"$r_c$ (correct-endorse suppression)"
        ),
    ]

    model_handles = []
    for model in _MODELS:
        key = model["key"]
        style = PAL.MODEL_STYLES[key]
        ms = 5.0 if key != "qwen_thinking" else 4.6
        model_handles.append(
            mlines.Line2D(
                [],
                [],
                color=PAL.medium_gray,
                linestyle=_MODEL_LS[key],
                lw=1.7,
                marker=style["marker"],
                markersize=ms,
                markerfacecolor=style["color"],
                markeredgecolor=PAL.bg_warm,
                markeredgewidth=0.6,
                markevery=[1],
                label=style["label"],
            )
        )

    handles = metric_handles + model_handles
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=7,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=1.1,
        frameon=False,
        bbox_to_anchor=(0.50, 0.01),
    )


def _make_fig3a(items: list, output_dir: Path, formats: list[str]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 4.35), constrained_layout=False)

    _plot_panel_a(ax, items)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.16)

    return save_fig(fig, "fig3a_prior_consistency_geometry", output_dir, formats)


def _make_fig3b(base: Path, output_dir: Path, formats: list[str]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.82, top=0.96, bottom=0.30)

    _plot_panel_b(ax, base)
    _legend_panel_b(fig)

    return save_fig(fig, "fig3b_prior_consistency_decomposition", output_dir, formats)


def _make_fig3_combined(
    items: list, base: Path, output_dir: Path, formats: list[str]
) -> list[Path]:
    """Two-panel double-column Figure 3 for ACL \\begin{figure*} environment.

    Panel A (left, ~55% inner width): prior-confidence geometry scatter.
    Panel B (right, ~45% inner width): r_w vs r_c decomposition.

    Target: 7.0-inch text width, so both panels get ~3 inches each after
    accounting for a modest wspace.  The compact callout in panel A keeps
    the right half of the scatter unobscured.
    """
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.6),
        gridspec_kw={"width_ratios": [1.1, 0.9]},
        constrained_layout=False,
    )
    # wspace=0.25  → gap is 25% of mean axes width ≈ 0.7 in, leaving each
    # panel ~3.0 in of drawable area.
    # right=0.90   → 10% right margin for panel-B annotation labels.
    fig.subplots_adjust(left=0.07, right=0.90, top=0.94, bottom=0.26, wspace=0.25)

    _plot_panel_a(ax_a, items, compact_callout=True, show_callout=False)
    _plot_panel_b(ax_b, base, show_title=False)

    # Give panel-B a little extra x-room so the right-side labels sit inside.
    ax_b.set_xlim(-0.32, 2.95)

    label_panel(ax_a, "A")
    label_panel(ax_b, "B")

    _legend_panel_b(fig)

    return save_fig(fig, "fig3_combined", output_dir, formats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("new-phase-results/figures/paper")
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()

    apply_theme()
    base = Path("new-phase-results")
    items = _load_jsonl(model_result_path(base, _PANEL_A_MODEL, _PANEL_A_EXP10))

    out_a = _make_fig3a(items, args.output_dir, args.formats)
    out_b = _make_fig3b(base, args.output_dir, args.formats)
    out_c = _make_fig3_combined(items, base, args.output_dir, args.formats)

    for p in out_a + out_b + out_c:
        print(f"  {p}")


if __name__ == "__main__":
    main()
