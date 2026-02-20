#!/usr/bin/env python3
"""Evidence-quality scaling by speaker tag (4-tag comparison).

Layout: 3 columns (models) x 2 rows (correct / wrong endorsement shift).
Each panel has 4 lines: Expert, Note, User, Someone online.
Uses Okabe-Ito palette for CVD-safe tag discrimination.

Data sources (per model):
  - plus_online combined_analysis.json  -> expert, note, someone_online
  - user_only   combined_analysis.json  -> user

Run:
    uv run python -m src.paper_figures.fig_evidence_quality_tags
    uv run python -m src.paper_figures.fig_evidence_quality_tags --slice-name prior_wrong
    uv run python -m src.paper_figures.fig_evidence_quality_tags --shared-y
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.theme import (
    PAL,
    apply_theme,
    add_y_grid,
    add_zero_line,
    save_fig,
)

# ── Models ────────────────────────────────────────────────────────────

MODEL_ORDER = [
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B\nInstruct"),
    ("Qwen/Qwen3-4B-Instruct-2507", "Qwen 3 4B\nInstruct"),
    ("Qwen/Qwen3-4B-Thinking-2507", "Qwen 3 4B\nThinking"),
]

# ── Tags & Okabe-Ito palette ─────────────────────────────────────────

TAG_ORDER: List[Tuple[str, str]] = [
    ("expert", "Expert"),
    ("note", "Note"),
    ("user", "User"),
    ("someone_online", "Someone online"),
]

# Okabe-Ito subset (yellow skipped — poor contrast on warm cream).
TAG_COLORS: Dict[str, str] = {
    "expert": "#0072B2",  # blue
    "note": "#D55E00",  # vermillion
    "user": "#009E73",  # bluish green
    "someone_online": "#CC79A7",  # reddish purple
}

TAG_MARKERS: Dict[str, str] = {
    "expert": "o",
    "note": "s",
    "user": "D",
    "someone_online": "^",
}

# ── Evidence labels ───────────────────────────────────────────────────

EVIDENCE_LABELS = {
    "bare": "E0",
    "reason1": "E1",
    "reason2": "E2",
    "reason_data": "E3",
}

# ── Direction labels for row titles ───────────────────────────────────

DIRECTION_ORDER: List[Tuple[str, str]] = [
    ("correct", "Correct endorsement"),
    ("wrong", "Wrong endorsement"),
]

# ── Helpers ───────────────────────────────────────────────────────────


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _smooth_curve(
    x: np.ndarray,
    y: np.ndarray,
    n: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """PCHIP interpolation — smooth without overshoot."""
    try:
        from scipy.interpolate import PchipInterpolator
    except Exception:
        return x, y
    if x.size < 2:
        return x, y
    xp = np.linspace(float(x.min()), float(x.max()), n)
    yp = PchipInterpolator(x, y)(xp)
    return xp, yp


def _extract_delta_series(
    model_payload: dict,
    *,
    slice_name: str,
    tag: str,
    instruction_key: str,
    direction: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], int]]:
    """Return (means_delta, ci_lo_delta, ci_hi_delta, levels, n_items) or None."""
    try:
        payload = model_payload["monotonicity"][slice_name][tag][instruction_key][
            direction
        ]
    except KeyError:
        return None

    level_rows = payload["mean_by_level"]
    levels = model_payload["evidence_levels"]

    means = np.array([float(row["mean"]) for row in level_rows], dtype=float)
    lo = np.array([float(row["ci95_low"]) for row in level_rows], dtype=float)
    hi = np.array([float(row["ci95_high"]) for row in level_rows], dtype=float)

    # Delta from E0 so all lines start at 0.
    means_d = means - means[0]
    lo_d = lo - hi[0]
    hi_d = hi - lo[0]
    return means_d, lo_d, hi_d, levels, int(payload["n_items"])


def _add_semantic_gradient(ax: plt.Axes, direction: str = "correct") -> None:
    """Add vertical semantic background tint.

    'correct' row: green-at-top (larger shift = model updating toward truth).
    'wrong' row: rose-at-top (larger shift = more susceptibility to wrong evidence).
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    grad = np.linspace(0, 1, 512).reshape(512, 1)

    if direction == "correct":
        colors = [PAL.negative_region, PAL.bg_warm, PAL.positive_region]
    else:
        # Rose-at-top: higher wrong-endorsement shift = more susceptible.
        colors = [PAL.bg_warm, PAL.bg_warm, PAL.negative_region]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "tag_semantic_vertical",
        colors,
    )
    ax.imshow(
        grad,
        extent=[x0, x1, y0, y1],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        alpha=0.45,
        interpolation="bicubic",
        zorder=0,
    )


# ── Data loading ──────────────────────────────────────────────────────


def _resolve_model_data(
    results_root: Path,
    model_dirs: Dict[str, str],
) -> List[Dict[str, dict]]:
    """Load & merge plus_online + user_only analysis for each model.

    Returns a list (one per model in MODEL_ORDER) of dicts mapping
    tag_key -> model_payload.
    """
    merged: List[Dict[str, dict]] = []

    for model_id, _label in MODEL_ORDER:
        model_dir = model_dirs.get(model_id)
        if model_dir is None:
            continue

        tag_payloads: Dict[str, dict] = {}

        # plus_online has expert, note, someone_online.
        plus_online_path = (
            results_root
            / model_dir
            / "exp14"
            / "assertive"
            / "plus_online"
            / "analysis"
            / "combined_analysis.json"
        )
        if plus_online_path.exists():
            combined = load_json(plus_online_path)
            for m in combined.get("models", []):
                if m["model"] == model_id:
                    for t in m.get("tags", []):
                        tag_payloads[t] = m
                    break

        # user_only has user.
        user_only_path = (
            results_root
            / model_dir
            / "exp14"
            / "assertive"
            / "user_only"
            / "analysis"
            / "combined_analysis.json"
        )
        if user_only_path.exists():
            combined = load_json(user_only_path)
            for m in combined.get("models", []):
                if m["model"] == model_id:
                    tag_payloads["user"] = m
                    break

        if tag_payloads:
            merged.append({"model_id": model_id, "tags": tag_payloads})

    return merged


# ── Figure construction ───────────────────────────────────────────────


def make_figure(
    model_data: List[Dict],
    *,
    output_dir: Path,
    formats: Sequence[str],
    slice_name: str,
    instruction_key: str,
    shared_y: bool = False,
) -> None:
    n_cols = len(model_data)
    n_rows = len(DIRECTION_ORDER)
    is_prior_wrong = slice_name.startswith("prior_wrong")

    fig = plt.figure(figsize=(7.2, 5.6), constrained_layout=False)
    fig.subplots_adjust(
        left=0.09,
        right=0.97,
        top=0.84,
        bottom=0.15,
        wspace=0.32,
        hspace=0.32,
    )
    gs = fig.add_gridspec(n_rows, n_cols)

    axes_grid: List[List[plt.Axes]] = []
    for row_idx in range(n_rows):
        row_axes = [fig.add_subplot(gs[row_idx, col_idx]) for col_idx in range(n_cols)]
        axes_grid.append(row_axes)

    # ── Compute y-limits ──────────────────────────────────────────────
    # Gather per-row ranges first.
    row_ranges: Dict[str, Tuple[float, float]] = {}
    for dir_key, _ in DIRECTION_ORDER:
        mins: List[float] = [0.0]
        maxs: List[float] = [0.0]
        for md in model_data:
            for tag_key, _ in TAG_ORDER:
                payload = md["tags"].get(tag_key)
                if payload is None:
                    continue
                series = _extract_delta_series(
                    payload,
                    slice_name=slice_name,
                    tag=tag_key,
                    instruction_key=instruction_key,
                    direction=dir_key,
                )
                if series is None:
                    continue
                _m, lo, hi, _l, _n = series
                mins.append(float(np.min(lo)))
                maxs.append(float(np.max(hi)))
        y0, y1 = min(mins), max(maxs)
        pad = 0.12 * (y1 - y0 + 1e-6)
        row_ranges[dir_key] = (y0 - pad, y1 + pad)

    if shared_y:
        # Unify across both rows for honest cross-row comparison.
        global_min = min(r[0] for r in row_ranges.values())
        global_max = max(r[1] for r in row_ranges.values())
        row_limits = {k: (global_min, global_max) for k in row_ranges}
    else:
        row_limits = row_ranges

    x = np.arange(4, dtype=float)
    last_levels: List[str] = ["bare", "reason1", "reason2", "reason_data"]

    # ── Plot ──────────────────────────────────────────────────────────
    for row_idx, (dir_key, dir_label) in enumerate(DIRECTION_ORDER):
        ylim = row_limits[dir_key]

        for col_idx, md in enumerate(model_data):
            ax = axes_grid[row_idx][col_idx]
            ax.set_xlim(-0.20, 3.20)
            ax.set_ylim(*ylim)

            _add_semantic_gradient(ax, direction=dir_key)
            add_zero_line(ax, alpha=0.75, linewidth=0.75)
            add_y_grid(ax, alpha=0.18)

            n_by_tag: Dict[str, int] = {}

            for tag_key, tag_label in TAG_ORDER:
                payload = md["tags"].get(tag_key)
                if payload is None:
                    continue

                series = _extract_delta_series(
                    payload,
                    slice_name=slice_name,
                    tag=tag_key,
                    instruction_key=instruction_key,
                    direction=dir_key,
                )
                if series is None:
                    continue

                means, lo, hi, levels, n_items = series
                last_levels = levels
                n_by_tag[tag_key] = n_items
                color = TAG_COLORS[tag_key]
                marker = TAG_MARKERS[tag_key]

                # Dash User line in prior_wrong to flag different prior.
                linestyle = "--" if (is_prior_wrong and tag_key == "user") else "-"

                xs, ys = _smooth_curve(x, means, n=200)

                # CI band.
                ax.fill_between(
                    x,
                    lo,
                    hi,
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                    zorder=1,
                )
                # Smooth line.
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linewidth=2.0,
                    alpha=0.92,
                    linestyle=linestyle,
                    zorder=3,
                )
                # Data-point markers.
                ax.scatter(
                    x,
                    means,
                    s=28,
                    color=color,
                    marker=marker,
                    edgecolors=PAL.bg_warm,
                    linewidths=0.7,
                    zorder=4,
                )

            # ── Semantic arrow for wrong row (rightmost column) ───
            if dir_key == "wrong" and col_idx == n_cols - 1:
                ax.annotate(
                    "more\nsusceptible",
                    xy=(0.98, 0.95),
                    xycoords="axes fraction",
                    fontsize=5.5,
                    fontstyle="italic",
                    color=PAL.medium_gray,
                    ha="right",
                    va="top",
                )

            # ── N annotation ──────────────────────────────────────
            if row_idx == 0 and n_by_tag:
                if is_prior_wrong:
                    n_main = max(
                        (v for k, v in n_by_tag.items() if k != "user"),
                        default=0,
                    )
                    n_user = n_by_tag.get("user", 0)
                    if n_user and n_main and n_user != n_main:
                        n_text = f"n={n_main}\nUser n={n_user}"
                    elif n_main:
                        n_text = f"n={n_main}"
                    else:
                        n_text = f"n={n_user}"
                    ax.text(
                        0.97,
                        0.96,
                        n_text,
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=6.2,
                        color=PAL.medium_gray,
                        linespacing=1.3,
                    )
                elif col_idx == n_cols - 1:
                    # Single global n for all-items (same across panels).
                    n_val = max(n_by_tag.values())
                    ax.text(
                        0.97,
                        0.96,
                        f"n={n_val} per tag",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=6.2,
                        color=PAL.medium_gray,
                    )

            # ── Y-axis label (left column) ────────────────────────
            if col_idx == 0:
                ax.set_ylabel(
                    f"{dir_label}\n\u0394 logit shift from E0",
                    fontsize=7.5,
                )
            else:
                ax.set_ylabel("")

            # ── X-axis (bottom row) ───────────────────────────────
            if row_idx == n_rows - 1:
                tick_labels = [EVIDENCE_LABELS.get(lv, lv) for lv in last_levels]
                ax.set_xticks(x)
                ax.set_xticklabels(tick_labels, fontsize=7.2)
                ax.set_xlabel("Evidence level", fontsize=8)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([""] * 4)

    # ── Column headers ────────────────────────────────────────────────
    for col_idx, md in enumerate(model_data):
        model_id = md["model_id"]
        short = next(
            (label for key, label in MODEL_ORDER if key == model_id),
            model_id,
        )
        bbox = axes_grid[0][col_idx].get_position()
        x_mid = 0.5 * (bbox.x0 + bbox.x1)
        fig.text(
            x_mid,
            0.875,
            short,
            ha="center",
            va="bottom",
            fontsize=8.8,
            color=PAL.dark_text,
            fontweight="bold",
            linespacing=1.0,
        )

    # ── Legend ─────────────────────────────────────────────────────────
    legend_handles = []
    for tag_key, tag_label in TAG_ORDER:
        ls = "--" if (is_prior_wrong and tag_key == "user") else "-"
        lbl = (
            f"{tag_label} (diff. prior)"
            if (is_prior_wrong and tag_key == "user")
            else tag_label
        )
        legend_handles.append(
            mlines.Line2D(
                [0],
                [0],
                color=TAG_COLORS[tag_key],
                marker=TAG_MARKERS[tag_key],
                markersize=5,
                markeredgecolor=PAL.bg_warm,
                markeredgewidth=0.6,
                linewidth=2,
                linestyle=ls,
                label=lbl,
            )
        )

    # Position legend.
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        fontsize=7.0,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    # ── Title ─────────────────────────────────────────────────────────
    if is_prior_wrong:
        slice_note = "prior-wrong slice"
    else:
        slice_note = "all items"

    if shared_y:
        title = (
            f"Evidence-quality scaling by speaker tag  ({slice_note}, shared y-axis)"
        )
    else:
        title = f"Evidence-quality scaling by speaker tag  ({slice_note})"

    fig.suptitle(
        title,
        fontsize=10,
        fontweight="bold",
        y=0.97,
        color=PAL.dark_text,
    )

    # ── Save ──────────────────────────────────────────────────────────
    suffix = "_shared_y" if shared_y else ""
    stem = f"fig_evidence_quality_tags_{slice_name}{suffix}"
    for path in save_fig(fig, stem, output_dir, formats):
        print(f"  {path}")


# ── CLI ───────────────────────────────────────────────────────────────

_MODEL_DIRS: Dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b-results",
    "Qwen/Qwen3-4B-Instruct-2507": "qwen3-4b-results",
    "Qwen/Qwen3-4B-Thinking-2507": "qwen3-4b-thinking-results",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evidence-quality scaling figure with 4 speaker tags.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("new-phase-results"),
    )
    parser.add_argument(
        "--slice-name",
        default="all",
        choices=["all", "prior_wrong", "prior_wrong_high_conf_top25"],
    )
    parser.add_argument(
        "--instruction-key",
        default="I0",
        choices=["I0", "I1"],
    )
    parser.add_argument(
        "--shared-y",
        action="store_true",
        help="Use a single y-axis scale across both rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("new-phase-results/figures/paper"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
    )
    args = parser.parse_args()

    apply_theme()

    model_data = _resolve_model_data(args.results_root, _MODEL_DIRS)
    if not model_data:
        raise ValueError(
            "No model data found. Check that plus_online and user_only "
            "analysis directories exist under --results-root."
        )

    make_figure(
        model_data,
        output_dir=args.output_dir,
        formats=args.formats,
        slice_name=args.slice_name,
        instruction_key=args.instruction_key,
        shared_y=args.shared_y,
    )


if __name__ == "__main__":
    main()
