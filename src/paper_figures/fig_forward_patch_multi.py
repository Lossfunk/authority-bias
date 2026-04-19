"""Fig 3: forward-patch flip rate — 4 models × 3 layers.

Small-multiples grid in the Assistant Axis Fig 4 style: one row per
model, one column per (shared) layer index.  A red shaded line shows
the right-to-wrong flip rate as the patching magnitude ``alpha``
sweeps from 0 → 1.

Run::

    uv run python -m src.paper_figures.fig_forward_patch_multi
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.axis_theme import (
    AXIS,
    apply_axis_theme,
    line_with_markers,
    save_fig,
    smooth_fill,
    soften_spines,
    y_grid,
)


OUTPUT_DIR = Path("figures/neurips/v2/final/appendix")

# (model_key, display_name, data path, layer triple, default colour)
MODELS: List[Dict] = [
    {
        "name": "Qwen-3.5-27B",
        "path": Path("neurips-results/qwen35/mechanism/forward_patch_expanded_512_w1source/steering_rows.jsonl"),
        "layers": (2, 5, 10),
        "accent": "#5B84B5",
    },
    {
        "name": "GPT-oss-20B",
        "path": Path("neurips-results/gpt-oss/mechanism/forward_patch_test_256/steering_rows.jsonl"),
        "layers": (16, 18, 20),
        "accent": "#C36662",
    },
    {
        "name": "Gemma-4-26B",
        "path": Path("neurips-results/gemma4/mechanism/forward_patch_test_256_no_thinking/steering_rows.jsonl"),
        "layers": (20, 22, 24),
        "accent": "#6F9C79",
    },
    {
        "name": "OLMo-2-32B",
        "path": Path("neurips-results/olmo2/mechanism/forward_patch_test_256/steering_rows.jsonl"),
        "layers": (10, 16, 22),
        "accent": "#D5A048",
    },
]


def _load_flip_rates(path: Path) -> Dict[int, List[Tuple[float, float]]]:
    """Return {layer: [(alpha, flip_rate_pct)]}."""
    grouped = defaultdict(lambda: defaultdict(dict))
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("condition_code") != "N0_note":
                continue
            layer = int(row["target_layers"][0])
            alpha = float(row["alpha"])
            grouped[layer][alpha][row["uid"]] = row
    out: Dict[int, List[Tuple[float, float]]] = {}
    for layer, by_alpha in grouped.items():
        base_rows = by_alpha.get(0.0, {})
        baseline_correct = {
            uid for uid, r in base_rows.items() if r.get("is_correct") is True
        }
        series: List[Tuple[float, float]] = []
        for alpha in sorted(by_alpha):
            rows = by_alpha[alpha]
            if not baseline_correct:
                series.append((alpha, 0.0))
                continue
            flips = sum(
                1
                for uid in baseline_correct
                if rows.get(uid, {}).get("chose_wrong") is True
            )
            series.append((alpha, 100.0 * flips / len(baseline_correct)))
        out[layer] = sorted(series)
    return out


def _draw_panel(
    ax: plt.Axes,
    alphas: np.ndarray,
    flips: np.ndarray,
    accent: str,
    y_lim: Tuple[float, float],
    show_peak: bool = True,
) -> None:
    smooth_fill(
        ax,
        alphas,
        flips,
        color=accent,
        fill_color=accent,
        alpha_fill=0.18,
        linewidth=1.6,
    )
    line_with_markers(
        ax,
        alphas,
        flips,
        color=accent,
        marker="o",
        linewidth=0.0,
        markersize=5.5,
    )
    if show_peak:
        idx = int(np.argmax(flips))
        x_peak = float(alphas[idx])
        y_peak = float(flips[idx])
        # annotation position
        offset_x = -6 if x_peak > 0.82 * alphas.max() else 8
        offset_y = -4 if y_peak > 0.80 * y_lim[1] else 7
        ha = "right" if offset_x < 0 else "left"
        va = "top" if offset_y < 0 else "bottom"
        ax.annotate(
            f"{y_peak:.1f}%",
            xy=(x_peak, y_peak),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=8.3,
            fontweight="bold",
            color=accent,
            ha=ha, va=va,
            bbox=dict(
                facecolor=AXIS.bg_card,
                edgecolor="none",
                pad=1.0,
                alpha=0.75,
            ),
        )

    ax.axvline(0.0, color=AXIS.rule, linestyle=(0, (2, 3)), linewidth=0.8, zorder=1)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(*y_lim)
    y_grid(ax, alpha=0.55)
    soften_spines(ax)


def main() -> None:
    apply_axis_theme()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_rates = {m["name"]: _load_flip_rates(m["path"]) for m in MODELS}

    # determine overall y range
    all_vals = [
        v for rates in model_rates.values() for series in rates.values() for _, v in series
    ]
    y_max = float(np.ceil(max(all_vals) / 5.0) * 5.0) + 2
    y_lim = (0.0, min(y_max, 42.0))

    fig = plt.figure(figsize=(9.5, 7.2))
    fig.patch.set_facecolor(AXIS.bg_canvas)

    n_rows = len(MODELS)
    n_cols = 3  # layers
    gs = fig.add_gridspec(
        n_rows, n_cols,
        left=0.11, right=0.985,
        bottom=0.10, top=0.90,
        wspace=0.18, hspace=0.30,
    )

    for r, cfg in enumerate(MODELS):
        layers = cfg["layers"]
        series_dict = model_rates[cfg["name"]]
        for c, layer in enumerate(layers):
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor(AXIS.bg_card)

            if layer not in series_dict:
                ax.text(
                    0.5, 0.5, f"L{layer} missing",
                    ha="center", va="center", transform=ax.transAxes,
                    color=AXIS.muted, fontsize=9,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                soften_spines(ax)
                continue

            series = series_dict[layer]
            alphas = np.array([a for a, _ in series])
            flips = np.array([v for _, v in series])
            _draw_panel(ax, alphas, flips, cfg["accent"], y_lim)

            # Titles
            ax.set_title(f"L{layer}", fontsize=10.0, fontweight="bold", pad=5)

            if c == 0:
                ax.set_ylabel("Right-to-wrong flips (%)", fontsize=9.0, color=AXIS.ink_soft)
            else:
                ax.set_yticklabels([])
            if r == n_rows - 1:
                ax.set_xlabel("Patch magnitude (α)", fontsize=9.0, color=AXIS.ink_soft)
            else:
                ax.set_xticklabels([])

        # Row label (model name) on the left
        first_ax = fig.axes[-n_cols]
        first_ax.annotate(
            cfg["name"],
            xy=(-0.38, 0.5),
            xycoords="axes fraction",
            ha="center", va="center",
            rotation=90,
            fontsize=10.8,
            fontweight="bold",
            color=cfg["accent"],
        )

    fig.text(
        0.08, 0.93,
        "Forward activation patching recovers the authority override",
        fontsize=13.2, fontweight="bold", color=AXIS.ink, ha="left", va="bottom",
    )

    save_fig(fig, "forward_patch_flip_multi", OUTPUT_DIR)
    print("Saved forward_patch_flip_multi.*")


if __name__ == "__main__":
    main()
