"""Fig 4: PIQA out-of-domain transfer — 4 models × 3 layers.

For each model we patch the compliance direction extracted from
factual-QA into the N0 hidden state and measure the change in
accuracy / wrong-answer rate on PIQA (a disjoint physical-commonsense
benchmark).  A two-line panel per cell: blue = Δ correct, red = Δ
wrong.

Run::

    uv run python -m src.paper_figures.fig_piqa_transfer_multi
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.axis_theme import (
    AXIS,
    apply_axis_theme,
    line_with_markers,
    save_fig,
    soften_spines,
    y_grid,
)


OUTPUT_DIR = Path("figures/neurips/v2/final/appendix")

MODELS: List[Dict] = [
    {
        "name": "Qwen-3.5-27B",
        "path": Path("neurips-results/qwen35/mechanism/piqa_forward_patch_200_freegen/piqa_summary.json"),
        "layers": (2, 5, 10),
        "accent": "#5B84B5",
    },
    {
        "name": "GPT-oss-20B",
        "path": Path("neurips-results/gpt-oss/mechanism/piqa_forward_patch_200_freegen/piqa_summary.json"),
        "layers": (16, 18, 20),
        "accent": "#C36662",
    },
    {
        "name": "Gemma-4-26B",
        "path": Path("neurips-results/gemma4/mechanism/piqa_forward_patch_200_freegen_no_thinking/piqa_summary.json"),
        "layers": (20, 22, 24),
        "accent": "#6F9C79",
    },
    {
        "name": "OLMo-2-32B",
        "path": Path("neurips-results/olmo2/mechanism/piqa_forward_patch_200_freegen/piqa_summary.json"),
        "layers": (10, 16, 22),
        "accent": "#D5A048",
    },
]


def _load_series(path: Path, layer: int):
    rows = json.loads(path.read_text())
    rows = [r for r in rows if r["condition_code"] == "N0_note"
            and int(r["target_layers"][0]) == layer]
    rows.sort(key=lambda r: float(r["alpha"]))
    alphas = np.array([float(r["alpha"]) for r in rows])
    correct = np.array([100.0 * float(r["accuracy_parsed"]) for r in rows])
    wrong = np.array([100.0 * float(r["wrong_rate_parsed"]) for r in rows])
    # Δ from baseline
    correct_d = correct - correct[0]
    wrong_d = wrong - wrong[0]
    return alphas, correct_d, wrong_d


def _smooth(alphas: np.ndarray, ys: np.ndarray):
    try:
        from scipy.interpolate import PchipInterpolator
        xs = np.linspace(float(alphas.min()), float(alphas.max()), 240)
        ysm = PchipInterpolator(alphas, ys)(xs)
        return xs, ysm
    except Exception:
        return alphas, ys


def _draw_panel(
    ax: plt.Axes,
    alphas: np.ndarray,
    correct_d: np.ndarray,
    wrong_d: np.ndarray,
    y_lim=(-12, 12),
) -> None:
    # Smooth curves (PCHIP), markers at measured alphas.
    xs_w, ys_w = _smooth(alphas, wrong_d)
    xs_c, ys_c = _smooth(alphas, correct_d)
    ax.plot(xs_w, ys_w, color=AXIS.red, linewidth=1.9, zorder=3, label="Δ wrong answer")
    ax.plot(xs_c, ys_c, color=AXIS.blue, linewidth=1.9, zorder=3, label="Δ correct answer")
    ax.plot(
        alphas, wrong_d, marker="o", linestyle="None",
        color=AXIS.red, markerfacecolor=AXIS.red,
        markeredgecolor=AXIS.bg_card, markeredgewidth=0.9,
        markersize=5.0, zorder=4,
    )
    ax.plot(
        alphas, correct_d, marker="o", linestyle="None",
        color=AXIS.blue, markerfacecolor=AXIS.blue,
        markeredgecolor=AXIS.bg_card, markeredgewidth=0.9,
        markersize=5.0, zorder=4,
    )

    ax.axhline(0, color=AXIS.rule, linewidth=0.8, zorder=1)
    ax.axvline(0, color=AXIS.rule, linestyle=(0, (2, 3)), linewidth=0.8, zorder=1)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(*y_lim)
    ax.set_yticks([-12, -6, 0, 6, 12])
    y_grid(ax, alpha=0.55)
    soften_spines(ax)


def main() -> None:
    apply_axis_theme()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9.5, 7.4))
    fig.patch.set_facecolor(AXIS.bg_canvas)

    n_rows = len(MODELS)
    n_cols = 3
    gs = fig.add_gridspec(
        n_rows, n_cols,
        left=0.11, right=0.985,
        bottom=0.10, top=0.86,
        wspace=0.18, hspace=0.30,
    )

    # Collect legend handles once (from the first panel we draw).
    legend_handles = None
    legend_labels = None

    for r, cfg in enumerate(MODELS):
        for c, layer in enumerate(cfg["layers"]):
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor(AXIS.bg_card)
            try:
                alphas, correct_d, wrong_d = _load_series(cfg["path"], layer)
                _draw_panel(ax, alphas, correct_d, wrong_d)
            except Exception as exc:
                ax.text(
                    0.5, 0.5, f"missing: {exc}",
                    ha="center", va="center", transform=ax.transAxes,
                    color=AXIS.muted, fontsize=8,
                )
                ax.set_xticks([]); ax.set_yticks([])
                soften_spines(ax)
                continue

            ax.set_title(f"L{layer}", fontsize=10.0, fontweight="bold", pad=5)

            if c == 0:
                ax.set_ylabel("Δ response rate (pp)", fontsize=9.0, color=AXIS.ink_soft)
            else:
                ax.set_yticklabels([])
            if r == n_rows - 1:
                ax.set_xlabel("Patch magnitude (α)", fontsize=9.0, color=AXIS.ink_soft)
            else:
                ax.set_xticklabels([])

            if legend_handles is None:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    # order: wrong (red) first, correct (blue) second
                    seen = {}
                    for h, l in zip(handles, labels):
                        seen[l] = h
                    legend_handles = [seen.get("Δ wrong answer"), seen.get("Δ correct answer")]
                    legend_labels = ["Δ wrong answer", "Δ correct answer"]

        # Row label
        first_ax = fig.axes[-n_cols]
        first_ax.annotate(
            cfg["name"],
            xy=(-0.38, 0.5),
            xycoords="axes fraction",
            ha="center", va="center", rotation=90,
            fontsize=10.8, fontweight="bold", color=cfg["accent"],
        )

    # Legend
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="center", bbox_to_anchor=(0.5, 0.905),
            ncol=2, frameon=False,
            fontsize=9.2, handlelength=1.6, columnspacing=2.5,
        )

    fig.text(
        0.08, 0.94,
        "Patching the authority direction shifts answers on PIQA",
        fontsize=13.2, fontweight="bold", color=AXIS.ink, ha="left", va="bottom",
    )

    save_fig(fig, "piqa_transfer_multi", OUTPUT_DIR)
    print("Saved piqa_transfer_multi.*")


if __name__ == "__main__":
    main()
