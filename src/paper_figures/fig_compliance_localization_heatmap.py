"""Cross-model localization heatmap for the authority-compliance signal.

Uses the W1 AUROC sweep across layer/position for GPT-oss, OLMo, and Gemma,
with the chosen primary pair marked in each panel.

Run:

    uv run python -m src.paper_figures.fig_compliance_localization_heatmap
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.axis_theme import AXIS, apply_axis_theme, save_fig, soften_spines


OUTPUT_DIR = Path("figures/neurips/v2/candidates")

MODELS: List[Dict] = [
    {
        "name": "GPT-oss-20B",
        "slug": "gpt-oss",
        "sweep_path": Path("neurips-results/gpt-oss/mechanism/gpt_oss_compliance_analysis/layer_position_sweep.json"),
        "probe_path": Path("neurips-results/gpt-oss/mechanism/gpt_oss_compliance_analysis/probe_results.json"),
    },
    {
        "name": "OLMo-2-32B",
        "slug": "olmo2",
        "sweep_path": Path("neurips-results/olmo2/mechanism/olmo2_compliance_analysis/layer_position_sweep.json"),
        "probe_path": Path("neurips-results/olmo2/mechanism/olmo2_compliance_analysis/probe_results.json"),
    },
    {
        "name": "Gemma-4-26B",
        "slug": "gemma4",
        "sweep_path": Path("neurips-results/gemma4/mechanism/gemma4_compliance_analysis_no_thinking/layer_position_sweep.json"),
        "probe_path": Path("neurips-results/gemma4/mechanism/gemma4_compliance_analysis_no_thinking/probe_results.json"),
    },
]

POSITION_LABELS = {
    "endorsement_start": "endorsement\nstart",
    "endorsed_answer": "endorsed\nanswer",
    "endorsement_end": "endorsement\nend",
    "answer_position": "answer\nposition",
    "endorsement_mean": "endorsement\nmean",
}


def _okabe_ito_seq_cmap() -> mcolors.LinearSegmentedColormap:
    # Okabe-Ito inspired: light neutral -> sky blue -> bluish green -> orange.
    colors = [
        "#F7F7F7",
        "#DDEAF4",
        "#9FD3F3",  # inspired by sky blue
        "#4DB6A6",  # inspired by bluish green
        "#E7B54A",  # inspired by orange
    ]
    return mcolors.LinearSegmentedColormap.from_list("okabe_ito_seq", colors, N=256)


def _load_model(cfg: Dict) -> Dict:
    sweep = json.loads(cfg["sweep_path"].read_text())
    probe = json.loads(cfg["probe_path"].read_text())
    return {
        "name": cfg["name"],
        "slug": cfg["slug"],
        "positions": sweep["positions"],
        "layers": sweep["layers"],
        "w1": np.array(sweep["w1_auroc_matrix"], dtype=float),
        "primary": probe["primary_pair"],
    }


def main() -> None:
    apply_axis_theme()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = [_load_model(cfg) for cfg in MODELS]

    cmap = _okabe_ito_seq_cmap()
    vmin, vmax = 0.30, 0.82

    fig = plt.figure(figsize=(10.7, 3.7))
    fig.patch.set_facecolor(AXIS.bg_canvas)
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1, 1, 1, 0.045],
        left=0.08, right=0.95, bottom=0.16, top=0.80, wspace=0.28,
    )

    for i, model in enumerate(models):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(AXIS.bg_card)
        soften_spines(ax)

        mat = model["w1"]
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_title(model["name"], fontsize=10.5, fontweight="bold", pad=10)
        ax.set_xticks(range(len(model["layers"])))
        ax.set_xticklabels(model["layers"], rotation=45, ha="right")
        ax.set_yticks(range(len(model["positions"])))

        if i == 0:
            ax.set_yticklabels([POSITION_LABELS.get(p, p) for p in model["positions"]], fontsize=8.4)
            ax.set_ylabel("Token position", fontsize=9.1, color=AXIS.ink_soft)
        else:
            ax.set_yticklabels([])

        ax.set_xlabel("Layer", fontsize=9.1, color=AXIS.ink_soft)

        primary_pos = model["primary"]["position"]
        primary_layer = int(model["primary"]["layer"])
        y = model["positions"].index(primary_pos)
        x = model["layers"].index(primary_layer)

        rect = mpatches.Rectangle(
            (x - 0.5, y - 0.5),
            1.0,
            1.0,
            fill=False,
            linewidth=2.2,
            edgecolor="#D55E00",
            zorder=5,
        )
        ax.add_patch(rect)
        ax.scatter(
            [x],
            [y],
            marker="*",
            s=115,
            c="#D55E00",
            edgecolors=AXIS.bg_card,
            linewidths=0.7,
            zorder=6,
        )

        ax.text(
            0.02,
            1.01,
            f"primary: L{primary_layer}, {primary_pos.replace('_', ' ')}",
            transform=ax.transAxes,
            fontsize=7.8,
            color=AXIS.ink_soft,
            ha="left",
            va="bottom",
        )

    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_edgecolor(AXIS.rule)
    cb.outline.set_linewidth(0.7)
    cb.set_label("W1 AUROC", fontsize=9.0, color=AXIS.ink_soft)
    cb.ax.tick_params(labelsize=8.2, color=AXIS.rule, labelcolor=AXIS.ink_soft, width=0.7, length=3)

    fig.text(
        0.08,
        0.95,
        "The authority-compliance signal localizes differently across models",
        fontsize=12.0,
        fontweight="bold",
        color=AXIS.ink,
        ha="left",
    )
    fig.text(
        0.08,
        0.905,
        "Marked cell = primary layer/position chosen by the compliance-direction analysis.",
        fontsize=8.6,
        color=AXIS.muted,
        ha="left",
    )

    save_fig(fig, "compliance_localization_heatmap", OUTPUT_DIR)
    print("Saved compliance_localization_heatmap.*")


if __name__ == "__main__":
    main()
