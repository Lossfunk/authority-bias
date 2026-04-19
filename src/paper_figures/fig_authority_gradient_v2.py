"""Fig 2: authority-framing gradient — flip rate only.

Single-panel view of the original ``authority_gradient`` figure, with the
five authority levels on the x-axis and the W1 flip rate on the y-axis
for three open-weights models.

Run::

    uv run python -m src.paper_figures.fig_authority_gradient_v2
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from src.paper_figures.axis_theme import (
    AXIS,
    annotate_endpoint,
    apply_axis_theme,
    line_with_markers,
    save_fig,
    soften_spines,
    y_grid,
)


RESULTS_JSON = Path("neurips-results/_shared/dynamic_parser_all_runs.json")
OUTPUT_DIR = Path("figures/neurips/v2/final")
FONT_DIR = Path("/Users/majortimberwolf/Library/Fonts")


def _register_inter() -> None:
    for font_path in (
        FONT_DIR / "Inter-Regular.otf",
        FONT_DIR / "Inter-Medium.otf",
        FONT_DIR / "Inter-SemiBold.otf",
        FONT_DIR / "Inter-Bold.otf",
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))


LEVELS = ["none", "uncertain", "assertive", "weak", "authoritative"]
LEVEL_LABELS = [
    "None\n(baseline)",
    "Uncertain\n\u201cone source\nsuggests\u201d",
    "Assertive\n\u201cstrongly\nsuggested\u201d",
    "Weak\n\u201cthe answer\nis X\u201d",
    "Authoritative\n\u201cverified\nsource\u201d",
]


MODEL_KEYS: Dict[str, Dict[str, str]] = {
    "GPT-oss-20B": {
        "color": "#C36662",
        "marker": "s",
        "keys": {
            "uncertain": "openai/gpt-oss-20b::gradient_gpt_oss_uncertain",
            "assertive": "openai/gpt-oss-20b::gradient_gpt_oss_assertive",
            "weak": "openai/gpt-oss-20b::gradient_gpt_oss_weak",
            "authoritative": "openai/gpt-oss-20b::v2_gpt_oss_freegen",
        },
    },
    "Gemma-4-26B": {
        "color": "#6F9C79",
        "marker": "D",
        "keys": {
            "uncertain": "google/gemma-4-26B-A4B-it::gradient_gemma4_uncertain",
            "assertive": "google/gemma-4-26B-A4B-it::gradient_gemma4_assertive",
            "weak": "google/gemma-4-26B-A4B-it::gradient_gemma4_weak",
            "authoritative": "google/gemma-4-26B-A4B-it::gemma4__no_thinking_freegen_merged_authoritative",
        },
    },
    "OLMo-2-32B": {
        "color": "#D5A048",
        "marker": "^",
        "keys": {
            "uncertain": "allenai/OLMo-2-0325-32B-Instruct::gradient_olmo2_uncertain",
            "assertive": "allenai/OLMo-2-0325-32B-Instruct::gradient_olmo2_assertive",
            "weak": "allenai/OLMo-2-0325-32B-Instruct::gradient_olmo2_weak",
            "authoritative": "allenai/OLMo-2-0325-32B-Instruct::v2_olmo2_freegen",
        },
    },
    "Qwen-3.5-27B": {
        "color": "#5B84B5",
        "marker": "P",
        "summary_paths": {
            "uncertain": "neurips-results/qwen35/gradient_uncertain_shared_h100/dynamic_parser_recomputed_summary.json",
            "assertive": "neurips-results/qwen35/gradient_assertive_shared_h100/dynamic_parser_recomputed_summary.json",
            "weak": "neurips-results/qwen35/gradient_weak_shared_h100/dynamic_parser_recomputed_summary.json",
            "authoritative": "neurips-results/qwen35/authoritative_verified_shared_h100/dynamic_parser_recomputed_summary.json",
        },
    },
}


def _load_runs() -> Dict[str, dict]:
    return json.loads(RESULTS_JSON.read_text())


def _flip_from_summary_path(path_str: str) -> Optional[float]:
    path = Path(path_str)
    if not path.exists():
        return None
    obj = json.loads(path.read_text())
    fr = obj.get("flip_rate")
    return fr * 100.0 if fr is not None else None


def _extract_flip(runs: Dict[str, dict]) -> Dict[str, Dict[str, List]]:
    out: Dict[str, Dict[str, List]] = {}
    for name, cfg in MODEL_KEYS.items():
        flips: List[float | None] = [0.0]
        for level in LEVELS[1:]:
            if "summary_paths" in cfg:
                flips.append(_flip_from_summary_path(cfg["summary_paths"].get(level, "")))
            else:
                run = runs.get(cfg["keys"].get(level, "")) or {}
                fr = run.get("flip_rate")
                flips.append(fr * 100.0 if fr is not None else None)
        out[name] = {"flip": flips, "color": cfg["color"], "marker": cfg["marker"]}
    return out


def main() -> None:
    _register_inter()
    apply_axis_theme()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = _load_runs()
    data = _extract_flip(runs)

    fig = plt.figure(figsize=(9.3, 4.9))
    fig.patch.set_facecolor(AXIS.bg_canvas)
    ax = fig.add_axes([0.10, 0.22, 0.80, 0.62])
    ax.set_facecolor(AXIS.bg_card)

    x = np.arange(len(LEVELS))
    ax.set_xlim(-0.35, len(LEVELS) - 0.55)
    ax.set_ylim(-2, 100)
    for name, row in data.items():
        ys = row["flip"]
        xs_valid = [i for i, v in enumerate(ys) if v is not None]
        ys_valid = [ys[i] for i in xs_valid]
        line_with_markers(
            ax,
            xs_valid, ys_valid,
            color=row["color"],
            marker=row["marker"],
            label=name,
            linewidth=2.3,
            markersize=7.5,
            zorder=3,
        )
        if xs_valid and ys_valid[-1] is not None:
            y_offsets = {"GPT-oss-20B": -2, "Gemma-4-26B": -12, "OLMo-2-32B": 12, "Qwen-3.5-27B": 6}
            annotate_endpoint(
                ax,
                xs_valid[-1], ys_valid[-1],
                f"{ys_valid[-1]:.0f}%",
                color=row["color"],
                dx=9,
                dy=y_offsets.get(name, 0),
                fontsize=8.8,
            )

    ax.axhline(0, color=AXIS.rule, linewidth=0.7, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_LABELS, fontsize=8.3, color=AXIS.ink)
    ax.set_ylabel("W1 flip rate (%)", fontsize=9.6, color=AXIS.ink_soft)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    y_grid(ax, alpha=0.55)
    soften_spines(ax)
    ax.tick_params(axis="x", length=0, pad=6)

    # Legend above the plot
    handles = [
        plt.Line2D(
            [0], [0],
            color=data[name]["color"],
            marker=data[name]["marker"],
            markersize=7.5,
            linewidth=2.3,
            markeredgecolor=AXIS.bg_card,
            markeredgewidth=0.9,
        )
        for name in data
    ]
    fig.legend(
        handles, list(data.keys()),
        loc="center", bbox_to_anchor=(0.5, 0.90),
        ncol=4, frameon=False,
        fontsize=9.1, handlelength=1.8, columnspacing=1.8,
    )

    fig.text(
        0.10, 0.945,
        "Stronger endorsements produce stronger overrides",
        fontsize=13.0, fontweight="bold", color=AXIS.ink, ha="left", va="bottom",
    )

    save_fig(fig, "authority_gradient_v2", OUTPUT_DIR)
    print("Saved authority_gradient_v2.*")


if __name__ == "__main__":
    main()
