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
from typing import Dict, List, Optional, Tuple

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


RESULTS_JSON = Path("results/authority/shared_analysis/dynamic_parser_all_runs.json")
OUTPUT_DIR = Path("figures/neurips/v2/final")
FONT_DIR = Path("/Users/majortimberwolf/Library/Fonts")


def _wilson_ci_pct(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return 100.0 * max(0.0, centre - half), 100.0 * min(1.0, centre + half)


def _register_inter() -> None:
    for font_path in (
        FONT_DIR / "Inter-Regular.otf",
        FONT_DIR / "Inter-Medium.otf",
        FONT_DIR / "Inter-SemiBold.otf",
        FONT_DIR / "Inter-Bold.otf",
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))


LEVELS = ["none", "uncertain", "assertive", "authoritative"]
LEVEL_LABELS = [
    "No endorsement\n(baseline)",
    "Low authority\n\u201cone source\nsuggests X\u201d",
    "Assertive claim\n\u201cstrongly\nsuggested X\u201d",
    "Verified authority\n\u201cverified source\nsays X\u201d",
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
            "uncertain": "results/authority/qwen35/gradient_uncertain_shared_h100/dynamic_parser_recomputed_summary.json",
            "assertive": "results/authority/qwen35/gradient_assertive_shared_h100/dynamic_parser_recomputed_summary.json",
            "weak": "results/authority/qwen35/gradient_weak_shared_h100/dynamic_parser_recomputed_summary.json",
            "authoritative": "results/authority/qwen35/authoritative_verified_shared_h100/dynamic_parser_recomputed_summary.json",
        },
    },
    # OLMo-3.1: gradient {weak,uncertain,assertive} have dynamic_parser_recomputed
    # summaries; the authoritative_verified run does not, so we provide the
    # parsed-all-3 flip rate inline (recomputed locally from
    # `authoritative_verified_all_prior_wrong_shared_h100/..._dissociation_rows.jsonl`:
    # 121 W1 flips / 146 N0-correct UIDs = 82.88%).
    "OLMo-3.1-32B": {
        "color": "#B57FB1",
        "marker": "X",
        "summary_paths": {
            "uncertain": "results/authority/olmo31/gradient_uncertain_all_prior_wrong_shared_h100/dynamic_parser_recomputed_summary.json",
            "assertive": "results/authority/olmo31/gradient_assertive_all_prior_wrong_shared_h100/dynamic_parser_recomputed_summary.json",
            "weak": "results/authority/olmo31/gradient_weak_all_prior_wrong_shared_h100/dynamic_parser_recomputed_summary.json",
        },
        "inline_flip": {
            "authoritative": 82.88,
        },
        "inline_counts": {
            "authoritative": {"w1_flips": 121, "n0_correct": 146},
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


def _flip_ci_from_summary_path(path_str: str) -> Optional[Tuple[float, float]]:
    path = Path(path_str)
    if not path.exists():
        return None
    obj = json.loads(path.read_text())
    n = int(obj.get("parsed_all3_n0_correct") or obj.get("n0_correct") or 0)
    k = int(obj.get("w1_flips") or round(float(obj.get("flip_rate") or 0.0) * n))
    return _wilson_ci_pct(k, n) if n > 0 else None


def _extract_flip(runs: Dict[str, dict]) -> Dict[str, Dict[str, List]]:
    out: Dict[str, Dict[str, List]] = {}
    for name, cfg in MODEL_KEYS.items():
        flips: List[float | None] = [0.0]
        ci_los: List[float | None] = [None]
        ci_his: List[float | None] = [None]
        inline = cfg.get("inline_flip", {}) or {}
        inline_counts = cfg.get("inline_counts", {}) or {}
        for level in LEVELS[1:]:
            if level in inline:
                flips.append(float(inline[level]))
                counts = inline_counts.get(level, {})
                if counts:
                    lo, hi = _wilson_ci_pct(int(counts["w1_flips"]), int(counts["n0_correct"]))
                    ci_los.append(lo)
                    ci_his.append(hi)
                else:
                    ci_los.append(None)
                    ci_his.append(None)
                continue
            if "summary_paths" in cfg and level in cfg["summary_paths"]:
                flips.append(_flip_from_summary_path(cfg["summary_paths"][level]))
                ci = _flip_ci_from_summary_path(cfg["summary_paths"][level])
                ci_los.append(ci[0] if ci else None)
                ci_his.append(ci[1] if ci else None)
            elif "keys" in cfg and level in cfg["keys"]:
                run = runs.get(cfg["keys"][level]) or {}
                fr = run.get("flip_rate")
                flips.append(fr * 100.0 if fr is not None else None)
                n = int(run.get("n0_correct") or 0)
                k = int(run.get("w1_flips") or round(float(fr or 0.0) * n))
                if n > 0 and fr is not None:
                    lo, hi = _wilson_ci_pct(k, n)
                    ci_los.append(lo)
                    ci_his.append(hi)
                else:
                    ci_los.append(None)
                    ci_his.append(None)
            else:
                flips.append(None)
                ci_los.append(None)
                ci_his.append(None)
        out[name] = {
            "flip": flips,
            "ci_low": ci_los,
            "ci_high": ci_his,
            "color": cfg["color"],
            "marker": cfg["marker"],
        }
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
        ci_los = [row["ci_low"][i] for i in xs_valid]
        ci_his = [row["ci_high"][i] for i in xs_valid]
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
        yerr_lo = [
            (y - lo) if lo is not None else 0.0
            for y, lo in zip(ys_valid, ci_los)
        ]
        yerr_hi = [
            (hi - y) if hi is not None else 0.0
            for y, hi in zip(ys_valid, ci_his)
        ]
        ax.errorbar(
            xs_valid,
            ys_valid,
            yerr=[yerr_lo, yerr_hi],
            fmt="none",
            ecolor=row["color"],
            elinewidth=0.8,
            capsize=2.2,
            capthick=0.8,
            alpha=0.75,
            zorder=2.8,
        )
        if xs_valid and ys_valid[-1] is not None:
            y_offsets = {
                "GPT-oss-20B": -2,
                "Gemma-4-26B": -12,
                "OLMo-2-32B": 16,
                "Qwen-3.5-27B": 8,
                "OLMo-3.1-32B": -16,
            }
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
    ax.set_xlabel("Endorsement wording in prompt", fontsize=9.2, color=AXIS.ink_soft, labelpad=10)
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
        ncol=5, frameon=False,
        fontsize=9.1, handlelength=1.8, columnspacing=1.8,
    )

    paths = save_fig(fig, "authority_gradient_v2", OUTPUT_DIR)
    for path in paths:
        if path.suffix == ".svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print("Saved authority_gradient_v2.*")


if __name__ == "__main__":
    main()
