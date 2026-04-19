"""Forward patching as small multiples in the editorial line-chart style.

Each panel is one model. Lines are the tested layers for that model.
The y-axis uses the matched flip rate among baseline-correct items.

Run:

    uv run python -m src.paper_figures.fig_forward_patch_small_multiples
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.axis_theme import AXIS, apply_axis_theme, save_fig, soften_spines


OUTPUT_DIR = Path("figures/neurips/v2")

DEPTH_COLORS = ["#5B2A86", "#2F9C95", "#D7A13B"]

MODELS: List[Dict] = [
    {
        "name": "GPT-oss-20B",
        "path": Path("neurips-results/gpt-oss/mechanism/forward_patch_test_256/steering_rows.jsonl"),
        "layers": (16, 18, 20),
    },
    {
        "name": "Gemma-4-26B",
        "path": Path(
            "neurips-results/gemma4/mechanism/forward_patch_test_256_no_thinking_l15_l18_clean_freegen/steering_rows.jsonl"
        ),
        "layers": (15, 18),
    },
    {
        "name": "OLMo-2-32B",
        "path": Path("neurips-results/olmo2/mechanism/forward_patch_test_256/steering_rows.jsonl"),
        "layers": (10, 16, 22),
    },
]


def _load_flip_rates(path: Path) -> Dict[int, List[Tuple[float, float]]]:
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
        baseline_rows = by_alpha.get(0.0, {})
        baseline_correct = {uid for uid, r in baseline_rows.items() if r.get("is_correct") is True}
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
        out[layer] = series
    return out


def _annotate_line_end(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, label: str, color: str, dy: float = 0.0) -> None:
    x = float(xs[-1])
    y = float(ys[-1])
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(7, dy),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.1,
        color=color,
        bbox=dict(facecolor=AXIS.bg_card, edgecolor="none", alpha=0.82, pad=0.8),
    )


def main() -> None:
    apply_axis_theme()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_series = {cfg["name"]: _load_flip_rates(cfg["path"]) for cfg in MODELS}
    y_max = 0.0
    for rates in all_series.values():
        for series in rates.values():
            for _, val in series:
                y_max = max(y_max, float(val))
    y_top = min(35.0, float(np.ceil(y_max / 5.0) * 5.0) + 2.0)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    fig.patch.set_facecolor(AXIS.bg_canvas)

    for ax, cfg in zip(axes, MODELS):
        ax.set_facecolor(AXIS.bg_card)
        soften_spines(ax)
        ax.axvline(0.0, color=AXIS.rule, linestyle=(0, (2, 3)), linewidth=0.8, zorder=1)
        ax.grid(True, axis="y", color=AXIS.hairline, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylim(0, y_top)

        layers = list(cfg["layers"])
        rates = all_series[cfg["name"]]
        depth_colors = DEPTH_COLORS[: len(layers)]
        dy_cycle = [0, 7, -7]

        for i, layer in enumerate(layers):
            if layer not in rates:
                continue
            series = rates[layer]
            xs = np.array([x for x, _ in series], dtype=float)
            ys = np.array([y for _, y in series], dtype=float)
            color = depth_colors[i]
            ax.plot(xs, ys, color=color, linewidth=1.55, marker="o", markersize=4.7)
            _annotate_line_end(ax, xs, ys, f"L{layer}", color, dy=dy_cycle[i % len(dy_cycle)])

        ax.set_title(cfg["name"], fontsize=10.2, fontweight="bold", pad=5)
        ax.set_xlabel("Patch magnitude (α)", fontsize=9.0, color=AXIS.ink_soft)

    axes[0].set_ylabel("Matched flip rate (%)", fontsize=9.4, color=AXIS.ink_soft)

    fig.text(
        0.12,
        0.96,
        "Forward patching flips baseline-correct answers to wrong",
        fontsize=12.5,
        fontweight="bold",
        color=AXIS.ink,
        ha="left",
    )

    save_fig(fig, "forward_patch_small_multiples", OUTPUT_DIR)
    print("Saved forward_patch_small_multiples.*")


if __name__ == "__main__":
    main()
