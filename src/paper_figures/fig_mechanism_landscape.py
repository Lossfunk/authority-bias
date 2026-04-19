"""Nine.png-style mechanism landscape.

This panel is intentionally styled as a dense intervention landscape rather
than a narrated summary graphic. Each marker is one matched `(layer, alpha)`
forward-patch setting for an open-weight model.

Encoding:
* x-axis: within-domain wrong-rate shift
* y-axis: PIQA wrong-rate shift
* marker shape: model family
* marker size: patch alpha
* marker color: relative patch depth (early -> late)

Run:

    uv run python -m src.paper_figures.fig_mechanism_landscape
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.axis_theme import AXIS, apply_axis_theme, save_fig, soften_spines


RUNS_JSON = Path("neurips-results/_shared/dynamic_parser_all_runs.json")
OUTPUT_DIR = Path("figures/neurips/v2/candidates")

MODELS: List[Dict] = [
    {
        "slug": "gpt-oss-20b",
        "name": "GPT-oss-20B",
        "runs_key": "openai/gpt-oss-20b::v2_gpt_oss_freegen",
        "patch_path": Path("neurips-results/gpt-oss/mechanism/forward_patch_test_256/steering_rows.jsonl"),
        "piqa_path": Path("neurips-results/gpt-oss/mechanism/piqa_forward_patch_200_freegen/piqa_summary.json"),
        "marker": "s",
        "edge": "#C36662",
        "label_offset": (-10, 8),
    },
    {
        "slug": "gemma4",
        "name": "Gemma-4-26B",
        "runs_key": "google/gemma-4-26B-A4B-it::gemma4__no_thinking_freegen_merged_authoritative",
        "patch_path": Path("neurips-results/gemma4/mechanism/forward_patch_test_256_no_thinking/steering_rows.jsonl"),
        "piqa_path": Path("neurips-results/gemma4/mechanism/piqa_forward_patch_200_freegen_no_thinking/piqa_summary.json"),
        "marker": "D",
        "edge": "#6F9C79",
        "label_offset": (10, -10),
    },
    {
        "slug": "olmo2",
        "name": "OLMo-2-32B",
        "runs_key": "allenai/OLMo-2-0325-32B-Instruct::v2_olmo2_freegen",
        "patch_path": Path("neurips-results/olmo2/mechanism/forward_patch_test_256/steering_rows.jsonl"),
        "piqa_path": Path("neurips-results/olmo2/mechanism/piqa_forward_patch_200_freegen/piqa_summary.json"),
        "marker": "^",
        "edge": "#D5A048",
        "label_offset": (-10, 8),
    },
]

ALPHA_SIZES = {
    0.3: 34,
    0.5: 62,
    0.7: 92,
    1.0: 134,
}


LABEL_SPECS = {
    ("gpt-oss-20b", 16, 1.0): {"dx": -10, "dy": -2, "ha": "right"},
    ("olmo2", 16, 0.5): {"dx": -10, "dy": 2, "ha": "right"},
    ("gemma4", 24, 1.0): {"dx": 10, "dy": 4, "ha": "left"},
    ("gemma4", 22, 0.7): {"dx": -2, "dy": -10, "ha": "left"},
}


def _load_behavioral_flip_rates() -> Dict[str, float]:
    runs = json.loads(RUNS_JSON.read_text())
    out: Dict[str, float] = {}
    for cfg in MODELS:
        out[cfg["slug"]] = 100.0 * float(runs[cfg["runs_key"]]["flip_rate"])
    return out


def _load_patch_rates(path: Path) -> Dict[Tuple[int, float], float]:
    grouped = defaultdict(lambda: defaultdict(dict))
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("condition_code") != "N0_note":
            continue
        layer = int(row["target_layers"][0])
        alpha = float(row["alpha"])
        grouped[layer][alpha][row["uid"]] = row

    out: Dict[Tuple[int, float], float] = {}
    for layer, by_alpha in grouped.items():
        base = by_alpha.get(0.0, {})
        baseline_correct = {uid for uid, r in base.items() if r.get("is_correct") is True}
        if not baseline_correct:
            continue
        for alpha, rows in by_alpha.items():
            if alpha == 0.0:
                continue
            flips = sum(
                1
                for uid in baseline_correct
                if rows.get(uid, {}).get("chose_wrong") is True
            )
            out[(layer, alpha)] = 100.0 * flips / len(baseline_correct)
    return out


def _load_within_domain_deltas(path: Path) -> Dict[Tuple[int, float], float]:
    grouped = defaultdict(lambda: defaultdict(list))
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("condition_code") != "N0_note":
            continue
        layer = int(row["target_layers"][0])
        alpha = float(row["alpha"])
        grouped[layer][alpha].append(row)

    def wrong_rate(rows: List[Dict]) -> float | None:
        parsed = [r for r in rows if r.get("parsed_label") is not None]
        if not parsed:
            return None
        wrong = sum(1 for r in parsed if r.get("chose_wrong") is True)
        return 100.0 * wrong / len(parsed)

    out: Dict[Tuple[int, float], float] = {}
    for layer, by_alpha in grouped.items():
        baseline = wrong_rate(by_alpha.get(0.0, []))
        if baseline is None:
            continue
        for alpha, rows in by_alpha.items():
            if alpha == 0.0:
                continue
            current = wrong_rate(rows)
            if current is None:
                continue
            out[(layer, alpha)] = current - baseline
    return out


def _load_piqa_deltas(path: Path) -> Dict[Tuple[int, float], float]:
    rows = [r for r in json.loads(path.read_text()) if r.get("condition_code") == "N0_note"]
    by_layer = defaultdict(list)
    for row in rows:
        by_layer[int(row["target_layers"][0])].append(row)

    out: Dict[Tuple[int, float], float] = {}
    for layer, layer_rows in by_layer.items():
        layer_rows.sort(key=lambda r: float(r["alpha"]))
        base = next(
            (100.0 * float(r["wrong_rate_parsed"]) for r in layer_rows if float(r["alpha"]) == 0.0),
            None,
        )
        if base is None:
            continue
        for row in layer_rows:
            alpha = float(row["alpha"])
            if alpha == 0.0:
                continue
            out[(layer, alpha)] = 100.0 * float(row["wrong_rate_parsed"]) - base
    return out


def _collect_points() -> Tuple[List[Dict], List[Dict]]:
    behavior = _load_behavioral_flip_rates()
    points: List[Dict] = []
    highlights: List[Dict] = []

    for cfg in MODELS:
        patch = _load_patch_rates(cfg["patch_path"])
        domain_delta = _load_within_domain_deltas(cfg["patch_path"])
        piqa = _load_piqa_deltas(cfg["piqa_path"])
        matched = sorted(set(patch) & set(piqa) & set(domain_delta))
        layers = sorted({layer for layer, _ in matched})
        if not matched or not layers:
            continue

        if len(layers) == 1:
            depth_map = {layers[0]: 0.5}
        else:
            depth_map = {
                layer: i / (len(layers) - 1)
                for i, layer in enumerate(layers)
            }

        best_row = None
        best_score = -1e9
        for layer, alpha in matched:
            recovery = 100.0 * patch[(layer, alpha)] / max(behavior[cfg["slug"]], 1e-6)
            transfer = piqa[(layer, alpha)]
            within_domain = domain_delta[(layer, alpha)]
            row = {
                **cfg,
                "layer": layer,
                "alpha": alpha,
                "domain_delta": within_domain,
                "recovery": recovery,
                "transfer": transfer,
                "depth": depth_map[layer],
                "behavior": behavior[cfg["slug"]],
            }
            points.append(row)

            # Prefer points that are both recoverable and portable.
            score = transfer + 0.7 * within_domain
            if score > best_score:
                best_score = score
                best_row = row

        if best_row is not None:
            highlights.append(best_row)

    return points, highlights


def _label_text(row: Dict) -> str:
    return f"{row['name']} · L{row['layer']}, α={row['alpha']:.1f}"


def main() -> None:
    apply_axis_theme()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    points, _ = _collect_points()

    fig = plt.figure(figsize=(8.2, 5.65))
    fig.patch.set_facecolor(AXIS.bg_canvas)
    ax = fig.add_axes([0.09, 0.11, 0.75, 0.81])
    cax = fig.add_axes([0.875, 0.14, 0.03, 0.78])

    ax.set_facecolor(AXIS.bg_card)
    domain_values = [row["domain_delta"] for row in points]
    xmin = min(domain_values) - 0.8
    xmax = max(domain_values) + 0.8
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-2.2, 11.2)
    xticks = [-2, 0, 2, 4, 6, 8, 10, 12, 14, 16]
    xticks = [tick for tick in xticks if xmin <= tick <= xmax]
    ax.set_xticks(xticks)
    ax.set_yticks([0, 2, 4, 6, 8, 10])

    ax.axvline(0, color=AXIS.rule, linestyle=(0, (1.1, 2.1)), linewidth=0.85, zorder=1)
    ax.axhline(0, color=AXIS.rule, linestyle=(0, (1.1, 2.1)), linewidth=0.85, zorder=1)
    ax.set_axisbelow(True)
    soften_spines(ax)

    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    for row in points:
        ax.scatter(
            row["domain_delta"],
            row["transfer"],
            s=ALPHA_SIZES[row["alpha"]],
            marker=row["marker"],
            facecolor=cmap(norm(row["depth"])),
            edgecolor=AXIS.bg_card,
            linewidth=0.7,
            alpha=0.96,
            zorder=4,
        )

    for row in points:
        key = (row["slug"], row["layer"], row["alpha"])
        if key not in LABEL_SPECS:
            continue
        spec = LABEL_SPECS[key]
        ax.annotate(
            _label_text(row),
            xy=(row["domain_delta"], row["transfer"]),
            xytext=(spec["dx"], spec["dy"]),
            textcoords="offset points",
            fontsize=8.1,
            color=AXIS.ink_soft,
            ha=spec["ha"],
            va="center",
        )

    ax.set_xlabel("Within-domain wrong-rate shift (pp)", fontsize=10.0, color=AXIS.ink_soft)
    ax.set_ylabel("PIQA wrong-rate shift (pp)", fontsize=10.0, color=AXIS.ink_soft)

    depth_cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    depth_cbar.outline.set_edgecolor(AXIS.rule)
    depth_cbar.outline.set_linewidth(0.7)
    depth_cbar.set_ticks([0.0, 0.5, 1.0])
    depth_cbar.set_ticklabels(["early", "mid", "late"])
    depth_cbar.ax.tick_params(labelsize=8.0, color=AXIS.rule, labelcolor=AXIS.ink_soft, width=0.7, length=3)
    depth_cbar.set_label("Relative patch depth", fontsize=8.8, color=AXIS.ink_soft)

    # Model-shape legend.
    model_handles = []
    for cfg in MODELS:
        handle = plt.Line2D(
            [],
            [],
            linestyle="None",
            marker=cfg["marker"],
            markersize=7.0,
            markerfacecolor="#A9B0B8",
            markeredgecolor="#A9B0B8",
            markeredgewidth=0.8,
            label=cfg["name"],
        )
        model_handles.append(handle)
    leg_models = ax.legend(
        handles=model_handles,
        title="Model",
        title_fontsize=8.4,
        fontsize=8.0,
        loc="lower right",
        bbox_to_anchor=(1.005, 0.17),
        frameon=False,
        handlelength=1.0,
        labelspacing=0.4,
    )
    leg_models.get_title().set_color(AXIS.ink_soft)
    for text in leg_models.get_texts():
        text.set_color(AXIS.ink_soft)
    ax.add_artist(leg_models)

    alpha_handles = []
    for alpha, size in ALPHA_SIZES.items():
        handle = plt.Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=np.sqrt(size) / 1.8,
            markerfacecolor="#A9B0B8",
            markeredgecolor="#A9B0B8",
            markeredgewidth=0.8,
            label=f"{alpha:.1f}",
        )
        alpha_handles.append(handle)
    leg_alpha = ax.legend(
        handles=alpha_handles,
        title="Patch α",
        title_fontsize=8.4,
        fontsize=8.0,
        loc="lower right",
        bbox_to_anchor=(1.005, 0.01),
        frameon=False,
        handlelength=1.0,
        labelspacing=0.45,
    )
    leg_alpha.get_title().set_color(AXIS.ink_soft)
    for text in leg_alpha.get_texts():
        text.set_color(AXIS.ink_soft)

    save_fig(fig, "mechanism_landscape", OUTPUT_DIR)
    print("Saved mechanism_landscape.*")


if __name__ == "__main__":
    main()
