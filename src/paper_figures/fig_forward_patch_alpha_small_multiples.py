from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path("/Users/majortimberwolf/Projects/lossfunk/persona-vectors")
OUT_DIR = ROOT / "figures" / "neurips" / "v2" / "final"
OUT_PATH = OUT_DIR / "forward_patch_alpha_small_multiples.jpeg"
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


@dataclass(frozen=True)
class ModelSpec:
    name: str
    layers: List[int]
    summary_path: Path | None = None
    layer_paths: Dict[int, Path] | None = None


SPECS: List[ModelSpec] = [
    ModelSpec(
        name="Qwen",
        layers=[2, 5, 10],
        summary_path=ROOT / "neurips-results" / "qwen35" / "mechanism" / "forward_patch_expanded_512_w1source" / "steering_summary.json",
    ),
    ModelSpec(
        name="GPT-OSS",
        layers=[16, 18, 20],
        summary_path=ROOT / "neurips-results" / "gpt-oss" / "mechanism" / "forward_patch_test_256" / "steering_summary.json",
    ),
    ModelSpec(
        name="Gemma",
        layers=[15, 18, 20],
        layer_paths={
            15: ROOT / "neurips-results" / "gemma4" / "mechanism" / "forward_patch_test_256_no_thinking_l15_l18_clean_freegen" / "steering_summary.json",
            18: ROOT / "neurips-results" / "gemma4" / "mechanism" / "forward_patch_test_256_no_thinking_l15_l18_clean_freegen" / "steering_summary.json",
            20: ROOT / "neurips-results" / "gemma4" / "mechanism" / "forward_patch_test_256_no_thinking" / "steering_summary.json",
        },
    ),
    ModelSpec(
        name="OLMo",
        layers=[10, 16, 22],
        summary_path=ROOT / "neurips-results" / "olmo2" / "mechanism" / "forward_patch_test_256" / "steering_summary.json",
    ),
]


LAYER_COLORS = ["#5B8DB8", "#8B4A8B", "#E34A55"]
LAYER_MARKERS = ["o", "s", "D"]


def _load_summary(path: Path) -> List[dict]:
    return json.loads(path.read_text())


def _layer_from_config(config_id: str) -> int:
    return int(config_id.split("_L")[1].split("_")[0])


def _series_for_layer(spec: ModelSpec, layer: int) -> tuple[np.ndarray, np.ndarray]:
    path = spec.layer_paths.get(layer) if spec.layer_paths else spec.summary_path
    rows = [
        row
        for row in _load_summary(path)
        if row["condition"] == "N0_note" and _layer_from_config(row["config_id"]) == layer
    ]
    rows.sort(key=lambda r: float(r["alpha"]))
    xs = np.array([float(r["alpha"]) for r in rows], dtype=float)
    ys = np.array([100.0 * float(r["matched_flip_rate"] or 0.0) for r in rows], dtype=float)
    return xs, ys


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _register_inter()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "Inter Regular",
                "Inter Medium",
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.1), sharey=True)
    fig.patch.set_facecolor("white")

    plotted = []
    y_max = 0.0
    for spec in SPECS:
        for layer in spec.layers:
            xs, ys = _series_for_layer(spec, layer)
            plotted.append((spec.name, layer, xs, ys))
            y_max = max(y_max, float(ys.max()))

    for ax, spec in zip(axes, SPECS):
        ax.set_title(spec.name, fontsize=17, pad=4)
        ax.axvline(0.0, color="#c7c7c7", linestyle="--", linewidth=0.9, zorder=0)
        ax.set_facecolor("white")
        ax.grid(False)

        for i, layer in enumerate(spec.layers):
            xs, ys = next((x, y) for name, lyr, x, y in plotted if name == spec.name and lyr == layer)
            ax.plot(
                xs,
                ys,
                color=LAYER_COLORS[i],
                marker=LAYER_MARKERS[i],
                markersize=4.5,
                linewidth=1.0,
                label=f"L{layer}",
            )

        ax.set_xlim(-0.02, 1.04)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylim(0, max(25.0, np.ceil(y_max / 5.0) * 5.0))
        ax.tick_params(axis="both", labelsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("black")
        ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=2.2)

    axes[0].set_ylabel("Matched flip rate (%)", fontsize=12)
    for ax in axes:
        ax.set_xlabel("Patch strength (α)", fontsize=12)

    plt.tight_layout(w_pad=1.0)
    fig.savefig(OUT_PATH, dpi=220, format="jpeg", bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
