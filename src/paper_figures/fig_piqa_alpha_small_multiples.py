from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path("/Users/majortimberwolf/Projects/lossfunk/persona-vectors")
OUT_DIR = ROOT / "figures" / "neurips" / "v2" / "final"
OUT_PATH = OUT_DIR / "piqa_alpha_small_multiples.jpeg"
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
    summary_path: Path
    layers: List[int]
    rows_path: Path | None = None


SPECS: List[ModelSpec] = [
    ModelSpec(
        name="Qwen",
        summary_path=ROOT / "results/authority" / "qwen35" / "mechanism" / "piqa_forward_patch_200_freegen" / "piqa_summary.json",
        layers=[2, 5, 10],
    ),
    ModelSpec(
        name="GPT-OSS",
        summary_path=ROOT / "results/authority" / "gpt-oss" / "mechanism" / "piqa_forward_patch_200_freegen" / "piqa_summary.json",
        layers=[16, 18, 20],
    ),
    ModelSpec(
        name="Gemma",
        summary_path=ROOT / "results/authority" / "gemma4" / "mechanism" / "piqa_forward_patch_200_freegen_no_thinking" / "piqa_summary.json",
        layers=[20, 22, 24],
    ),
    ModelSpec(
        name="OLMo-2",
        summary_path=ROOT / "results/authority" / "olmo2" / "mechanism" / "piqa_forward_patch_200_freegen" / "piqa_summary.json",
        layers=[10, 16, 22],
    ),
    ModelSpec(
        name="OLMo-3.1",
        summary_path=ROOT / "results/authority" / "exp16" / "olmo31_piqa_n0_matched_flip_interpolate_l15_l18_l22" / "piqa_summary.json",
        rows_path=ROOT / "results/authority" / "exp16" / "olmo31_piqa_n0_matched_flip_interpolate_l15_l18_l22" / "piqa_rows.jsonl",
        layers=[15, 18, 22],
    ),
]


LAYER_COLORS = ["#5B8DB8", "#8B4A8B", "#E34A55"]
LAYER_MARKERS = ["o", "s", "D"]


def _load_summary(path: Path) -> List[dict]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _layer_from_row(row: dict) -> int:
    return int(row["target_layers"][0])


def _matched_flip_from_rows(spec: ModelSpec, layer: int, alpha: float) -> float:
    if spec.rows_path is None:
        return 0.0
    rows = [
        row
        for row in _load_jsonl(spec.rows_path)
        if row.get("condition_code") == "N0_note" and _layer_from_row(row) == layer
    ]
    baseline_correct = {
        row["uid"]
        for row in rows
        if float(row["alpha"]) == 0.0 and row.get("is_correct") is True
    }
    if not baseline_correct:
        return 0.0
    current = [
        row
        for row in rows
        if float(row["alpha"]) == alpha and row["uid"] in baseline_correct
    ]
    flips = sum(1 for row in current if row.get("chose_wrong") is True)
    return flips / len(baseline_correct)


def _series_for_layer(spec: ModelSpec, layer: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        row
        for row in _load_summary(spec.summary_path)
        if row["condition_code"] == "N0_note" and _layer_from_row(row) == layer
    ]
    rows.sort(key=lambda r: float(r["alpha"]))
    xs = np.array([float(r["alpha"]) for r in rows], dtype=float)
    ys = np.array(
        [
            100.0
            * (
                float(row["matched_flip_rate"])
                if row.get("matched_flip_rate") is not None
                else _matched_flip_from_rows(spec, layer, float(row["alpha"]))
            )
            for row in rows
        ],
        dtype=float,
    )
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

    fig, axes = plt.subplots(1, len(SPECS), figsize=(16.8, 3.1), sharey=True)
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
        ax.set_ylim(0, max(20.0, np.ceil(y_max / 5.0) * 5.0))
        ax.tick_params(axis="both", labelsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("black")
        ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=2.2)

    axes[0].set_ylabel("Matched PIQA flip rate (%)", fontsize=12)
    for ax in axes:
        ax.set_xlabel("Patch strength (α)", fontsize=12)

    plt.tight_layout(w_pad=1.0)
    fig.savefig(OUT_PATH, dpi=220, format="jpeg", bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
