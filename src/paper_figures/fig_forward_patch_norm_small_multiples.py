from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.exp16.run_steering_test import _load_extraction_store, _load_masks


ROOT = Path("/Users/majortimberwolf/Projects/lossfunk/persona-vectors")
OUT_DIR = ROOT / "figures" / "neurips" / "v2" / "candidates"
OUT_PATH = OUT_DIR / "forward_patch_norm_small_multiples.jpeg"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    summary_path: Path
    extraction_dir: Path
    masks_path: Path
    layers: List[int]
    direction_position: str
    patch_source_position: str


SPECS: List[ModelSpec] = [
    ModelSpec(
        name="GPT-OSS",
        summary_path=ROOT / "results/authority" / "gpt-oss" / "mechanism" / "forward_patch_test_256" / "steering_summary.json",
        extraction_dir=ROOT / "results/authority" / "gpt-oss" / "mechanism" / "gpt_oss_authority_activations",
        masks_path=ROOT / "results/authority" / "gpt-oss" / "mechanism" / "gpt_oss_authority_activations" / "label_masks.json",
        layers=[16, 18, 20],
        direction_position="endorsement_end",
        patch_source_position="endorsement_end",
    ),
    ModelSpec(
        name="Gemma",
        summary_path=ROOT / "results/authority" / "gemma4" / "mechanism" / "forward_patch_test_256_no_thinking_l15_l18_clean_freegen" / "steering_summary.json",
        extraction_dir=ROOT / "results/authority" / "gemma4" / "mechanism" / "gemma4_authority_activations_no_thinking",
        masks_path=ROOT / "results/authority" / "gemma4" / "mechanism" / "gemma4_authority_activations_no_thinking" / "label_masks.json",
        layers=[15, 18],
        direction_position="endorsement_mean",
        patch_source_position="endorsement_mean",
    ),
    ModelSpec(
        name="OLMo",
        summary_path=ROOT / "results/authority" / "olmo2" / "mechanism" / "forward_patch_test_256" / "steering_summary.json",
        extraction_dir=ROOT / "results/authority" / "olmo2" / "mechanism" / "olmo2_authority_activations",
        masks_path=ROOT / "results/authority" / "olmo2" / "mechanism" / "olmo2_authority_activations" / "label_masks.json",
        layers=[10, 16, 22],
        direction_position="answer_position",
        patch_source_position="answer_position",
    ),
]


LAYER_COLORS = ["#5B8DB8", "#8B4A8B", "#E34A55"]
LAYER_MARKERS = ["o", "s", "D"]


def _load_summary(path: Path) -> List[dict]:
    return json.loads(path.read_text())


def _avg_hidden_norm(spec: ModelSpec, layer: int) -> float:
    store = _load_extraction_store(spec.extraction_dir)
    masks = _load_masks(spec.masks_path)
    uids = list(masks["mask_primary_w1"])
    mat = store.matrix(
        position=spec.patch_source_position,
        layer_value=layer,
        uids=uids,
        style="authoritative_verified",
        condition_code="N0_note",
    )
    if mat.shape[0] == 0:
        raise ValueError(f"No baseline activations for {spec.name} layer {layer}")
    return float(mat.norm(dim=1).mean().item())


def _avg_patch_delta_norm(spec: ModelSpec, layer: int) -> float:
    store = _load_extraction_store(spec.extraction_dir)
    masks = _load_masks(spec.masks_path)
    uids = list(masks["mask_primary_w1"])
    base = store.matrix(
        position=spec.patch_source_position,
        layer_value=layer,
        uids=uids,
        style="authoritative_verified",
        condition_code="N0_note",
    )
    patch_mean = store.mean(
        position=spec.patch_source_position,
        layer_value=layer,
        uids=uids,
        style="authoritative_verified",
        condition_code="W1_note",
    )
    if base.shape[0] == 0 or patch_mean is None:
        raise ValueError(f"No patch activations for {spec.name} layer {layer}")
    delta = patch_mean.unsqueeze(0) - base
    return float(delta.norm(dim=1).mean().item())


def _series_for_layer(spec: ModelSpec, layer: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        row
        for row in _load_summary(spec.summary_path)
        if row["condition"] == "N0_note" and int(row["config_id"].split("_L")[1].split("_")[0]) == layer
    ]
    rows.sort(key=lambda r: float(r["alpha"]))
    avg_norm = _avg_hidden_norm(spec, layer)
    avg_delta_norm = _avg_patch_delta_norm(spec, layer)
    xs = np.array([float(r["alpha"]) * avg_delta_norm / avg_norm for r in rows], dtype=float)
    ys = np.array(
        [
            100.0 * float(r["matched_flip_rate"] or 0.0)
            for r in rows
        ],
        dtype=float,
    )
    return xs, ys


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), sharey=True)
    fig.patch.set_facecolor("white")

    y_max = 0.0
    plotted = []
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
        panel_x_max = max(
            float(x.max())
            for name, lyr, x, _ in plotted
            if name == spec.name and lyr in spec.layers
        )
        ax.set_xlim(-0.002, panel_x_max + max(0.002, 0.12 * panel_x_max))
        ax.set_ylim(0, max(25.0, np.ceil(y_max / 5.0) * 5.0))
        ax.tick_params(axis="both", labelsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("black")
        ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=2.2)

    axes[0].set_ylabel("Matched flip rate (%)", fontsize=12)
    for ax in axes:
        ax.set_xlabel("Fraction of Avg. Norm", fontsize=12)

    plt.tight_layout(w_pad=1.0)
    fig.savefig(OUT_PATH, dpi=220, format="jpeg", bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
