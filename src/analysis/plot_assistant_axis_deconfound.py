#!/usr/bin/env python3
"""Single-panel Assistant-axis deconfound figure (paper-Fig-4 aesthetic).

Pooled role-vector histogram along the Assistant Axis with smooth red -> blue
gradient, one authority marker per model, and no title / no legend box.

Run:
  uv run python -m src.analysis.plot_assistant_axis_deconfound
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch


OUTPUT_DIR = Path("figures/neurips/v2/candidates")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    overlap_path: Path
    marker: str
    marker_color: str
    marker_size: float


MODELS: List[ModelSpec] = [
    ModelSpec(
        key="gpt-oss",
        display="GPT-OSS",
        overlap_path=Path("external/assistant-axis/hardened/overlap_summary.json"),
        marker="*",
        marker_color="#F2B233",
        marker_size=360,
    ),
    ModelSpec(
        key="gemma4",
        display="Gemma-4",
        overlap_path=Path(
            "results/authority/gemma4/mechanism/assistant_axis_hardened/overlap_summary.json"
        ),
        marker="D",
        marker_color="#E67E22",
        marker_size=140,
    ),
    ModelSpec(
        key="olmo2",
        display="OLMo-2",
        overlap_path=Path(
            "results/authority/olmo2/mechanism/assistant_axis_hardened/overlap_summary.json"
        ),
        marker="^",
        marker_color="#2E9E5E",
        marker_size=180,
    ),
]

AUTHORITY_KEY = "authority_given_correct"


# ---------------------------------------------------------------------------
# Styling -- smooth red -> grey -> blue like the paper
# ---------------------------------------------------------------------------

ROLE_CMAP = LinearSegmentedColormap.from_list(
    "role_axis_rb",
    [
        (0.00, "#D86A66"),
        (0.22, "#E08F8A"),
        (0.42, "#D9B9B6"),
        (0.55, "#BFBFC4"),
        (0.70, "#9CB1CC"),
        (0.88, "#6E8FC0"),
        (1.00, "#4A75B8"),
    ],
)

COLOR_AXIS_LINE = "#111111"
COLOR_ROLE_TAG = "#333333"
COLOR_ROLE_CONNECT = "#b8b8b8"
COLOR_RP = "#C95C57"
COLOR_AS = "#4A75B8"


plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.labelsize": 11.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "axes.linewidth": 1.0,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class ModelData:
    spec: ModelSpec
    layer: int
    role_cos: Dict[str, float]
    default_cos: float
    authority_cos: float


def load_model(spec: ModelSpec) -> ModelData:
    d = json.loads(spec.overlap_path.read_text())
    return ModelData(
        spec=spec,
        layer=int(d["layer"]),
        role_cos={r: float(v) for r, v in d["cos_axis_with_roles"].items()},
        default_cos=float(d["cos_axis_with_default"]),
        authority_cos=float(d["directions"][AUTHORITY_KEY]["cos_with_assistant_axis"]),
    )


def _pooled(models: List[ModelData]) -> Tuple[np.ndarray, List[Tuple[str, str, float]]]:
    pooled: List[float] = []
    tagged: List[Tuple[str, str, float]] = []
    for m in models:
        for role, c in m.role_cos.items():
            pooled.append(c)
            tagged.append((m.spec.key, role, c))
    return np.asarray(pooled), tagged


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _gaussian_kde(values: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    """Simple Gaussian kernel density (vectorised)."""
    if values.size == 0:
        return np.zeros_like(grid)
    diffs = (grid[:, None] - values[None, :]) / bw
    kernel = np.exp(-0.5 * diffs ** 2) / np.sqrt(2 * np.pi)
    return kernel.sum(axis=1) / (values.size * bw)


def _gradient_histogram(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    xlim: Tuple[float, float],
    height: float,
    bw: float | None = None,
    n_grid: int = 600,
) -> None:
    """Smooth KDE-style shaded curve coloured with a left-to-right gradient
    that fades vertically to transparent -- replaces the blocky histogram."""

    if values.size == 0:
        return

    lo, hi = xlim
    span = hi - lo if hi > lo else 1.0
    if bw is None:
        std = float(values.std(ddof=1)) if values.size > 1 else span / 20.0
        # Silverman-ish, but nudged smoother for visual cleanliness.
        bw = max(1.06 * std * values.size ** (-1 / 5), span / 60.0)

    grid = np.linspace(lo, hi, n_grid)
    density = _gaussian_kde(values, grid, bw)
    peak = density.max()
    if peak <= 0:
        return
    curve = density / peak * height

    # Draw a horizontal gradient image clipped to the KDE curve silhouette.
    n_rows, n_cols = 200, n_grid
    img = np.zeros((n_rows, n_cols, 4))
    # Colour per column from the cmap
    col_t = (grid - lo) / span
    col_rgba = ROLE_CMAP(np.clip(col_t, 0, 1))  # (n_cols, 4)
    img[..., :3] = col_rgba[None, :, :3]

    # Vertical fade: opaque at bottom, transparent at top, but only up to each
    # column's curve height.
    ys = np.linspace(0.0, height, n_rows)[:, None]           # (n_rows, 1)
    base_alpha = np.linspace(0.78, 0.05, n_rows)[:, None]    # (n_rows, 1)
    mask = (ys <= curve[None, :]).astype(float)              # (n_rows, n_cols)
    img[..., 3] = base_alpha * mask * col_rgba[None, :, 3]

    ax.imshow(
        img,
        aspect="auto",
        interpolation="bilinear",
        extent=(lo, hi, 0.0, height),
        origin="lower",
        zorder=1,
    )

    # Thin colour-matched outline along the curve for definition.
    ax.plot(
        grid, curve, color="#6b6b6b", lw=0.6, alpha=0.35, zorder=2,
    )


def _role_dots(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    xlim: Tuple[float, float],
    y: float = 0.0,
    size: float = 18,
) -> None:
    lo, hi = xlim
    span = hi - lo if hi > lo else 1.0
    colors = [ROLE_CMAP(np.clip((v - lo) / span, 0.0, 1.0)) for v in values]
    ax.scatter(
        values, np.full_like(values, y, dtype=float),
        c=colors, s=size, edgecolor="none",
        alpha=0.90, zorder=3,
    )


def _label_extremes(
    ax: plt.Axes,
    tagged: List[Tuple[str, str, float]],
    *,
    n_left: int = 3,
    n_right: int = 3,
    font_size: float = 9.4,
) -> None:
    """Paper-Fig-4 style: three extreme roles *above* on one side and three
    *below* on the other side, nicely staggered so labels don't touch each
    other regardless of how tightly x-clustered they are."""

    sorted_items = sorted(tagged, key=lambda t: t[2])
    left = sorted_items[:n_left]                     # most negative first
    right = list(reversed(sorted_items[-n_right:]))  # most positive first

    def draw(items, *, above: bool) -> None:
        y_rows = ([2.55, 3.05, 3.55] if above
                  else [-0.55, -1.05, -1.55])
        va = "bottom" if above else "top"
        for idx, (_k, role, value) in enumerate(items):
            y = y_rows[idx % len(y_rows)]
            ax.annotate(
                role.capitalize(),
                xy=(value, 0.0),
                xytext=(value, y),
                textcoords="data",
                ha="center", va=va,
                fontsize=font_size,
                color=COLOR_ROLE_TAG,
                arrowprops=dict(
                    arrowstyle="-",
                    color=COLOR_ROLE_CONNECT,
                    lw=0.55, shrinkA=0, shrinkB=2,
                ),
            )

    draw(left, above=True)
    draw(right, above=True)


def _side_arrows(ax: plt.Axes, xlim: Tuple[float, float], y: float) -> None:
    lo, hi = xlim
    span = hi - lo
    # Left
    lt_x0 = lo + 0.28 * span
    lt_x1 = lo + 0.08 * span
    ax.add_patch(FancyArrowPatch(
        (lt_x0, y), (lt_x1, y),
        arrowstyle="-|>", color=COLOR_RP, lw=1.7, mutation_scale=15, zorder=8,
    ))
    ax.text(
        (lt_x0 + lt_x1) / 2, y + 0.12, "Role-playing",
        ha="center", va="bottom",
        fontsize=12.5, fontweight="bold", color=COLOR_RP,
    )
    # Right
    rt_x0 = hi - 0.28 * span
    rt_x1 = hi - 0.08 * span
    ax.add_patch(FancyArrowPatch(
        (rt_x0, y), (rt_x1, y),
        arrowstyle="-|>", color=COLOR_AS, lw=1.7, mutation_scale=15, zorder=8,
    ))
    ax.text(
        (rt_x0 + rt_x1) / 2, y + 0.12, "Assistant-like",
        ha="center", va="bottom",
        fontsize=12.5, fontweight="bold", color=COLOR_AS,
    )


def _authority_markers(
    ax: plt.Axes,
    models: List[ModelData],
    *,
    base_y: float = 0.75,
    step_y: float = 0.55,
) -> None:
    ordered = sorted(models, key=lambda m: m.authority_cos)
    for idx, m in enumerate(ordered):
        y = base_y + step_y * idx
        ax.scatter(
            [m.authority_cos], [y],
            marker=m.spec.marker,
            s=m.spec.marker_size,
            color=m.spec.marker_color,
            edgecolor="black", linewidth=0.9,
            zorder=7,
        )
        # Inline text label to the right of each marker -- tiny, subtle.
        ax.text(
            m.authority_cos, y + 0.02, f"  {m.spec.display}",
            ha="left", va="center",
            fontsize=9.2, color="#222",
            zorder=7,
        )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def build_figure(models: List[ModelData], output_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 4.8))

    pooled, tagged = _pooled(models)
    all_vals = (
        list(pooled)
        + [m.default_cos for m in models]
        + [m.authority_cos for m in models]
    )
    lo, hi = min(all_vals), max(all_vals)
    pad = 0.08 * (hi - lo + 1e-6)
    xlim = (lo - pad, hi + pad)

    # Axis styling -- bare, paper-like strip
    ax.set_xlim(*xlim)
    ax.set_ylim(-1.70, 4.20)
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.axhline(0.0, color=COLOR_AXIS_LINE, linewidth=1.3, zorder=4)
    ax.tick_params(axis="x", length=4, pad=3)

    # Smooth KDE-style pooled histogram with soft gradient
    _gradient_histogram(ax, pooled, xlim=xlim, height=2.20)
    _role_dots(ax, pooled, xlim=xlim, y=0.0, size=16)

    # Authority markers (one per model)
    _authority_markers(ax, models, base_y=0.70, step_y=0.55)

    # Extreme role labels pooled across models -- all above so they don't
    # collide with the bottom arrows.
    _label_extremes(ax, tagged, n_left=3, n_right=3, font_size=9.4)

    # Role-playing / Assistant-like arrows at bottom (well clear of labels)
    _side_arrows(ax, xlim, y=-1.20)

    ax.set_xlabel("Cosine similarity with the Assistant Axis", labelpad=6)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        out_path = output_stem.with_suffix(f".{ext}")
        fig.savefig(out_path)
        print(f"Saved: {out_path}")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models = [load_model(spec) for spec in MODELS]
    for m in models:
        print(
            f"{m.spec.display:10s}  L{m.layer:<3d}  n={len(m.role_cos):2d}  "
            f"default={m.default_cos:+.3f}  authority={m.authority_cos:+.4f}"
        )
    build_figure(models, OUTPUT_DIR / "assistant_axis_deconfound")


if __name__ == "__main__":
    main()
