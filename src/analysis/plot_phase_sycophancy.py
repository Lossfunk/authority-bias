#!/usr/bin/env python3
"""Plots for Phase 1 + Phase 2 sycophancy results.

Generates two high-quality figures using the Okabe-Ito palette and rounded
matplotlib patches (FancyBboxPatch).

Run:
  uv run python -m src.analysis.plot_phase_sycophancy
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from textwrap import shorten
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#999999",
    "black": "#000000",
    "white": "#FFFFFF",
}


mpl.rcParams.update(
    {
        "figure.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.5,
    }
)


def _read_jsonl(path: Path) -> List[dict]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _short_label(dataset: str, question: str, width: int = 62) -> str:
    q = " ".join((question or "").split())
    return f"{dataset}: {shorten(q, width=width, placeholder='…')}"


def _draw_rounded_bar(
    ax,
    *,
    x_center: float,
    height: float,
    width: float,
    color: str,
    alpha: float = 0.9,
    rounding: float = 0.12,
) -> FancyBboxPatch:
    """Vertical bar using FancyBboxPatch. Supports negative heights."""
    y0 = min(0.0, height)
    h = abs(height)
    x0 = x_center - width / 2
    # Avoid overly-large rounding when bars are tiny.
    rounding_eff = min(rounding, width * 0.45, (h * 0.45) if h > 0 else rounding)
    patch = FancyBboxPatch(
        (x0, y0),
        width,
        h,
        boxstyle=f"round,pad=0,rounding_size={rounding_eff}",
        facecolor=color,
        edgecolor=OKABE_ITO["white"],
        linewidth=1.2,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def _draw_rounded_barh(
    ax,
    *,
    y_center: float,
    width: float,
    height: float,
    color: str,
    alpha: float = 0.9,
    rounding: Optional[float] = None,
) -> FancyBboxPatch:
    """Horizontal bar using FancyBboxPatch. Supports negative widths."""
    if rounding is None:
        rounding = height * 0.45
    x0 = 0.0 if width >= 0 else width
    w = abs(width)
    y0 = y_center - height / 2
    patch = FancyBboxPatch(
        (x0, y0),
        w,
        height,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor=color,
        edgecolor=OKABE_ITO["white"],
        linewidth=1.2,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def _add_stat_box(
    ax,
    *,
    text: str,
    xy: Tuple[float, float] = (0.02, 0.98),
    box_size: Tuple[float, float] = (0.52, 0.18),
    facecolor: str = "#FFFFFF",
    edgecolor: str = "#000000",
    alpha: float = 0.08,
    text_color: str = "#000000",
) -> None:
    """Add a rounded stats box in axes coords."""
    x, y = xy
    w, h = box_size
    patch = FancyBboxPatch(
        (x, y - h),
        w,
        h,
        transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.4,
        alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.02,
        y - 0.02,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=text_color,
    )


def plot_phase1(*, phase1_path: Path, output_dir: Path, formats: Sequence[str]) -> List[Path]:
    records = _read_jsonl(phase1_path)
    rows = []
    for r in records:
        fc = r["forced_choice"]
        rows.append(
            {
                "uid": r.get("uid", ""),
                "dataset": r.get("dataset", ""),
                "question": r.get("question", ""),
                "neutral": float(fc["neutral"]),
                "assert_wrong": float(fc["user_asserts_wrong"]),
                "assert_correct": float(fc["user_asserts_correct"]),
                "deny_correct": float(fc["user_denies_correct"]),
                "delta": float(r["sycophancy_score"]),
            }
        )

    rows.sort(key=lambda x: x["delta"], reverse=True)
    n = len(rows)
    y = np.arange(n)

    labels = [_short_label(r["dataset"], r["question"]) for r in rows]
    neutral = np.array([r["neutral"] for r in rows])
    uw = np.array([r["assert_wrong"] for r in rows])
    uc = np.array([r["assert_correct"] for r in rows])
    ud = np.array([r["deny_correct"] for r in rows])
    delta = np.array([r["delta"] for r in rows])

    fig = plt.figure(figsize=(13.5, max(5.5, 0.45 * n)))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[3.6, 1.4], wspace=0.06)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)

    # Panel A: per-question forced-choice probabilities across conditions
    for i in range(n):
        ax0.plot([neutral[i], uw[i]], [y[i], y[i]], color=OKABE_ITO["grey"], alpha=0.55, lw=2)

    ax0.scatter(neutral, y, s=52, color=OKABE_ITO["grey"], label="Neutral", zorder=3)
    ax0.scatter(uw, y, s=60, color=OKABE_ITO["vermillion"], label="User asserts wrong", zorder=4)
    ax0.scatter(uc, y, s=46, color=OKABE_ITO["bluish_green"], label="User asserts correct", zorder=3)
    ax0.scatter(ud, y, s=46, color=OKABE_ITO["orange"], label="User denies correct", zorder=3)

    ax0.set_xlim(-0.02, 1.02)
    ax0.set_xlabel(r"Forced-choice $P(\mathrm{wrong})$")
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels)
    ax0.grid(axis="x", alpha=0.25, color=OKABE_ITO["grey"], linewidth=0.8)
    ax0.set_title("Phase 1 (10 examples): forced-choice shift by user belief", loc="left")
    ax0.invert_yaxis()
    ax0.legend(loc="lower right", ncol=2, fontsize=9)

    # Panel B: delta bars (assert wrong - neutral)
    ax1.axvline(0, color=OKABE_ITO["black"], lw=1.2, alpha=0.5)
    bar_h = 0.62
    for i in range(n):
        c = OKABE_ITO["vermillion"] if delta[i] >= 0 else OKABE_ITO["bluish_green"]
        _draw_rounded_barh(ax1, y_center=float(y[i]), width=float(delta[i]), height=bar_h, color=c, alpha=0.88)

    ax1.set_xlabel(r"$\Delta$ (assert wrong $-$ neutral)")
    ax1.set_xlim(min(-0.2, float(delta.min()) - 0.05), max(0.25, float(delta.max()) + 0.05))
    ax1.grid(axis="x", alpha=0.18, color=OKABE_ITO["grey"], linewidth=0.8)
    ax1.tick_params(axis="y", left=False, labelleft=False)

    m = float(delta.mean()) if n else 0.0
    med = float(np.median(delta)) if n else 0.0
    pos = float(np.mean(delta > 0)) if n else 0.0
    _add_stat_box(
        ax1,
        text=(
            f"n = {n}\n"
            f"mean Δ = {m:+.3f}\n"
            f"median Δ = {med:+.3f}\n"
            f"% positive = {pos*100:.1f}%"
        ),
        xy=(0.05, 0.98),
        box_size=(0.9, 0.18),
        facecolor=OKABE_ITO["vermillion"],
        edgecolor=OKABE_ITO["vermillion"],
        alpha=0.10,
        text_color=OKABE_ITO["black"],
    )

    plt.tight_layout()

    out_paths: List[Path] = []
    for fmt in formats:
        out = output_dir / f"phase1_sycophancy.{fmt}"
        _ensure_parent(out)
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        out_paths.append(out)
    plt.close(fig)
    return out_paths


@dataclass(frozen=True)
class Phase2ModelData:
    name: str
    syc: np.ndarray
    margin: np.ndarray
    neutral: np.ndarray


def _load_phase2_model(path: Path) -> Phase2ModelData:
    rows = _read_jsonl(path)
    syc = np.array([float(r["sycophancy_score"]) for r in rows], dtype=float)
    margin = np.array([float(r["delta_margin"]) for r in rows], dtype=float)
    neutral = np.array([float(r["forced_choice"]["neutral"]) for r in rows], dtype=float)
    name = str(rows[0].get("model", path.stem)) if rows else path.stem
    return Phase2ModelData(name=name, syc=syc, margin=margin, neutral=neutral)


def _paired_by_uid(base_path: Path, instruct_path: Path) -> Tuple[dict, dict, List[str]]:
    base = {r["uid"]: r for r in _read_jsonl(base_path)}
    inst = {r["uid"]: r for r in _read_jsonl(instruct_path)}
    uids = sorted(set(base) & set(inst))
    return base, inst, uids


def _mean_ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    vals = list(values)
    if not vals:
        return 0.0, 0.0, 0.0
    mu = mean(vals)
    if len(vals) == 1:
        return mu, mu, mu
    se = pstdev(vals) / math.sqrt(len(vals))
    lo = mu - 1.96 * se
    hi = mu + 1.96 * se
    return mu, lo, hi


def plot_phase2(
    *,
    base_path: Path,
    instruct_path: Path,
    output_dir: Path,
    formats: Sequence[str],
) -> List[Path]:
    base_data = _load_phase2_model(base_path)
    inst_data = _load_phase2_model(instruct_path)

    base_by, inst_by, uids = _paired_by_uid(base_path, instruct_path)
    d_fc = [float(inst_by[uid]["sycophancy_score"]) - float(base_by[uid]["sycophancy_score"]) for uid in uids]
    d_dm = [float(inst_by[uid]["delta_margin"]) - float(base_by[uid]["delta_margin"]) for uid in uids]
    mu_fc, lo_fc, hi_fc = _mean_ci95(d_fc)
    mu_dm, lo_dm, hi_dm = _mean_ci95(d_dm)

    fig = plt.figure(figsize=(11.5, 10.5))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.0, 1.05], hspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])

    # Panel A: distribution of Δforced-choice (sycophancy_score)
    bins_fc = np.linspace(-1.0, 1.0, 61)
    ax0.hist(
        base_data.syc,
        bins=bins_fc,
        density=True,
        color=OKABE_ITO["blue"],
        alpha=0.35,
        edgecolor=OKABE_ITO["white"],
        linewidth=0.6,
        label="Base",
    )
    ax0.hist(
        inst_data.syc,
        bins=bins_fc,
        density=True,
        color=OKABE_ITO["vermillion"],
        alpha=0.30,
        edgecolor=OKABE_ITO["white"],
        linewidth=0.6,
        label="Instruct",
    )
    ax0.axvline(0, color=OKABE_ITO["black"], lw=1.2, alpha=0.55)
    ax0.axvline(float(base_data.syc.mean()), color=OKABE_ITO["blue"], lw=2.0, alpha=0.85)
    ax0.axvline(float(inst_data.syc.mean()), color=OKABE_ITO["vermillion"], lw=2.0, alpha=0.85)
    ax0.set_xlim(-1.0, 1.0)
    ax0.set_ylabel("Density")
    ax0.set_title("Phase 2 (n=1813): Base vs Instruct distribution shift", loc="left")
    ax0.set_xlabel(r"Sycophancy score: $\Delta\,P(\mathrm{wrong})$ (assert wrong $-$ neutral)")
    ax0.grid(axis="y", alpha=0.18, color=OKABE_ITO["grey"], linewidth=0.8)
    ax0.legend(loc="upper left")

    _add_stat_box(
        ax0,
        text=(
            "Δfc summary\n"
            f"Base mean={base_data.syc.mean():+.3f}  median={np.median(base_data.syc):+.3f}\n"
            f"Inst  mean={inst_data.syc.mean():+.3f}  median={np.median(inst_data.syc):+.3f}\n"
            f"Inst-Base mean={mu_fc:+.3f}  95% CI [{lo_fc:+.3f}, {hi_fc:+.3f}]"
        ),
        xy=(0.52, 0.98),
        box_size=(0.46, 0.22),
        facecolor=OKABE_ITO["grey"],
        edgecolor=OKABE_ITO["black"],
        alpha=0.08,
        text_color=OKABE_ITO["black"],
    )

    # Panel B: distribution of Δmargin
    lo = float(min(base_data.margin.min(), inst_data.margin.min()))
    hi = float(max(base_data.margin.max(), inst_data.margin.max()))
    span = hi - lo
    bins_dm = np.linspace(lo - 0.05 * span, hi + 0.05 * span, 71)
    ax1.hist(
        base_data.margin,
        bins=bins_dm,
        density=True,
        color=OKABE_ITO["blue"],
        alpha=0.35,
        edgecolor=OKABE_ITO["white"],
        linewidth=0.6,
        label="Base",
    )
    ax1.hist(
        inst_data.margin,
        bins=bins_dm,
        density=True,
        color=OKABE_ITO["vermillion"],
        alpha=0.30,
        edgecolor=OKABE_ITO["white"],
        linewidth=0.6,
        label="Instruct",
    )
    ax1.axvline(0, color=OKABE_ITO["black"], lw=1.2, alpha=0.55)
    ax1.axvline(float(base_data.margin.mean()), color=OKABE_ITO["blue"], lw=2.0, alpha=0.85)
    ax1.axvline(float(inst_data.margin.mean()), color=OKABE_ITO["vermillion"], lw=2.0, alpha=0.85)
    ax1.set_ylabel("Density")
    ax1.set_xlabel(r"$\Delta$margin: $(\log P_w-\log P_c)_{assert\ wrong}-(\log P_w-\log P_c)_{neutral}$")
    ax1.grid(axis="y", alpha=0.18, color=OKABE_ITO["grey"], linewidth=0.8)
    ax1.legend(loc="upper left")

    _add_stat_box(
        ax1,
        text=(
            "Δmargin summary\n"
            f"Base mean={base_data.margin.mean():+.2f}  median={np.median(base_data.margin):+.2f}\n"
            f"Inst  mean={inst_data.margin.mean():+.2f}  median={np.median(inst_data.margin):+.2f}\n"
            f"Inst-Base mean={mu_dm:+.2f}  95% CI [{lo_dm:+.2f}, {hi_dm:+.2f}]"
        ),
        xy=(0.52, 0.98),
        box_size=(0.46, 0.22),
        facecolor=OKABE_ITO["grey"],
        edgecolor=OKABE_ITO["black"],
        alpha=0.08,
        text_color=OKABE_ITO["black"],
    )

    # Panel C: Instruct-Base Δfc as a function of Base neutral confidence
    base_neu = np.array([float(base_by[uid]["forced_choice"]["neutral"]) for uid in uids], dtype=float)
    d_fc_arr = np.array(d_fc, dtype=float)
    d_dm_arr = np.array(d_dm, dtype=float)

    edges = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(base_neu, edges[1:-1], right=False)  # 0..9
    x = np.arange(len(edges) - 1)
    mean_dfc = np.zeros_like(x, dtype=float)
    mean_ddm = np.zeros_like(x, dtype=float)
    counts = np.zeros_like(x, dtype=int)
    for i in range(len(x)):
        mask = bin_ids == i
        counts[i] = int(mask.sum())
        if counts[i] == 0:
            mean_dfc[i] = np.nan
            mean_ddm[i] = np.nan
            continue
        mean_dfc[i] = float(d_fc_arr[mask].mean())
        mean_ddm[i] = float(d_dm_arr[mask].mean())

    ax2.axhline(0, color=OKABE_ITO["black"], lw=1.2, alpha=0.55)
    bar_w = 0.82
    for i in range(len(x)):
        if np.isnan(mean_dfc[i]):
            continue
        c = OKABE_ITO["vermillion"] if mean_dfc[i] >= 0 else OKABE_ITO["bluish_green"]
        _draw_rounded_bar(
            ax2,
            x_center=float(x[i]),
            height=float(mean_dfc[i]),
            width=bar_w,
            color=c,
            alpha=0.88,
            rounding=0.18,
        )

    ax2.set_xlim(-0.6, float(x.max()) + 0.6)
    ax2.set_ylabel("Mean (Instruct − Base) Δfc")
    ax2.set_xlabel("Base neutral forced-choice P(wrong) bin")
    ax2.grid(axis="y", alpha=0.18, color=OKABE_ITO["grey"], linewidth=0.8)
    xticklabels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}\n(n={counts[i]})" for i in range(len(x))]
    ax2.set_xticks(x)
    ax2.set_xticklabels(xticklabels)
    ax2.set_title("Nonlinearity: Δprob depends on where you start (binning by Base neutral)", loc="left")

    ax2b = ax2.twinx()
    ax2b.plot(x, mean_ddm, color=OKABE_ITO["reddish_purple"], marker="o", lw=2.0, ms=4.5, label="Mean Δmargin")
    ax2b.set_ylabel("Mean (Instruct − Base) Δmargin", color=OKABE_ITO["reddish_purple"])
    ax2b.tick_params(axis="y", labelcolor=OKABE_ITO["reddish_purple"])

    # Legend for panel C
    handles = [
        Line2D([0], [0], color=OKABE_ITO["vermillion"], lw=8, label="Δfc > 0"),
        Line2D([0], [0], color=OKABE_ITO["bluish_green"], lw=8, label="Δfc < 0"),
        Line2D([0], [0], color=OKABE_ITO["reddish_purple"], lw=2, marker="o", label="Δmargin"),
    ]
    ax2.legend(handles=handles, loc="upper right", fontsize=9)

    plt.tight_layout()
    out_paths: List[Path] = []
    for fmt in formats:
        out = output_dir / f"phase2_base_vs_instruct.{fmt}"
        _ensure_parent(out)
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        out_paths.append(out)
    plt.close(fig)
    return out_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Phase 1 + Phase 2 sycophancy results")
    parser.add_argument(
        "--phase1-path",
        type=Path,
        default=Path("new-phase-results/phase1_sycophancy.jsonl"),
        help="Path to Phase 1 JSONL.",
    )
    parser.add_argument(
        "--phase2-dir",
        type=Path,
        default=Path("new-phase-results/phase2"),
        help="Directory containing Phase 2 JSONLs.",
    )
    parser.add_argument(
        "--phase2-base",
        type=Path,
        default=None,
        help="Phase 2 Base JSONL path (defaults to *Llama-3.1-8B.jsonl in --phase2-dir).",
    )
    parser.add_argument(
        "--phase2-instruct",
        type=Path,
        default=None,
        help="Phase 2 Instruct JSONL path (defaults to *Llama-3.1-8B-Instruct.jsonl in --phase2-dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("new-phase-results/figures"),
        help="Where to write figures.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["png"],
        help="Output formats (e.g. png pdf).",
    )
    return parser.parse_args()


def _resolve_phase2_paths(phase2_dir: Path, base_path: Optional[Path], instruct_path: Optional[Path]) -> Tuple[Path, Path]:
    if base_path is not None and instruct_path is not None:
        return base_path, instruct_path

    jsonls = sorted(phase2_dir.glob("*.jsonl"))
    if not jsonls:
        raise FileNotFoundError(f"No .jsonl files found under {phase2_dir}")

    def pick(pred) -> Optional[Path]:
        for p in jsonls:
            if pred(p.name):
                return p
        return None

    base = base_path or pick(lambda n: ("Instruct" not in n) and ("Llama-3.1-8B" in n))
    inst = instruct_path or pick(lambda n: ("Instruct" in n) and ("Llama-3.1-8B" in n))
    if base is None or inst is None:
        raise FileNotFoundError(
            "Could not infer phase2 base/instruct jsonl paths. Provide --phase2-base and --phase2-instruct explicitly."
        )
    return base, inst


def main() -> None:
    args = parse_args()

    if not args.phase1_path.exists():
        fallback = Path("results/phase1_sycophancy.jsonl")
        if fallback.exists():
            args.phase1_path = fallback
        else:
            raise FileNotFoundError(f"Phase 1 file not found: {args.phase1_path}")

    base_path, instruct_path = _resolve_phase2_paths(args.phase2_dir, args.phase2_base, args.phase2_instruct)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out1 = plot_phase1(phase1_path=args.phase1_path, output_dir=args.output_dir, formats=args.formats)
    out2 = plot_phase2(base_path=base_path, instruct_path=instruct_path, output_dir=args.output_dir, formats=args.formats)

    print("Wrote:")
    for p in out1 + out2:
        print(f"  {p}")


if __name__ == "__main__":
    main()
