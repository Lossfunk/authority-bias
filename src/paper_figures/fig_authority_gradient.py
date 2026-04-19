"""Authority gradient figure: flip rate across 5 framing levels.

Style: Assistant Axis / Anthropic editorial look.
Palette: Okabe-Ito adapted through our theme.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.theme import apply_theme, PAL, save_fig, add_y_grid

# Okabe-Ito palette (CVD-safe)
OI = {
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

MODEL_STYLE = {
    "GPT-oss-20B": {"color": OI["vermillion"], "marker": "s", "label": "GPT-oss-20B"},
    "Gemma-4-26B": {"color": OI["green"], "marker": "D", "label": "Gemma-4-26B"},
    "OLMo-2-32B": {"color": OI["orange"], "marker": "^", "label": "OLMo-2-32B"},
}

LEVELS = ["none", "uncertain", "assertive", "weak", "authoritative"]
LEVEL_LABELS = [
    "None\n(baseline)",
    "Uncertain\n\"one source\nsuggests\"",
    "Assertive\n\"strongly\nsuggested\"",
    "Weak\n\"the answer\nis X\"",
    "Authoritative\n\"verified\nsource\"",
]


def load_data(results_dir: Path) -> dict:
    audit_path = results_dir / "_shared" / "dynamic_parser_all_runs.json"
    if not audit_path.exists():
        audit_path = results_dir / "dynamic_parser_all_runs.json"
    audit = json.load(open(audit_path))

    data = {}
    for model_name, mappings in [
        ("GPT-oss-20B", {
            "weak": "openai/gpt-oss-20b::gradient_gpt_oss_weak",
            "uncertain": "openai/gpt-oss-20b::gradient_gpt_oss_uncertain",
            "assertive": "openai/gpt-oss-20b::gradient_gpt_oss_assertive",
            "authoritative": "openai/gpt-oss-20b::v2_gpt_oss_freegen",
        }),
        ("Gemma-4-26B", {
            "weak": "google/gemma-4-26B-A4B-it::gradient_gemma4_weak",
            "uncertain": "google/gemma-4-26B-A4B-it::gradient_gemma4_uncertain",
            "assertive": "google/gemma-4-26B-A4B-it::gradient_gemma4_assertive",
            "authoritative": "google/gemma-4-26B-A4B-it::v2_gemma4_freegen",
        }),
        ("OLMo-2-32B", {
            "weak": "allenai/OLMo-2-0325-32B-Instruct::gradient_olmo2_weak",
            "uncertain": "allenai/OLMo-2-0325-32B-Instruct::gradient_olmo2_uncertain",
            "assertive": "allenai/OLMo-2-0325-32B-Instruct::gradient_olmo2_assertive",
            "authoritative": "allenai/OLMo-2-0325-32B-Instruct::v2_olmo2_freegen",
        }),
    ]:
        flip_rates = []
        for level in LEVELS:
            if level == "none":
                flip_rates.append(0.0)
            elif level in mappings:
                entry = audit[mappings[level]]
                fr = entry.get("flip_rate")
                flip_rates.append(fr * 100 if fr is not None else None)
            else:
                flip_rates.append(None)
        data[model_name] = flip_rates

    return data


def make_figure(data: dict, output_dir: Path) -> None:
    apply_theme()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x = np.arange(len(LEVELS))

    for model_name, style in MODEL_STYLE.items():
        rates = data.get(model_name)
        if rates is None:
            continue
        xs_valid = [i for i, r in enumerate(rates) if r is not None]
        ys_valid = [rates[i] for i in xs_valid]

        ax.plot(
            xs_valid, ys_valid,
            color=style["color"],
            marker=style["marker"],
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.2,
            linewidth=2.2,
            label=style["label"],
            zorder=3,
        )

        for xi, yi in zip(xs_valid, ys_valid):
            if yi is not None and xi == len(LEVELS) - 1:
                y_offset = 0
                if model_name == "GPT-oss-20B":
                    y_offset = 10
                elif model_name == "Gemma-4-26B":
                    y_offset = -10
                ax.annotate(
                    f"{yi:.1f}%",
                    xy=(xi, yi),
                    xytext=(8, y_offset),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color=style["color"],
                    va="center",
                    ha="left",
                )

    ax.axhspan(0, 100, color="#FDF0EE", alpha=0.4, zorder=0, linewidth=0)

    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_LABELS, fontsize=8, color=PAL.dark_text)
    ax.set_ylabel("W1 flip rate among baseline-correct items (%)", fontsize=9)
    ax.set_xlabel("Authority framing level", fontsize=9, labelpad=8)
    ax.set_ylim(-2, 100)
    ax.set_xlim(-0.3, len(LEVELS) - 0.4)

    add_y_grid(ax, alpha=0.15)

    ax.axhline(0, color=PAL.faint_gray, linewidth=0.7, zorder=1)

    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor=PAL.faint_gray,
        framealpha=0.9,
        fontsize=8,
    )

    ax.set_title(
        "Authority-framing gradient: flip rate in free generation",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )

    ax.text(
        0.5, -0.22,
        "Flip rate = fraction of baseline-correct items that switch to wrong answer under wrong authority.",
        transform=ax.transAxes,
        fontsize=7,
        color=PAL.medium_gray,
        ha="center",
        style="italic",
    )

    save_fig(fig, "authority_gradient", output_dir)
    print(f"Saved to {output_dir}/authority_gradient.png")


if __name__ == "__main__":
    results_dir = Path("neurips-results")
    output_dir = Path("figures/neurips")
    data = load_data(results_dir)
    make_figure(data, output_dir)
