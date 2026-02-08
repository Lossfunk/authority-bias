"""Analyze Exp12 persistence/washout summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp12 persistence/washout results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/exp12"),
        help="Directory containing exp12 outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/exp12/figures"),
        help="Directory for plots/report",
    )
    return parser.parse_args()


def load_model_summaries(results_dir: Path) -> List[Dict]:
    summaries = []
    for p in sorted(results_dir.glob("*_summary.json")):
        if p.name == "summary.json":
            continue
        with p.open("r") as f:
            summaries.append(json.load(f))
    return summaries


def _safe_get(d: Dict, path: List[str], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _instruction_keys(summary: Dict) -> List[str]:
    configured = summary.get("instruction_schedule_keys")
    if isinstance(configured, list) and configured:
        return [str(x) for x in configured]

    metrics = summary.get("metrics", {})
    if isinstance(metrics, dict) and metrics:
        first_tag = next(iter(metrics.values()))
        if isinstance(first_tag, dict):
            keys = [str(k)[len("instr_") :] for k in first_tag.keys() if str(k).startswith("instr_")]
            if keys:
                return sorted(keys)
    return ["0", "1"]


def _instruction_label(summary: Dict, key: str) -> str:
    labels = summary.get("instruction_schedule_labels")
    if isinstance(labels, dict) and key in labels:
        return str(labels[key])
    if key == "0":
        return "no_instruction"
    if key == "1":
        return "instruction_at_T0"
    return f"instr_{key}"


def _slug(text: str) -> str:
    return text.replace(" ", "_").replace("/", "_")


def plot_residual_curves(summary: Dict, output_dir: Path) -> List[Path]:
    paths: List[Path] = []
    model_tag = summary["model"].replace("/", "__")
    turns = int(summary["turns"])
    tags = summary["tags"]
    styles = summary["styles"]
    modes = summary.get("probe_modes", ["context"])
    instruction_keys = _instruction_keys(summary)

    for tag in tags:
        tag_key = tag.lower().replace(" ", "_")
        for instr_key in instruction_keys:
            instr_metric = f"instr_{instr_key}"
            instr_label = _instruction_label(summary, instr_key)
            fig, ax = plt.subplots(figsize=(8, 5))
            x = list(range(1, turns + 1))
            for mode in modes:
                for style in styles:
                    y = []
                    for t in x:
                        val = _safe_get(
                            summary,
                            [
                                "metrics", tag_key, instr_metric, "probes",
                                mode, style, f"T{t}", "residual_wrong_shift", "mean",
                            ],
                            0.0,
                        )
                        y.append(float(val))
                    ax.plot(x, y, marker="o", label=f"{mode}:{style}")

            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("Follow-up turn")
            ax.set_ylabel("Residual wrong shift (margin)")
            ax.set_title(f"{summary['model']} | {tag} | {instr_label}")
            ax.legend(frameon=False)
            fig.tight_layout()

            out = output_dir / f"{model_tag}_{tag_key}_{_slug(instr_key)}_residual_wrong.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            paths.append(out)
    return paths


def write_report(summaries: List[Dict], output_dir: Path) -> Path:
    report_path = output_dir / "report.md"
    lines: List[str] = []
    lines.append("# Exp12 Persistence/Washout Report")
    lines.append("")

    for summary in summaries:
        lines.append(f"## {summary['model']}")
        lines.append(f"- n_examples: {summary['n_examples']}")
        lines.append(f"- turns: {summary['turns']}")
        lines.append(f"- probe_modes: {', '.join(summary.get('probe_modes', ['context']))}")
        lines.append(f"- styles: {', '.join(summary['styles'])}")
        lines.append(f"- style_profile: {summary.get('style_profile', 'legacy')}")
        lines.append(
            f"- instruction_schedules: {', '.join(_instruction_label(summary, k) for k in _instruction_keys(summary))}"
        )
        lines.append("")

        for tag in summary["tags"]:
            tag_key = tag.lower().replace(" ", "_")
            lines.append(f"### {tag}")
            for instr_key in _instruction_keys(summary):
                i_key = f"instr_{instr_key}"
                i_label = _instruction_label(summary, instr_key)
                immediate = _safe_get(
                    summary,
                    ["metrics", tag_key, i_key, "initial", "immediate_wrong_shift", "mean"],
                    0.0,
                )
                lines.append(f"- {i_label} immediate_wrong_shift_mean: {immediate:.4f}")
                for mode in summary.get("probe_modes", ["context"]):
                    for style in summary["styles"]:
                        resid_t1 = _safe_get(
                            summary,
                            ["metrics", tag_key, i_key, "probes", mode, style, "T1", "residual_wrong_shift", "mean"],
                            0.0,
                        )
                        lines.append(f"- {i_label} {mode} {style} T1 residual_wrong_shift_mean: {resid_t1:.4f}")
            lines.append("")
        lines.append("")

    report_path.write_text("\n".join(lines))
    return report_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_model_summaries(args.results_dir)
    if not summaries:
        raise FileNotFoundError(f"No model summary files found in {args.results_dir}")

    plot_paths: List[Path] = []
    for summary in summaries:
        plot_paths.extend(plot_residual_curves(summary, args.output_dir))
    report_path = write_report(summaries, args.output_dir)

    print(f"Loaded {len(summaries)} model summaries from {args.results_dir}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote {len(plot_paths)} plots to {args.output_dir}")


if __name__ == "__main__":
    main()
