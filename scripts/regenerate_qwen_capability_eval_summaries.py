"""Recompute Qwen capability-eval summaries (trivia + PIQA, N0_note) from raw_text.

For each *_capability variant directory under
wang-pareto-results/qwen/capability_eval/qwen35, this script:
  1. Reads the *_rows.jsonl
  2. Runs dynamic_parser.discover_model_markers + dynamic_parse_row on raw_text
  3. Updates parsed_label / is_correct / chose_wrong on each row
  4. Aggregates per-config_id+alpha and writes the parser-recomputed summary
  5. Writes a consolidated qwen_capability_eval_table.json
  6. Writes a comparison table joining capability (N0_note) with W1_note Pareto.

Mirrors the OLMo-2 / W1 Qwen procedure used for the verified Wang Pareto numbers.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp16.dynamic_parser import discover_model_markers, dynamic_parse_row  # noqa: E402

CAP_DIR = ROOT / "wang-pareto-results" / "qwen" / "capability_eval" / "qwen35"
W1_TABLE_PATH = (
    ROOT
    / "wang-pareto-results"
    / "qwen"
    / "mitigation_pareto"
    / "qwen35"
    / "qwen_wang_pareto_table.json"
)

VARIANTS = ["residualized", "assistant", "caa"]


def _row_to_parser_input(row: dict) -> dict:
    return {
        "generation": {"raw_text": row.get("raw_text", row.get("response", "")) or ""},
        "prompt_text": row.get("prompt_text", ""),
        "correct_answer": row.get("correct_text", ""),
        "wrong_answer": row.get("wrong_text", ""),
        "correct_label": row.get("correct_label", "B"),
        "wrong_label": row.get("wrong_label", "A"),
    }


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _save_rows(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _reparse(rows: list[dict]) -> list[dict]:
    parser_inputs = [_row_to_parser_input(r) for r in rows]
    markers = discover_model_markers(parser_inputs)
    out_rows: list[dict] = []
    for original, parser_in in zip(rows, parser_inputs):
        label, _meta = dynamic_parse_row(parser_in, markers)
        new = dict(original)
        if label in {"A", "B"}:
            new["parsed_label"] = label
            new["is_correct"] = label == original.get("correct_label")
            new["chose_wrong"] = label == original.get("wrong_label")
        else:
            new["parsed_label"] = None
            new["is_correct"] = None
            new["chose_wrong"] = None
        out_rows.append(new)
    return out_rows


def _summarize_trivia(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float, str], list[dict]] = defaultdict(list)
    for r in rows:
        key = (r.get("config_id", "?"), float(r.get("alpha", 0.0)), r.get("condition_code", "?"))
        groups[key].append(r)

    summaries: list[dict] = []
    for (cfg, alpha, cond), grp in sorted(groups.items()):
        total = len(grp)
        parsed = sum(1 for r in grp if r.get("parsed_label") in {"A", "B"})
        n_correct = sum(1 for r in grp if r.get("is_correct") is True)
        n_wrong = sum(1 for r in grp if r.get("chose_wrong") is True)
        summaries.append(
            {
                "config_id": cfg,
                "alpha": alpha,
                "condition": cond,
                "total": total,
                "parsed": parsed,
                "parse_rate": parsed / total if total else 0.0,
                "accuracy": n_correct / parsed if parsed else 0.0,
                "wrong_rate": n_wrong / parsed if parsed else 0.0,
                "flip_rate": None,  # N0_note has no Wang flip semantics
                "mean_fc_logit_wrong": None,
                "std_fc_logit_wrong": None,
            }
        )
    return summaries


def _summarize_piqa(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float, str], list[dict]] = defaultdict(list)
    for r in rows:
        key = (r.get("config_id", "?"), float(r.get("alpha", 0.0)), r.get("condition_code", "?"))
        groups[key].append(r)

    summaries: list[dict] = []
    for (cfg, alpha, cond), grp in sorted(groups.items()):
        n = len(grp)
        parsed = sum(1 for r in grp if r.get("parsed_label") in {"A", "B"})
        n_correct = sum(1 for r in grp if r.get("is_correct") is True)
        n_wrong = sum(1 for r in grp if r.get("chose_wrong") is True)
        sample = grp[0]
        summaries.append(
            {
                "config_id": cfg,
                "condition_code": cond,
                "n": n,
                "parsed": parsed,
                "parse_rate": parsed / n if n else 0.0,
                "accuracy_parsed": n_correct / parsed if parsed else 0.0,
                "wrong_rate_parsed": n_wrong / parsed if parsed else 0.0,
                "flip_rate_parsed": None,
                "mode": sample.get("mode"),
                "alpha": alpha,
                "direction_kind": sample.get("direction_kind"),
                "sparse_topk": sample.get("sparse_topk", 0),
                "target_layers": sample.get("target_layers", []),
                "components": sample.get("components", []),
                "position_mode": sample.get("position_mode"),
                "apply_phase": sample.get("apply_phase"),
                "norm_scaling": sample.get("norm_scaling"),
            }
        )
    return summaries


def process_variant(variant_dir: Path, kind: str) -> dict:
    if kind == "trivia":
        rows_path = variant_dir / "steering_rows.jsonl"
        summary_path = variant_dir / "steering_summary.json"
    else:
        rows_path = variant_dir / "piqa_rows.jsonl"
        summary_path = variant_dir / "piqa_summary.json"

    if not rows_path.exists():
        raise FileNotFoundError(rows_path)

    raw_rows = _load_rows(rows_path)
    reparsed = _reparse(raw_rows)
    _save_rows(reparsed, rows_path)

    summaries = _summarize_trivia(reparsed) if kind == "trivia" else _summarize_piqa(reparsed)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return {
        "rows_path": str(rows_path),
        "summary_path": str(summary_path),
        "n_rows": len(reparsed),
        "n_configs": len(summaries),
    }


def build_capability_table(qwen_dir: Path) -> dict:
    out: dict = {
        "model": "qwen35",
        "condition": "N0_note",
        "rows_source": str(qwen_dir.relative_to(ROOT)),
        "tasks": {"trivia": [], "piqa": []},
    }

    for variant in VARIANTS:
        sp = qwen_dir / f"trivia_{variant}_capability" / "steering_summary.json"
        if not sp.exists():
            continue
        for entry in json.loads(sp.read_text()):
            target_layers = entry.get("target_layers")
            if target_layers is None and "config_id" in entry:
                m = re.search(r"_L(\d+)(?:_|$)", entry["config_id"])
                if m:
                    target_layers = [int(m.group(1))]
            layer = target_layers[0] if target_layers else None
            out["tasks"]["trivia"].append(
                {
                    "variant": variant,
                    "layer": layer,
                    "alpha": entry["alpha"],
                    "wrong_rate": entry["wrong_rate"],
                    "accuracy": entry["accuracy"],
                    "parsed": entry["parsed"],
                    "n": entry["total"],
                }
            )

    for variant in VARIANTS:
        sp = qwen_dir / f"piqa_{variant}_capability" / "piqa_summary.json"
        if not sp.exists():
            continue
        for entry in json.loads(sp.read_text()):
            target_layers = entry.get("target_layers")
            layer = target_layers[0] if target_layers else None
            out["tasks"]["piqa"].append(
                {
                    "variant": variant,
                    "layer": layer,
                    "alpha": entry["alpha"],
                    "wrong_rate": entry["wrong_rate_parsed"],
                    "accuracy": entry["accuracy_parsed"],
                    "parsed": entry["parsed"],
                    "n": entry["n"],
                }
            )

    return out


def build_comparison_table(cap_table: dict, w1_table: dict | None) -> dict | None:
    """Join N0_note (capability) with W1_note (Pareto) on (variant, alpha, layer)."""
    if w1_table is None:
        return None

    def key(r: dict) -> tuple:
        return (r["variant"], float(r["alpha"]), r.get("layer"))

    out = {
        "model": "qwen35",
        "tasks": {"trivia": [], "piqa": []},
        "notes": {
            "w1_wrong_rate": "wrong-rate when authority cue endorses the wrong answer (lower = compliance suppressed)",
            "n0_wrong_rate": "wrong-rate with no authority cue (lower = capability preserved)",
            "delta_capability": "n0_wrong_rate - n0_baseline (per variant, alpha=0)",
            "delta_compliance": "w1_wrong_rate - w1_baseline (per variant, alpha=0)",
        },
    }

    for task in ("trivia", "piqa"):
        cap_by_key = {key(r): r for r in cap_table["tasks"][task]}
        w1_by_key = {key(r): r for r in w1_table["tasks"][task]}
        # Per-variant baseline at alpha=0
        n0_baseline_by_variant = {
            r["variant"]: r["wrong_rate"] for r in cap_table["tasks"][task] if float(r["alpha"]) == 0.0
        }
        w1_baseline_by_variant = {
            r["variant"]: r["wrong_rate"] for r in w1_table["tasks"][task] if float(r["alpha"]) == 0.0
        }
        for k in sorted(set(cap_by_key) & set(w1_by_key)):
            cap = cap_by_key[k]
            w1 = w1_by_key[k]
            out["tasks"][task].append(
                {
                    "variant": cap["variant"],
                    "layer": cap["layer"],
                    "alpha": cap["alpha"],
                    "n0_wrong_rate": cap["wrong_rate"],
                    "w1_wrong_rate": w1["wrong_rate"],
                    "delta_capability_pp": (cap["wrong_rate"] - n0_baseline_by_variant[cap["variant"]]) * 100,
                    "delta_compliance_pp": (w1["wrong_rate"] - w1_baseline_by_variant[cap["variant"]]) * 100,
                    "n0_parsed": cap["parsed"],
                    "n0_n": cap["n"],
                    "w1_parsed": w1["parsed"],
                    "w1_n": w1["n"],
                }
            )

    return out


def main() -> None:
    print(f"[regen] capability dir = {CAP_DIR}")
    info = []
    for variant in VARIANTS:
        d = CAP_DIR / f"trivia_{variant}_capability"
        info.append(("trivia", variant, process_variant(d, "trivia")))
    for variant in VARIANTS:
        d = CAP_DIR / f"piqa_{variant}_capability"
        info.append(("piqa", variant, process_variant(d, "piqa")))

    for kind, variant, meta in info:
        print(f"  [{kind}/{variant}] rows={meta['n_rows']} configs={meta['n_configs']}")

    cap_table = build_capability_table(CAP_DIR)
    cap_path = CAP_DIR / "qwen_capability_eval_table.json"
    cap_path.write_text(json.dumps(cap_table, indent=2), encoding="utf-8")
    print(f"[regen] wrote {cap_path}")

    w1_table = json.loads(W1_TABLE_PATH.read_text()) if W1_TABLE_PATH.exists() else None
    comp = build_comparison_table(cap_table, w1_table)
    if comp is not None:
        comp_path = CAP_DIR / "qwen_specificity_comparison.json"
        comp_path.write_text(json.dumps(comp, indent=2), encoding="utf-8")
        print(f"[regen] wrote {comp_path}")


if __name__ == "__main__":
    main()
