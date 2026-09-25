"""Recompute Qwen Wang Pareto summaries (trivia + PIQA) from raw_text.

For each variant directory under results/baselines/qwen/mitigation_pareto/qwen35,
this script:
  1. Reads the *_rows.jsonl
  2. Runs dynamic_parser.discover_model_markers + dynamic_parse_row on raw_text
  3. Updates parsed_label / is_correct / chose_wrong on each row
  4. Aggregates per-config_id+alpha and writes the parser-recomputed summary
  5. Writes a consolidated qwen_wang_pareto_table.json

Mirrors the OLMo-2 procedure already used for the verified Wang Pareto numbers.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.authority_steering.dynamic_parser import discover_model_markers, dynamic_parse_row  # noqa: E402

QWEN_DIR = ROOT / "results/baselines" / "qwen" / "mitigation_pareto" / "qwen35"

TRIVIA_VARIANTS = ["residualized", "assistant", "caa"]
PIQA_VARIANTS = ["residualized", "assistant", "caa"]


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
                "flip_rate": n_wrong / parsed if parsed else 0.0,
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
                "flip_rate_parsed": n_wrong / parsed if parsed else 0.0,
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


def build_consolidated_table(qwen_dir: Path) -> dict:
    out: dict = {
        "model": "qwen35",
        "rows_source": str(qwen_dir.relative_to(ROOT)),
        "tasks": {"trivia": [], "piqa": []},
    }

    for variant in TRIVIA_VARIANTS:
        sp = qwen_dir / f"trivia_{variant}" / "steering_summary.json"
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
                    "parsed": entry["parsed"],
                    "n": entry["total"],
                }
            )

    for variant in PIQA_VARIANTS:
        sp = qwen_dir / f"piqa_{variant}" / "piqa_summary.json"
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
                    "parsed": entry["parsed"],
                    "n": entry["n"],
                }
            )

    return out


def main() -> None:
    print(f"[regen] qwen dir = {QWEN_DIR}")
    info = []
    for variant in TRIVIA_VARIANTS:
        d = QWEN_DIR / f"trivia_{variant}"
        info.append(("trivia", variant, process_variant(d, "trivia")))
    for variant in PIQA_VARIANTS:
        d = QWEN_DIR / f"piqa_{variant}"
        info.append(("piqa", variant, process_variant(d, "piqa")))

    for kind, variant, meta in info:
        print(f"  [{kind}/{variant}] rows={meta['n_rows']} configs={meta['n_configs']}")

    table = build_consolidated_table(QWEN_DIR)
    out_path = QWEN_DIR / "qwen_wang_pareto_table.json"
    out_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"[regen] wrote {out_path}")


if __name__ == "__main__":
    main()
