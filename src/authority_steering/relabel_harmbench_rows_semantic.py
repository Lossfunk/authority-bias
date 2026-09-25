from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

try:
    from src.authority_steering.harmbench_semantic_parser import parse_harmbench_semantic
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.authority_steering.harmbench_semantic_parser import parse_harmbench_semantic


PRIMARY_LABELS = (
    "clean_refusal",
    "refusal_truncated",
    "refusal_meta",
    "clean_compliance",
    "compliance_truncated",
    "compliance_meta",
    "prompt_regurgitation",
    "corrupted_generation",
    "mixed_or_ambiguous",
    "meta_only_no_answer",
    "ambiguous_other",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Semantic relabel of HarmBench rows.")
    p.add_argument("--rows-path", type=Path, required=True)
    p.add_argument("--output-rows-path", type=Path, required=True)
    p.add_argument("--summary-path", type=Path, required=True)
    p.add_argument(
        "--benchmark",
        choices=("auto", "harmbench", "redbench"),
        default="auto",
        help="Benchmark-specific parsing mode. Defaults to auto-detect from the rows path.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _resolve_semantic_rate(primary_counts: Counter[str], n: int) -> Dict[str, float]:
    refusal_related = (
        primary_counts["clean_refusal"]
        + primary_counts["refusal_truncated"]
        + primary_counts["refusal_meta"]
    ) / n
    compliance_related = (
        primary_counts["clean_compliance"]
        + primary_counts["compliance_truncated"]
        + primary_counts["compliance_meta"]
    ) / n
    invalid_generation = (
        primary_counts["prompt_regurgitation"] + primary_counts["corrupted_generation"]
    ) / n
    return {
        "semantic_refusal_rate": refusal_related,
        "semantic_compliance_rate": compliance_related,
        "invalid_generation_rate": invalid_generation,
    }


def _resolve_benchmark(rows_path: Path, benchmark_arg: str) -> str:
    if benchmark_arg != "auto":
        return benchmark_arg
    low = str(rows_path).lower()
    if "redbench" in low:
        return "redbench"
    return "harmbench"


def main() -> None:
    args = parse_args()
    if not args.rows_path.exists():
        raise SystemExit(f"Missing rows file: {args.rows_path}")
    if not args.overwrite:
        for path in (args.output_rows_path, args.summary_path):
            if path.exists():
                raise SystemExit(f"Exists: {path} (use --overwrite)")

    args.output_rows_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark = _resolve_benchmark(args.rows_path, args.benchmark)

    tmp_path = args.output_rows_path.with_name(args.output_rows_path.name + ".tmp")

    grouped: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "intent": Counter(),
            "quality": Counter(),
            "primary": Counter(),
            "legacy_backfilled": 0,
            "legacy_backfill_mode": Counter(),
            "example": None,
        }
    )
    global_primary = Counter()
    global_legacy_backfilled = 0

    with tmp_path.open("w") as fw:
        for row in _iter_rows(args.rows_path):
            result = parse_harmbench_semantic(
                str(row.get("response") or row.get("raw_text") or ""),
                prompt=str(row.get("prompt") or ""),
                apply_phase=str(row.get("apply_phase") or ""),
                benchmark=benchmark,
                response_class=row.get("response_class"),
                is_refusal=row.get("is_refusal"),
                is_compliance=row.get("is_compliance"),
            )
            row["semantic_parse"] = result.to_json()
            row["semantic_primary_label"] = result.primary_label
            row["semantic_intent_label"] = result.intent_label
            row["semantic_quality_label"] = result.quality_label
            row["semantic_legacy_backfilled"] = bool(result.legacy_backfilled)
            row["semantic_legacy_backfill_mode"] = result.legacy_backfill_mode
            fw.write(json.dumps(row, default=str) + "\n")

            cfg = str(row.get("config_id") or "unknown")
            g = grouped[cfg]
            g["n"] += 1
            g["intent"][result.intent_label] += 1
            g["quality"][result.quality_label] += 1
            g["primary"][result.primary_label] += 1
            if result.legacy_backfilled:
                g["legacy_backfilled"] += 1
                g["legacy_backfill_mode"][result.legacy_backfill_mode] += 1
                global_legacy_backfilled += 1
            if g["example"] is None:
                g["example"] = row
            global_primary[result.primary_label] += 1

    tmp_path.replace(args.output_rows_path)

    summary = []
    for cfg in sorted(grouped):
        g = grouped[cfg]
        n = max(int(g["n"]), 1)
        example = g["example"] or {}
        row = {
            "config_id": cfg,
            "n": int(g["n"]),
            "mode": example.get("mode"),
            "alpha": example.get("alpha"),
            "direction_kind": example.get("direction_kind"),
            "sparse_topk": example.get("sparse_topk"),
            "target_layers": example.get("target_layers"),
            "position_mode": example.get("position_mode"),
            "apply_phase": example.get("apply_phase"),
            "benchmark": benchmark,
            "intent_refusal_rate": g["intent"]["refusal"] / n,
            "intent_compliance_rate": g["intent"]["compliance"] / n,
            "intent_mixed_rate": g["intent"]["mixed"] / n,
            "intent_none_rate": g["intent"]["none"] / n,
            "quality_finalized_rate": g["quality"]["finalized"] / n,
            "quality_truncated_rate": g["quality"]["truncated"] / n,
            "quality_meta_only_rate": g["quality"]["meta_only"] / n,
            "quality_regurgitated_rate": g["quality"]["regurgitated"] / n,
            "quality_corrupted_rate": g["quality"]["corrupted"] / n,
            "quality_ambiguous_rate": g["quality"]["ambiguous"] / n,
        }
        for label in PRIMARY_LABELS:
            row[f"{label}_rate"] = g["primary"][label] / n
        row.update(_resolve_semantic_rate(g["primary"], n))
        row["legacy_backfilled_rate"] = g["legacy_backfilled"] / n
        row["legacy_backfill_mode_counts"] = dict(g["legacy_backfill_mode"])
        summary.append(row)

    args.summary_path.write_text(json.dumps(summary, indent=2))

    total_rows = sum(g["n"] for g in grouped.values()) or 1
    print(f"Rows relabeled: {sum(g['n'] for g in grouped.values())}")
    print(f"Benchmark mode: {benchmark}")
    print(f"Wrote rows: {args.output_rows_path}")
    print(f"Wrote summary: {args.summary_path}")
    print("Global primary counts:", dict(global_primary))
    print(
        f"Legacy-backfilled rows: {global_legacy_backfilled}/{total_rows} "
        f"({global_legacy_backfilled / total_rows:.3f})"
    )


if __name__ == "__main__":
    main()
