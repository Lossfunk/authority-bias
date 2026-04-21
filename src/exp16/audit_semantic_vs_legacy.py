"""Audit the semantic parser's labels against the legacy HarmBench classifier.

Given a pair of row files produced by run_safety_matched_conditions.py
(legacy) and relabel_harmbench_rows_semantic.py (semantic), build a confusion
matrix of response_class x semantic_primary_label, quantify where each class
leaks, and optionally dump a few representative rows per mismatch bucket.

Usage:

    uv run python -m src.exp16.audit_semantic_vs_legacy \
        --legacy-rows  neurips-results/exp16/safety_matched_qwen_harmbench80/safety_matched_rows.jsonl \
        --semantic-rows neurips-results/exp16/safety_matched_qwen_harmbench80/safety_matched_rows_semantic.jsonl \
        --output-json neurips-results/exp16/safety_matched_qwen_harmbench80/semantic_vs_legacy_audit.json \
        --samples-md  neurips-results/exp16/safety_matched_qwen_harmbench80/semantic_vs_legacy_mismatches.md \
        --samples-per-bucket 5

The script only reads; it never writes into either row file.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


LEGACY_REFUSAL_CLASSES = {"refusal", "likely_refusal_truncated"}
LEGACY_COMPLIANCE_CLASSES = {"compliance"}

SEMANTIC_REFUSAL_LABELS = {"clean_refusal", "refusal_truncated", "refusal_meta"}
SEMANTIC_COMPLIANCE_LABELS = {
    "clean_compliance",
    "compliance_truncated",
    "compliance_meta",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-rows", type=Path, required=True)
    p.add_argument("--semantic-rows", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--samples-md", type=Path, default=None)
    p.add_argument("--samples-per-bucket", type=int, default=3)
    p.add_argument(
        "--join-keys",
        default="prompt_id,condition_code",
        help="Comma-separated keys used to pair rows between the two files.",
    )
    return p.parse_args()


def _iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _row_key(row: Dict[str, Any], keys: List[str]) -> Tuple[Any, ...]:
    return tuple(row.get(k) for k in keys)


def _excerpt(text: str, n: int = 220) -> str:
    text = (text or "").strip().replace("\n", " / ")
    return text if len(text) <= n else text[:n] + "..."


def main() -> None:
    args = parse_args()
    join_keys = [k.strip() for k in args.join_keys.split(",") if k.strip()]
    if not join_keys:
        raise SystemExit("--join-keys must be non-empty.")

    legacy_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in _iter_rows(args.legacy_rows):
        legacy_by_key[_row_key(row, join_keys)] = row

    # confusion[legacy_class][semantic_label] = count
    confusion: Dict[str, Counter] = defaultdict(Counter)
    # confusion_by_condition[condition][legacy_class][semantic_label] = count
    confusion_by_condition: Dict[str, Dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    per_condition_counts: Dict[str, Counter] = defaultdict(Counter)

    # samples[(legacy_class, semantic_label)] -> list of rows
    samples: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    n_semantic = 0
    n_matched = 0
    n_unmatched = 0

    for srow in _iter_rows(args.semantic_rows):
        n_semantic += 1
        key = _row_key(srow, join_keys)
        lrow = legacy_by_key.get(key)
        if lrow is None:
            n_unmatched += 1
            continue
        n_matched += 1

        legacy_class = str(lrow.get("response_class") or "unknown")
        semantic_label = str(srow.get("semantic_primary_label") or "unknown")
        condition = str(srow.get("condition_code") or srow.get("config_id") or "all")

        confusion[legacy_class][semantic_label] += 1
        confusion_by_condition[condition][legacy_class][semantic_label] += 1
        per_condition_counts[condition][legacy_class] += 1

        bucket = (legacy_class, semantic_label)
        if len(samples[bucket]) < max(int(args.samples_per_bucket), 0):
            merged = dict(lrow)
            merged.update(
                {
                    "semantic_primary_label": srow.get("semantic_primary_label"),
                    "semantic_intent_label": srow.get("semantic_intent_label"),
                    "semantic_quality_label": srow.get("semantic_quality_label"),
                }
            )
            samples[bucket].append(merged)

    def _summarize_confusion(conf: Dict[str, Counter]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for legacy_class, counter in conf.items():
            total = sum(counter.values()) or 1
            # directional agreement: does semantic label land in same family?
            if legacy_class in LEGACY_REFUSAL_CLASSES:
                agree = sum(counter[l] for l in SEMANTIC_REFUSAL_LABELS)
            elif legacy_class in LEGACY_COMPLIANCE_CLASSES:
                agree = sum(counter[l] for l in SEMANTIC_COMPLIANCE_LABELS)
            else:
                agree = 0
            out[legacy_class] = {
                "n": total,
                "family_agree_rate": agree / total,
                "semantic_breakdown": dict(counter),
            }
        return out

    overall_summary = _summarize_confusion(confusion)
    by_condition_summary = {
        cond: _summarize_confusion(conf) for cond, conf in confusion_by_condition.items()
    }

    # Focused leak numbers that we actually care about for the writeup:
    def _leak(legacy_class: str, semantic_label: str, conf: Dict[str, Counter]) -> float:
        counter = conf.get(legacy_class, Counter())
        total = sum(counter.values()) or 1
        return counter[semantic_label] / total

    leak_metrics = {
        "overall": {
            "compliance_to_ambiguous_other_rate": _leak(
                "compliance", "ambiguous_other", confusion
            ),
            "compliance_to_any_compliance_rate": sum(
                confusion.get("compliance", Counter())[l]
                for l in SEMANTIC_COMPLIANCE_LABELS
            )
            / max(sum(confusion.get("compliance", Counter()).values()), 1),
            "refusal_to_refusal_meta_rate": _leak(
                "refusal", "refusal_meta", confusion
            ),
            "refusal_to_clean_refusal_rate": _leak(
                "refusal", "clean_refusal", confusion
            ),
        },
        "by_condition": {
            cond: {
                "compliance_to_ambiguous_other_rate": _leak(
                    "compliance", "ambiguous_other", conf
                ),
                "compliance_to_any_compliance_rate": sum(
                    conf.get("compliance", Counter())[l]
                    for l in SEMANTIC_COMPLIANCE_LABELS
                )
                / max(sum(conf.get("compliance", Counter()).values()), 1),
                "refusal_to_refusal_meta_rate": _leak(
                    "refusal", "refusal_meta", conf
                ),
                "refusal_to_clean_refusal_rate": _leak(
                    "refusal", "clean_refusal", conf
                ),
            }
            for cond, conf in confusion_by_condition.items()
        },
    }

    audit = {
        "n_semantic_rows": n_semantic,
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "join_keys": join_keys,
        "legacy_rows_path": str(args.legacy_rows),
        "semantic_rows_path": str(args.semantic_rows),
        "confusion_overall": overall_summary,
        "confusion_by_condition": by_condition_summary,
        "leak_metrics": leak_metrics,
        "per_condition_legacy_counts": {
            cond: dict(counts) for cond, counts in per_condition_counts.items()
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, default=str))

    if args.samples_md is not None and args.samples_per_bucket > 0:
        lines: List[str] = []
        lines.append(f"# Semantic vs legacy label audit — mismatch samples")
        lines.append("")
        lines.append(f"- legacy_rows: `{args.legacy_rows}`")
        lines.append(f"- semantic_rows: `{args.semantic_rows}`")
        lines.append(f"- join_keys: `{join_keys}`")
        lines.append(f"- n_matched: {n_matched} / n_semantic_rows: {n_semantic}")
        lines.append("")
        # Only show buckets where family-disagreement or unusual landing.
        for (legacy_class, semantic_label), rows in sorted(
            samples.items(), key=lambda kv: (kv[0][0], kv[0][1])
        ):
            family_ok = (
                legacy_class in LEGACY_REFUSAL_CLASSES
                and semantic_label in SEMANTIC_REFUSAL_LABELS
            ) or (
                legacy_class in LEGACY_COMPLIANCE_CLASSES
                and semantic_label in SEMANTIC_COMPLIANCE_LABELS
            )
            if family_ok:
                continue
            lines.append(f"## `response_class={legacy_class}` → `semantic={semantic_label}`")
            lines.append("")
            for r in rows:
                lines.append(f"- condition={r.get('condition_code')} prompt_id={r.get('prompt_id')}")
                lines.append(f"  - prompt: {_excerpt(r.get('prompt', ''))}")
                lines.append(f"  - response: {_excerpt(r.get('response', ''))}")
                lines.append(
                    "  - semantic: "
                    f"primary={r.get('semantic_primary_label')} "
                    f"intent={r.get('semantic_intent_label')} "
                    f"quality={r.get('semantic_quality_label')}"
                )
            lines.append("")
        args.samples_md.parent.mkdir(parents=True, exist_ok=True)
        args.samples_md.write_text("\n".join(lines))

    # Console summary
    print(f"Matched rows: {n_matched} / {n_semantic} (unmatched={n_unmatched})")
    print("Overall confusion (legacy -> semantic primary):")
    for legacy_class, info in overall_summary.items():
        print(
            f"  {legacy_class:30s} n={info['n']:5d}  "
            f"family_agree={info['family_agree_rate']:.3f}"
        )
        for lbl, cnt in sorted(info["semantic_breakdown"].items(), key=lambda kv: -kv[1]):
            print(f"      {lbl:28s} {cnt}")
    print("Leak metrics (overall):")
    for k, v in leak_metrics["overall"].items():
        print(f"  {k}: {v:.4f}")
    print("Leak metrics by condition:")
    for cond, metrics in leak_metrics["by_condition"].items():
        print(f"  [{cond}]")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

    print(f"Wrote audit JSON: {args.output_json}")
    if args.samples_md is not None:
        print(f"Wrote mismatch samples markdown: {args.samples_md}")


if __name__ == "__main__":
    main()
