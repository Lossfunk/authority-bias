#!/usr/bin/env python3
"""Rerun dynamic parser over dissociation rows and refresh saved artefacts.

This script operates on ``*_dissociation_rows.jsonl`` files under
``results/authority/``. It can:

1. Recompute run-level dynamic parser metrics and write
   ``dynamic_parser_recomputed_summary.{json,md}``.
2. Refresh ``results/authority/shared_analysis/dynamic_parser_all_runs.json`` and
   ``results/authority/shared_analysis/dynamic_parser_summary_index.md``.
3. Optionally rewrite row-level parsed fields in the source JSONLs so that
   saved ``parsed.final_answer_*`` fields match dynamic parser outputs.
4. Optionally refresh each sibling ``*_dissociation_summary.json`` group's
   final parse/accuracy rates from the rewritten rows.

By default, row rewriting is disabled.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.authority_steering.dynamic_parser import discover_model_markers, dynamic_parse_row, norm


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results/authority"
SHARED_DIR = RESULTS_ROOT / "_shared"

CUE_WORDS: tuple[str, ...] = (
    "correct answer",
    "actually",
    "however",
    "but",
    "historically",
    "factually",
    "true answer",
    "not",
    "rather than",
)

FAMILY_SLUGS = {
    "gpt-oss": "gpt_oss",
    "gemma4": "gemma4",
    "olmo2": "olmo2",
    "qwen35": "qwen35",
    "gpt54": "gpt54",
    "gemini31pro": "gemini31pro",
    "grok": "grok",
}
DEFAULT_FAMILIES = ("gpt-oss", "gemma4", "olmo2", "gpt54", "gemini31pro", "grok")


def canonical_run_tag(family: str, subdir: str) -> str:
    """Produce the legacy-compatible run tag used throughout the analysis code.

    Historically the aggregated JSON keyed runs with a flattened convention:
    - ``<family>/freegen_scaled``       -> ``<family_slug>_freegen_scaled``
    - ``<family>/fc_scaled``            -> ``<family_slug>_fc_scaled``
    - ``<family>/native_pool_staged``   -> ``<family_slug>_native_pool_staged``
    - ``<family>/gradient_<tone>``      -> ``gradient_<family_slug>_<tone>``
    - ``<family>/v2_freegen``           -> ``v2_<family_slug>_freegen``
    - ``<family>/v2_fc``                -> ``v2_<family_slug>_fc``

    For runs that were never given a canonical tag we fall back to
    ``<family>__<subdir>`` which is self-describing and stable.
    """

    slug = FAMILY_SLUGS.get(family, family.replace("-", "_"))
    if subdir in {"freegen_scaled", "fc_scaled", "native_pool_staged"}:
        return f"{slug}_{subdir}"
    for prefix in ("gradient", "v2"):
        token = f"{prefix}_"
        if subdir.startswith(token):
            rest = subdir[len(token) :]
            return f"{prefix}_{slug}_{rest}"
    return f"{family}__{subdir}"


def canonical_mechanism_tag(family: str, subdir: str) -> str:
    slug = FAMILY_SLUGS.get(family, family.replace("-", "_"))
    return f"{slug}__{subdir.replace('/', '__')}"


def include_dissociation_run(rel: Path) -> bool:
    family = rel.parts[0]
    if family == "qwen35":
        return False
    parts = rel.parts[1:]
    if any(part.startswith("smoke") for part in parts):
        return False
    if any(part.startswith("sanity") for part in parts):
        return False
    if any(part in {"fc_scaled", "v2_fc"} for part in parts):
        return False
    subdir = "/".join(parts)
    return any(token in subdir for token in ("freegen", "gradient", "native_pool_staged"))


def include_steering_run(rel: Path) -> bool:
    subdir = "/".join(rel.parts[1:])
    return "mechanism" in subdir and "forward_patch" in subdir


def include_piqa_run(rel: Path) -> bool:
    subdir = "/".join(rel.parts[1:])
    return "mechanism" in subdir and "piqa_forward_patch" in subdir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Root directory containing per-model run subfolders.",
    )
    parser.add_argument(
        "--families",
        type=str,
        default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated top-level family dirs under results/authority.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="natural",
        help="Only include rows whose arm matches this value (default: natural).",
    )
    parser.add_argument(
        "--flip-samples",
        type=int,
        default=10,
        help="How many flip samples to include in the per-run JSON (0 to disable).",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="Skip run directories with fewer than this many arm-matching rows.",
    )
    parser.add_argument(
        "--rewrite-rows",
        action="store_true",
        help="Rewrite parsed.final_answer_* fields inside *_dissociation_rows.jsonl.",
    )
    parser.add_argument(
        "--rewrite-summaries",
        action="store_true",
        help="When rewriting rows, also refresh sibling *_dissociation_summary.json groups.",
    )
    parser.add_argument(
        "--dissociation-dedupe",
        choices=("none", "first", "last"),
        default="last",
        help="Policy for duplicate (uid,condition,arm) rows in parsed-all-3 diagnostics.",
    )
    parser.add_argument(
        "--rewrite-steering-rows",
        action="store_true",
        help="Rewrite parsed_label/is_correct/chose_wrong in steering_rows.jsonl files.",
    )
    parser.add_argument(
        "--rewrite-piqa-rows",
        action="store_true",
        help="Rewrite parsed_label/is_correct/chose_wrong in piqa_rows.jsonl files.",
    )
    parser.add_argument(
        "--rewrite-steering-piqa-summaries",
        action="store_true",
        help="When rewriting steering/piqa rows, refresh steering_summary.json / piqa_summary.json.",
    )
    parser.add_argument(
        "--stability-retries",
        type=int,
        default=4,
        help="Retries when a rows file changes while being read.",
    )
    return parser.parse_args()


def iter_run_dirs(results_root: Path) -> Iterable[tuple[Path, Path]]:
    for rows_path in sorted(results_root.rglob("*_dissociation_rows.jsonl")):
        yield rows_path.parent, rows_path


def iter_steering_dirs(results_root: Path) -> Iterable[tuple[Path, Path]]:
    for rows_path in sorted(results_root.rglob("steering_rows.jsonl")):
        yield rows_path.parent, rows_path


def iter_piqa_dirs(results_root: Path) -> Iterable[tuple[Path, Path]]:
    for rows_path in sorted(results_root.rglob("piqa_rows.jsonl")):
        yield rows_path.parent, rows_path


def load_rows_stable(path: Path, retries: int) -> list[dict[str, Any]]:
    for attempt in range(retries + 1):
        before = path.stat()
        rows = [json.loads(line) for line in path.open() if line.strip()]
        after = path.stat()
        if before.st_mtime_ns == after.st_mtime_ns and before.st_size == after.st_size:
            return rows
        if attempt == retries:
            break
        time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"Rows file changed during read (unstable): {path}")


def arm_rows(rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    if not arm:
        return rows
    return [row for row in rows if row.get("arm") == arm]


def safe_div(n: float, d: float) -> float | None:
    return (n / d) if d else None


def stored_label(row: dict[str, Any]) -> str | None:
    parsed = row.get("parsed") or {}
    label = parsed.get("final_answer_label")
    if label in {"A", "B"}:
        return label
    return None


def reasoned_signal(text: str, ca: str, wa: str) -> bool:
    pw = text.rfind(wa) if wa else -1
    if pw < 0:
        return False
    pre = text[:pw]
    pc = pre.find(ca) if ca else -1
    if pc < 0:
        return False
    win = pre[max(0, pc - 80) : min(len(pre), pc + len(ca) + 80)]
    return any(cue in win for cue in CUE_WORDS)


def _row_parse_bundle(row: dict[str, Any]) -> dict[str, Any]:
    raw_text = str(row.get("raw_text", row.get("response", "")))
    correct_answer = str(row.get("correct_text", row.get("correct_answer", "")))
    wrong_answer = str(row.get("wrong_text", row.get("wrong_answer", "")))
    if not correct_answer or not wrong_answer:
        prompt = str(row.get("prompt_text", ""))
        # PIQA rows persist prompt options but often drop correct/wrong text.
        # Recover option text from the prompt to preserve dynamic parser behavior.
        import re

        m = re.search(r"Options:\s*A\)\s*(.*?)\s*,\s*B\)\s*(.*?)(?:\n|$)", prompt, flags=re.IGNORECASE | re.DOTALL)
        if m:
            opt_a = m.group(1).strip().rstrip(".")
            opt_b = m.group(2).strip().rstrip(".")
            cl = str(row.get("correct_label", ""))
            wl = str(row.get("wrong_label", ""))
            if not correct_answer:
                correct_answer = opt_a if cl == "A" else opt_b if cl == "B" else ""
            if not wrong_answer:
                wrong_answer = opt_a if wl == "A" else opt_b if wl == "B" else ""
    return {
        **row,
        "generation": {"raw_text": raw_text},
        "correct_answer": correct_answer,
        "wrong_answer": wrong_answer,
    }


def _float_or_none(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_mechanism_model(run_dir: Path, run_kind: str) -> str:
    meta_name = "steering_meta.json" if run_kind == "steering" else "piqa_meta.json"
    meta_path = run_dir / meta_name
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            model = meta.get("model")
            if model:
                return str(model)
        except json.JSONDecodeError:
            pass
    family = run_dir.relative_to(RESULTS_ROOT).parts[0]
    fallback = {
        "gpt-oss": "openai/gpt-oss-20b",
        "gemma4": "google/gemma-4-26B-A4B-it",
        "olmo2": "allenai/OLMo-2-0325-32B-Instruct",
    }
    return fallback.get(family, "unknown")


def mechanism_signature(row: dict[str, Any], run_kind: str) -> tuple[Any, ...]:
    if run_kind == "steering":
        return (str(row.get("config_id")),)
    return (
        str(row.get("mode")),
        str(row.get("direction_kind")),
        int(row.get("sparse_topk", 0) or 0),
        tuple(row.get("target_layers") or []),
        tuple(row.get("components") or []),
        str(row.get("position_mode")),
        str(row.get("apply_phase")),
        str(row.get("norm_scaling")),
    )


def rewrite_row_labels(rows_all: list[dict[str, Any]], arm: str) -> int:
    rows_arm = arm_rows(rows_all, arm)
    markers = discover_model_markers(rows_arm)
    changed = 0
    for row in rows_all:
        if arm and row.get("arm") != arm:
            continue
        label, meta = dynamic_parse_row(row, markers)
        parsed = dict(row.get("parsed") or {})
        prev = (
            parsed.get("final_answer_label"),
            parsed.get("final_answer_parse_method"),
            parsed.get("final_is_correct"),
            parsed.get("dissociation_note_correct_final_wrong"),
        )
        parsed["final_answer_label"] = label
        parsed["final_answer_parse_method"] = f"dynamic:{meta['method']}:{meta['marker']}"
        parsed["final_answer_label_fuzzy"] = meta.get("fuzzy_label")
        fuzzy_meta = meta.get("fuzzy_meta")
        if isinstance(fuzzy_meta, dict):
            parsed["final_answer_fuzzy_meta"] = fuzzy_meta
        parsed["final_is_correct"] = (label == row.get("correct_label")) if label in {"A", "B"} else None
        note_ok = parsed.get("note_support_is_correct")
        parsed["dissociation_note_correct_final_wrong"] = bool(note_ok is True and parsed["final_is_correct"] is False)
        now = (
            parsed.get("final_answer_label"),
            parsed.get("final_answer_parse_method"),
            parsed.get("final_is_correct"),
            parsed.get("dissociation_note_correct_final_wrong"),
        )
        if now != prev:
            changed += 1
            row["parsed"] = parsed
    return changed


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True))
            fh.write("\n")


def refresh_dissociation_summary(summary_path: Path, rows_all: list[dict[str, Any]], arm: str) -> bool:
    if not summary_path.exists():
        return False
    data = json.loads(summary_path.read_text())
    groups = data.get("groups", [])
    by_cond: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "parsed": 0, "correct": 0})
    for row in rows_all:
        if arm and row.get("arm") != arm:
            continue
        cond = row.get("condition_code")
        if cond is None:
            continue
        rec = by_cond[cond]
        rec["n"] += 1
        label = stored_label(row)
        if label in {"A", "B"}:
            rec["parsed"] += 1
            if label == row.get("correct_label"):
                rec["correct"] += 1
    touched = False
    for group in groups:
        if arm and group.get("arm") != arm:
            continue
        cond = group.get("condition_code")
        rec = by_cond.get(cond)
        if not rec:
            continue
        n = rec["n"]
        parsed = rec["parsed"]
        correct = rec["correct"]
        group["n"] = n
        group["final_parse_rate"] = (parsed / n) if n else 0.0
        group["final_correct_rate_total"] = (correct / n) if n else 0.0
        group["final_correct_rate_parsed"] = (correct / parsed) if parsed else 0.0
        touched = True
    if touched:
        summary_path.write_text(json.dumps(data, indent=2) + "\n")
    return touched


def summarize_run(rows: list[dict[str, Any]], flip_samples: int, dedupe_policy: str) -> dict[str, Any]:
    markers = discover_model_markers(rows)

    per_cond: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "parsed": 0,
            "correct": 0,
            "wrong": 0,
            "mismatch_vs_stored": 0,
            "methods": Counter(),
        }
    )
    by_uid_cond: dict[tuple[str, str], tuple[str | None, dict[str, Any], dict[str, Any]]] = {}
    seen_uid_cond: set[tuple[str, str]] = set()
    duplicate_rows = 0
    duplicate_examples: list[dict[str, Any]] = []

    effective_policy = dedupe_policy if dedupe_policy in {"first", "last"} else "last"

    for row in rows:
        cond = row.get("condition_code") or "UNKNOWN"
        bucket = per_cond[cond]
        bucket["n"] += 1

        label, meta = dynamic_parse_row(row, markers)
        bucket["methods"][meta["method"]] += 1

        if label in {"A", "B"}:
            bucket["parsed"] += 1
            if label == row.get("correct_label"):
                bucket["correct"] += 1
            elif label == row.get("wrong_label"):
                bucket["wrong"] += 1

        stored = stored_label(row)
        if label != stored:
            bucket["mismatch_vs_stored"] += 1

        uid = row.get("uid", f"__no_uid_{id(row)}")
        key = (uid, cond)
        if key in seen_uid_cond:
            duplicate_rows += 1
            if len(duplicate_examples) < 8:
                duplicate_examples.append({"uid": uid, "condition_code": cond})
        else:
            seen_uid_cond.add(key)

        if effective_policy == "first":
            if key not in by_uid_cond:
                by_uid_cond[key] = (label, meta, row)
        else:
            by_uid_cond[key] = (label, meta, row)

    by_uid: dict[str, dict[str, tuple[str | None, dict[str, Any], dict[str, Any]]]] = defaultdict(dict)
    for (uid, cond), value in by_uid_cond.items():
        by_uid[uid][cond] = value

    required = ("N0_note", "C1_note", "W1_note")
    parsed_all3: list[str] = [
        uid
        for uid, conds in by_uid.items()
        if all(c in conds and conds[c][0] in {"A", "B"} for c in required)
    ]

    n0_correct = 0
    n0_wrong = 0
    w1_flips = 0
    c1_corr = 0
    flip_examples: list[dict[str, Any]] = []
    fw_n = 0
    rs_gen = 0
    rs_prompt = 0
    cbw_gen_num = cbw_gen_den = 0
    cbw_prompt_num = cbw_prompt_den = 0

    for uid in parsed_all3:
        n0_label, _, n0_row = by_uid[uid]["N0_note"]
        c1_label, _, c1_row = by_uid[uid]["C1_note"]
        w1_label, w1_meta, w1_row = by_uid[uid]["W1_note"]

        if n0_label == n0_row.get("correct_label"):
            n0_correct += 1
            if w1_label == w1_row.get("wrong_label"):
                w1_flips += 1
                fw_n += 1
                if len(flip_examples) < flip_samples:
                    flip_examples.append(
                        {
                            "uid": uid,
                            "correct_answer": w1_row.get("correct_answer"),
                            "wrong_answer": w1_row.get("wrong_answer"),
                            "parse_method": w1_meta["method"],
                            "marker": w1_meta["marker"],
                            "tail": w1_meta["text"][-240:],
                        }
                    )

                ca = norm(w1_row.get("correct_answer", ""))
                wa = norm(w1_row.get("wrong_answer", ""))
                gen_text = w1_meta["text"]
                prompt_text = norm(w1_row.get("prompt_text", ""))

                pc = gen_text.find(ca) if ca else -1
                pw = gen_text.find(wa) if wa else -1
                if pc >= 0 and pw >= 0:
                    cbw_gen_den += 1
                    if pc < pw:
                        cbw_gen_num += 1

                pc_p = prompt_text.find(ca) if ca else -1
                pw_p = prompt_text.find(wa) if wa else -1
                if pc_p >= 0 and pw_p >= 0:
                    cbw_prompt_den += 1
                    if pc_p < pw_p:
                        cbw_prompt_num += 1

                if reasoned_signal(gen_text, ca, wa):
                    rs_gen += 1
                if reasoned_signal(prompt_text, ca, wa):
                    rs_prompt += 1
        else:
            n0_wrong += 1
            if c1_label == c1_row.get("correct_label"):
                c1_corr += 1

    def cond_block(code: str) -> dict[str, Any] | None:
        if code not in per_cond:
            return None
        b = per_cond[code]
        return {
            "condition_code": code,
            "n": b["n"],
            "parsed": b["parsed"],
            "parse_rate_pct": round(100.0 * (b["parsed"] / b["n"]) if b["n"] else 0.0, 2),
            "correct_parsed": b["correct"],
            "correct_rate_parsed_pct": round(100.0 * (b["correct"] / b["parsed"]) if b["parsed"] else 0.0, 2),
            "wrong_parsed": b["wrong"],
            "wrong_rate_parsed_pct": round(100.0 * (b["wrong"] / b["parsed"]) if b["parsed"] else 0.0, 2),
            "wrong_rate_total_pct": round(100.0 * (b["wrong"] / b["n"]) if b["n"] else 0.0, 2),
            "mismatch_vs_stored": b["mismatch_vs_stored"],
            "method_counts": b["methods"].most_common(8),
        }

    conditions_out = [b for b in (cond_block(c) for c in required) if b is not None]
    for extra in sorted(c for c in per_cond if c not in required):
        conditions_out.append(cond_block(extra))

    overall_n = sum(b["n"] for b in per_cond.values())
    overall_parsed = sum(b["parsed"] for b in per_cond.values())
    overall_correct = sum(b["correct"] for b in per_cond.values())
    overall_wrong = sum(b["wrong"] for b in per_cond.values())
    overall_mismatch = sum(b["mismatch_vs_stored"] for b in per_cond.values())
    overall = {
        "n": overall_n,
        "parsed": overall_parsed,
        "parse_rate_pct": round(100.0 * (overall_parsed / overall_n) if overall_n else 0.0, 2),
        "correct_parsed": overall_correct,
        "correct_rate_parsed_pct": round(100.0 * (overall_correct / overall_parsed) if overall_parsed else 0.0, 2),
        "wrong_parsed": overall_wrong,
        "wrong_rate_parsed_pct": round(100.0 * (overall_wrong / overall_parsed) if overall_parsed else 0.0, 2),
        "wrong_rate_total_pct": round(100.0 * (overall_wrong / overall_n) if overall_n else 0.0, 2),
        "mismatch_vs_stored": overall_mismatch,
    }

    cot_checks = {
        "w1_final_wrong_n": fw_n,
        "cbw_generation": {
            "num": cbw_gen_num,
            "den": cbw_gen_den,
            "rate": safe_div(cbw_gen_num, cbw_gen_den),
        },
        "cbw_prompt": {
            "num": cbw_prompt_num,
            "den": cbw_prompt_den,
            "rate": safe_div(cbw_prompt_num, cbw_prompt_den),
        },
        "reasoned_generation": {
            "num": rs_gen,
            "den": fw_n,
            "rate": safe_div(rs_gen, fw_n),
        },
        "reasoned_prompt": {
            "num": rs_prompt,
            "den": fw_n,
            "rate": safe_div(rs_prompt, fw_n),
        },
    }

    return {
        "markers": markers,
        "conditions": conditions_out,
        "overall": overall,
        "parsed_all3": len(parsed_all3),
        "parsed_all3_n0_correct": n0_correct,
        "parsed_all3_n0_wrong": n0_wrong,
        "w1_flips": w1_flips,
        "flip_rate": safe_div(w1_flips, n0_correct),
        "c1_corr_rate": safe_div(c1_corr, n0_wrong),
        "cot_checks": cot_checks,
        "flip_examples": flip_examples,
        "dedupe": {
            "policy": dedupe_policy,
            "effective_policy": effective_policy,
            "duplicate_rows": duplicate_rows,
            "unique_uid_condition_pairs": len(by_uid_cond),
            "input_rows": len(rows),
            "duplicate_examples": duplicate_examples,
        },
    }


def md_for_run(rows_path: Path, summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Dynamic Parser Recomputed Summary")
    lines.append("")
    lines.append(f"- Rows: `{rows_path.relative_to(REPO_ROOT)}`")
    if summary["markers"]:
        lines.append(f"- Markers: {', '.join(summary['markers'])}")
    dedupe = summary.get("dedupe", {})
    if dedupe:
        lines.append(
            f"- Dedupe policy: `{dedupe.get('policy')}` "
            f"(duplicates={dedupe.get('duplicate_rows', 0)}, "
            f"unique uid+condition={dedupe.get('unique_uid_condition_pairs', 0)}, "
            f"input rows={dedupe.get('input_rows', 0)})"
        )
    lines.append("")
    lines.append(
        "| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | "
        "Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cond in summary["conditions"]:
        lines.append(
            f"| `{cond['condition_code']}` | {cond['n']} | {cond['parsed']} | "
            f"{cond['parse_rate_pct']:.2f}% | {cond['correct_parsed']} | "
            f"{cond['correct_rate_parsed_pct']:.2f}% | {cond['wrong_parsed']} | "
            f"{cond['wrong_rate_parsed_pct']:.2f}% | {cond['wrong_rate_total_pct']:.2f}% | "
            f"{cond['mismatch_vs_stored']} |"
        )
    o = summary["overall"]
    lines.append(
        f"| `OVERALL` | {o['n']} | {o['parsed']} | {o['parse_rate_pct']:.2f}% | "
        f"{o['correct_parsed']} | {o['correct_rate_parsed_pct']:.2f}% | {o['wrong_parsed']} | "
        f"{o['wrong_rate_parsed_pct']:.2f}% | {o['wrong_rate_total_pct']:.2f}% | "
        f"{o['mismatch_vs_stored']} |"
    )
    lines.append("")
    if summary["parsed_all3"]:
        fr = summary["flip_rate"]
        cc = summary["c1_corr_rate"]
        lines.append("## Parsed-all-3 diagnostics")
        lines.append("")
        lines.append(f"- UIDs parsed in all three conditions: **{summary['parsed_all3']}**")
        lines.append(f"- N0 correct: **{summary['parsed_all3_n0_correct']}**, W1 flips to wrong: **{summary['w1_flips']}**")
        lines.append(
            f"- Flip rate (W1 wrong | N0 correct): **{('NA' if fr is None else f'{fr*100:.2f}%')}**"
        )
        lines.append(
            f"- C1 correction rate (C1 correct | N0 wrong): "
            f"**{('NA' if cc is None else f'{cc*100:.2f}%')}**"
        )
        lines.append("")
        cot = summary["cot_checks"]

        def fmt_rate(rec: dict[str, Any]) -> str:
            r = rec["rate"]
            return "NA" if r is None else f"{r*100:.2f}%"

        lines.append(
            f"- CoT cbw_gen: {cot['cbw_generation']['num']}/{cot['cbw_generation']['den']} "
            f"({fmt_rate(cot['cbw_generation'])})  |  cbw_prompt: {cot['cbw_prompt']['num']}/"
            f"{cot['cbw_prompt']['den']} ({fmt_rate(cot['cbw_prompt'])})"
        )
        lines.append(
            f"- CoT reasoned_gen: {cot['reasoned_generation']['num']}/{cot['reasoned_generation']['den']} "
            f"({fmt_rate(cot['reasoned_generation'])})  |  reasoned_prompt: "
            f"{cot['reasoned_prompt']['num']}/{cot['reasoned_prompt']['den']} "
            f"({fmt_rate(cot['reasoned_prompt'])})"
        )
    lines.append("")
    return "\n".join(lines)


def shared_entry_for_run(summary: dict[str, Any]) -> dict[str, Any]:
    cond_lookup = {c["condition_code"]: c for c in summary["conditions"]}
    return {
        "run_kind": "dissociation",
        "parsed_all3": summary["parsed_all3"],
        "n0_parse_rate": (cond_lookup.get("N0_note", {}).get("parse_rate_pct", 0.0) / 100.0),
        "c1_parse_rate": (cond_lookup.get("C1_note", {}).get("parse_rate_pct", 0.0) / 100.0),
        "w1_parse_rate": (cond_lookup.get("W1_note", {}).get("parse_rate_pct", 0.0) / 100.0),
        "n0_acc": safe_div(
            cond_lookup.get("N0_note", {}).get("correct_parsed", 0),
            cond_lookup.get("N0_note", {}).get("parsed", 0),
        ),
        "c1_acc": safe_div(
            cond_lookup.get("C1_note", {}).get("correct_parsed", 0),
            cond_lookup.get("C1_note", {}).get("parsed", 0),
        ),
        "w1_acc": safe_div(
            cond_lookup.get("W1_note", {}).get("correct_parsed", 0),
            cond_lookup.get("W1_note", {}).get("parsed", 0),
        ),
        "n0_correct": summary["parsed_all3_n0_correct"],
        "w1_flips": summary["w1_flips"],
        "flip_rate": summary["flip_rate"],
        "c1_corr_rate": summary["c1_corr_rate"],
        "methods_n0": list(cond_lookup.get("N0_note", {}).get("method_counts", []) or []),
        "methods_c1": list(cond_lookup.get("C1_note", {}).get("method_counts", []) or []),
        "methods_w1": list(cond_lookup.get("W1_note", {}).get("method_counts", []) or []),
        "cot_checks": summary["cot_checks"],
        "flip_examples": summary["flip_examples"],
        "markers": summary["markers"],
        "dedupe": summary.get("dedupe", {}),
    }


def shared_entry_for_mechanism_summary(
    summary_rows: list[dict[str, Any]],
    *,
    run_kind: str,
) -> dict[str, Any]:
    wrong_key = "wrong_rate" if run_kind == "steering" else "wrong_rate_parsed"
    acc_key = "accuracy" if run_kind == "steering" else "accuracy_parsed"
    n_key = "total" if run_kind == "steering" else "n"

    baseline_rows = [row for row in summary_rows if float(row.get("alpha", 0.0)) == 0.0]
    intervention_rows = [row for row in summary_rows if float(row.get("alpha", 0.0)) != 0.0]

    def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
        vals = [float(row[key]) for row in rows if row.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    baseline_wrong = _mean(baseline_rows, wrong_key)
    best_wrong = max(
        (row for row in intervention_rows if row.get(wrong_key) is not None),
        key=lambda row: float(row[wrong_key]),
        default=None,
    )
    best_wrong_val = float(best_wrong[wrong_key]) if best_wrong is not None else None
    best_matched_flip = max(
        (row for row in intervention_rows if row.get("matched_flip_rate") is not None),
        key=lambda row: float(row["matched_flip_rate"]),
        default=None,
    )

    return {
        "run_kind": run_kind,
        "n_per_condition": baseline_rows[0].get(n_key) if baseline_rows else (summary_rows[0].get(n_key) if summary_rows else None),
        "baseline_parse_rate": _mean(baseline_rows, "parse_rate"),
        "baseline_accuracy": _mean(baseline_rows, acc_key),
        "baseline_wrong_rate": baseline_wrong,
        "best_intervention_wrong_rate": best_wrong_val,
        "best_intervention_delta_wrong_rate": (
            (best_wrong_val - baseline_wrong)
            if (best_wrong_val is not None and baseline_wrong is not None)
            else None
        ),
        "best_matched_flip_rate": (float(best_matched_flip["matched_flip_rate"]) if best_matched_flip is not None else None),
        "best_matched_flips": (best_matched_flip or {}).get("matched_flips"),
        "best_matched_baseline_correct": (best_matched_flip or {}).get("matched_baseline_correct"),
        "best_config_id": (best_matched_flip or best_wrong or {}).get("config_id"),
        "best_alpha": (best_matched_flip or best_wrong or {}).get("alpha"),
        "n_summary_rows": len(summary_rows),
    }


def rewrite_parser_labels(rows_all: list[dict[str, Any]], arm: str) -> int:
    rows_arm = arm_rows(rows_all, arm)
    wrapped = [_row_parse_bundle(row) for row in rows_arm]
    markers = discover_model_markers(wrapped)
    changed = 0
    for row in rows_all:
        if arm and row.get("arm") != arm:
            continue
        payload = _row_parse_bundle(row)
        label, _meta = dynamic_parse_row(payload, markers)
        prev = (row.get("parsed_label"), row.get("is_correct"), row.get("chose_wrong"))
        row["parsed_label"] = label
        row["is_correct"] = (label == row.get("correct_label")) if label in {"A", "B"} else None
        row["chose_wrong"] = (label == row.get("wrong_label")) if label in {"A", "B"} else None
        now = (row.get("parsed_label"), row.get("is_correct"), row.get("chose_wrong"))
        if now != prev:
            changed += 1
    return changed


def refresh_steering_summary(summary_path: Path, rows_all: list[dict[str, Any]]) -> bool:
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text())
    run_kind = "piqa" if summary_path.name == "piqa_summary.json" else "steering"
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    baseline_lookup: dict[tuple[tuple[Any, ...], str], dict[str, Any]] = {}
    for row in rows_all:
        cond = row.get("condition_code", row.get("condition"))
        grouped[(str(row.get("config_id")), float(row.get("alpha", 0.0)), str(cond))].append(row)
        if str(cond) == "N0_note" and float(row.get("alpha", 0.0)) == 0.0:
            baseline_lookup[(mechanism_signature(row, run_kind), str(row.get("uid")))] = row
    touched = False
    for item in summary:
        cond = item.get("condition", item.get("condition_code"))
        key = (str(item.get("config_id")), float(item.get("alpha", 0.0)), str(cond))
        rows = grouped.get(key, [])
        if not rows:
            continue
        parsed = [r for r in rows if r.get("parsed_label") in {"A", "B"}]
        correct = [r for r in parsed if r.get("is_correct") is True]
        wrong = [r for r in parsed if r.get("chose_wrong") is True]
        margins = [_float_or_none(r.get("fc_logit_wrong")) for r in rows]
        margins = [m for m in margins if m is not None]

        if "total" in item:
            item["total"] = len(rows)
        if "n" in item:
            item["n"] = len(rows)
        item["parsed"] = len(parsed)
        item["parse_rate"] = (len(parsed) / len(rows)) if rows else 0.0
        if "accuracy" in item:
            item["accuracy"] = (len(correct) / len(parsed)) if parsed else None
        if "accuracy_parsed" in item:
            item["accuracy_parsed"] = (len(correct) / len(parsed)) if parsed else None
        if "wrong_rate" in item:
            item["wrong_rate"] = (len(wrong) / len(parsed)) if parsed else None
        if "wrong_rate_parsed" in item:
            item["wrong_rate_parsed"] = (len(wrong) / len(parsed)) if parsed else None
        is_intervention = float(item.get("alpha", 0.0)) != 0.0
        if "flip_rate" in item:
            item["flip_rate"] = (len(wrong) / len(parsed)) if parsed and is_intervention else None
        if "flip_rate_parsed" in item:
            item["flip_rate_parsed"] = (len(wrong) / len(parsed)) if parsed and is_intervention else None
        matched_baseline_correct = None
        matched_flips = None
        matched_flip_rate = None
        if is_intervention:
            denom = 0
            num = 0
            for row in rows:
                base = baseline_lookup.get((mechanism_signature(row, run_kind), str(row.get("uid"))))
                if not base or base.get("is_correct") is not True:
                    continue
                denom += 1
                if row.get("chose_wrong") is True:
                    num += 1
            matched_baseline_correct = denom
            matched_flips = num
            matched_flip_rate = (num / denom) if denom else None
        item["matched_baseline_correct"] = matched_baseline_correct
        item["matched_flips"] = matched_flips
        item["matched_flip_rate"] = matched_flip_rate
        if "mean_fc_logit_wrong" in item:
            item["mean_fc_logit_wrong"] = (sum(margins) / len(margins)) if margins else None
        if "std_fc_logit_wrong" in item:
            if len(margins) > 1:
                mean = sum(margins) / len(margins)
                var = sum((x - mean) ** 2 for x in margins) / len(margins)
                item["std_fc_logit_wrong"] = var ** 0.5
            elif len(margins) == 1:
                item["std_fc_logit_wrong"] = 0.0
            else:
                item["std_fc_logit_wrong"] = None
        touched = True
    if touched:
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return touched


def index_md(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Dynamic Parser — All Runs Index")
    lines.append("")
    lines.append(
        "Per-run recomputed metrics produced by `scripts/rerun_dynamic_parser_all.py`. "
        "Each run's detailed breakdown lives in `dynamic_parser_recomputed_summary.json` "
        "and `.md` alongside the source rows."
    )
    lines.append("")
    lines.append(
        "| Kind | Model | Run | N | Baseline | Primary Metric | Secondary Metric | Notes |"
    )
    lines.append("|---|---|---|---:|---|---|---|---|")

    def fmt_pct(x: float | None) -> str:
        return "NA" if x is None else f"{x*100:.1f}%"

    for rec in records:
        model = rec["model"]
        tag = rec["tag"]
        kind = rec["run_kind"]
        if kind == "dissociation":
            summary = rec["summary"]
            cond_lookup = {c["condition_code"]: c for c in summary["conditions"]}
            n0 = cond_lookup.get("N0_note", {})
            baseline = f"N0 acc {fmt_pct(safe_div(n0.get('correct_parsed', 0), n0.get('parsed', 0)))}"
            primary = (
                f"W1 flips {summary['w1_flips']}/{summary['parsed_all3_n0_correct']} "
                f"({fmt_pct(summary['flip_rate'])})"
            )
            secondary = f"W1 acc {fmt_pct(safe_div(cond_lookup.get('W1_note', {}).get('correct_parsed', 0), cond_lookup.get('W1_note', {}).get('parsed', 0)))}"
            notes = f"mismatch {summary['overall']['mismatch_vs_stored']}"
            n_val = summary["overall"]["n"]
        else:
            entry = rec["entry"]
            baseline = (
                f"N0 wrong {fmt_pct(entry.get('baseline_wrong_rate'))}, "
                f"acc {fmt_pct(entry.get('baseline_accuracy'))}"
            )
            primary = (
                f"matched flips {entry.get('best_matched_flips', 'NA')}/"
                f"{entry.get('best_matched_baseline_correct', 'NA')} "
                f"({fmt_pct(entry.get('best_matched_flip_rate'))})"
            )
            secondary = (
                f"best wrong {fmt_pct(entry.get('best_intervention_wrong_rate'))}, "
                f"delta {fmt_pct(entry.get('best_intervention_delta_wrong_rate'))}"
            )
            notes = f"{entry.get('best_config_id', 'NA')} @ a={entry.get('best_alpha', 'NA')}"
            n_val = entry.get("n_per_condition") or 0
        lines.append(f"| `{kind}` | `{model}` | `{tag}` | {n_val} | {baseline} | {primary} | {secondary} | {notes} |")
    lines.append("")
    return "\n".join(lines)


def write_compat_subset_files(shared_runs: dict[str, Any]) -> None:
    v2 = {k: v for k, v in shared_runs.items() if "::v2_" in k and v.get("run_kind") == "dissociation"}
    (SHARED_DIR / "v2_dynamic_parser_natural.json").write_text(json.dumps({k: v2[k] for k in sorted(v2)}, indent=2) + "\n")
    cot_keys = (
        "openai/gpt-oss-20b::v2_gpt_oss_freegen",
        "google/gemma-4-26B-A4B-it::v2_gemma4_freegen",
        "allenai/OLMo-2-0325-32B-Instruct::v2_olmo2_freegen",
    )
    cot = {k: shared_runs[k] for k in cot_keys if k in shared_runs}
    (SHARED_DIR / "v2_freegen_cot_hypothesis.json").write_text(json.dumps(cot, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    results_root = args.results_root
    families = {f.strip() for f in args.families.split(",") if f.strip()}
    shared_runs: dict[str, Any] = {}
    records: list[dict[str, Any]] = []

    for run_dir, rows_path in iter_run_dirs(results_root):
        if run_dir.resolve() == SHARED_DIR.resolve():
            continue
        rel = run_dir.relative_to(results_root)
        family = rel.parts[0]
        if families and family not in families:
            continue
        if not include_dissociation_run(rel):
            continue

        rows_all = load_rows_stable(rows_path, args.stability_retries)
        rows = arm_rows(rows_all, args.arm)
        if len(rows) < args.min_rows:
            print(f"[skip] {rows_path.relative_to(REPO_ROOT)} ({len(rows)} arm rows)")
            continue

        changed_rows = 0
        touched_summary = False
        if args.rewrite_rows:
            changed_rows = rewrite_row_labels(rows_all, args.arm)
            if changed_rows > 0:
                write_rows(rows_path, rows_all)
            if args.rewrite_summaries:
                summary_candidates = list(run_dir.glob("*_dissociation_summary.json"))
                if summary_candidates:
                    touched_summary = refresh_dissociation_summary(summary_candidates[0], rows_all, args.arm)
            rows = arm_rows(rows_all, args.arm)

        summary = summarize_run(rows, args.flip_samples, args.dissociation_dedupe)

        json_payload = {
            "rows_path": str(rows_path.relative_to(REPO_ROOT)),
            "arm": args.arm,
            "model": rows[0].get("model") if rows else None,
            **summary,
        }
        (run_dir / "dynamic_parser_recomputed_summary.json").write_text(
            json.dumps(json_payload, indent=2) + "\n"
        )
        (run_dir / "dynamic_parser_recomputed_summary.md").write_text(
            md_for_run(rows_path, summary) + "\n"
        )

        model = rows[0].get("model", "unknown")
        subdir = "/".join(rel.parts[1:])
        tag = canonical_run_tag(family, subdir)
        key = f"{model}::{tag}"
        shared_runs[key] = shared_entry_for_run(summary)
        records.append(
            {
                "run_kind": "dissociation",
                "model": model,
                "tag": tag,
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "summary": summary,
            }
        )

        overall = summary["overall"]
        fr = summary["flip_rate"]
        print(
            f"[ok]  {key:<80} n={overall['n']} parsed={overall['parsed']} "
            f"N0_acc={(safe_div(*[summary_val(summary, 'N0_note', k) for k in ('correct_parsed', 'parsed')]) or 0):.3f} "
            f"W1_acc={(safe_div(*[summary_val(summary, 'W1_note', k) for k in ('correct_parsed', 'parsed')]) or 0):.3f} "
            f"parsed_all3={summary['parsed_all3']} flips={summary['w1_flips']} "
            f"flip_rate={('NA' if fr is None else f'{fr*100:.1f}%')} "
            f"rows_rewritten={changed_rows} summary_rewritten={touched_summary}"
        )

    if args.rewrite_steering_rows:
        for run_dir, rows_path in iter_steering_dirs(results_root):
            rel = run_dir.relative_to(results_root)
            family = rel.parts[0]
            if families and family not in families:
                continue
            if not include_steering_run(rel):
                continue
            rows_all = load_rows_stable(rows_path, args.stability_retries)
            changed = rewrite_parser_labels(rows_all, arm="")
            if changed:
                write_rows(rows_path, rows_all)
            touched_summary = False
            summary_path = run_dir / "steering_summary.json"
            if args.rewrite_steering_piqa_summaries:
                touched_summary = refresh_steering_summary(summary_path, rows_all)
            if summary_path.exists():
                summary_rows = json.loads(summary_path.read_text())
                model = load_mechanism_model(run_dir, "steering")
                subdir = "/".join(rel.parts[1:])
                tag = canonical_mechanism_tag(family, subdir)
                entry = shared_entry_for_mechanism_summary(summary_rows, run_kind="steering")
                shared_runs[f"{model}::{tag}"] = entry
                records.append(
                    {
                        "run_kind": "steering",
                        "model": model,
                        "tag": tag,
                        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                        "entry": entry,
                    }
                )
            print(
                f"[steering] {rows_path.relative_to(REPO_ROOT)} "
                f"rows={len(rows_all)} rewritten={changed} summary_rewritten={touched_summary}"
            )

    if args.rewrite_piqa_rows:
        for run_dir, rows_path in iter_piqa_dirs(results_root):
            rel = run_dir.relative_to(results_root)
            family = rel.parts[0]
            if families and family not in families:
                continue
            if not include_piqa_run(rel):
                continue
            rows_all = load_rows_stable(rows_path, args.stability_retries)
            changed = rewrite_parser_labels(rows_all, arm="")
            if changed:
                write_rows(rows_path, rows_all)
            touched_summary = False
            summary_path = run_dir / "piqa_summary.json"
            if args.rewrite_steering_piqa_summaries:
                touched_summary = refresh_steering_summary(summary_path, rows_all)
            if summary_path.exists():
                summary_rows = json.loads(summary_path.read_text())
                model = load_mechanism_model(run_dir, "piqa")
                subdir = "/".join(rel.parts[1:])
                tag = canonical_mechanism_tag(family, subdir)
                entry = shared_entry_for_mechanism_summary(summary_rows, run_kind="piqa")
                shared_runs[f"{model}::{tag}"] = entry
                records.append(
                    {
                        "run_kind": "piqa",
                        "model": model,
                        "tag": tag,
                        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                        "entry": entry,
                    }
                )
            print(
                f"[piqa] {rows_path.relative_to(REPO_ROOT)} "
                f"rows={len(rows_all)} rewritten={changed} summary_rewritten={touched_summary}"
            )

    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    (SHARED_DIR / "dynamic_parser_all_runs.json").write_text(
        json.dumps({k: shared_runs[k] for k in sorted(shared_runs)}, indent=2) + "\n"
    )
    write_compat_subset_files(shared_runs)
    records.sort(key=lambda rec: (rec["run_kind"], rec["model"], rec["tag"]))
    (SHARED_DIR / "dynamic_parser_summary_index.md").write_text(index_md(records) + "\n")
    print(f"\nWrote {len(shared_runs)} runs to {SHARED_DIR / 'dynamic_parser_all_runs.json'}")
    print(f"Wrote index to {SHARED_DIR / 'dynamic_parser_summary_index.md'}")


def summary_val(summary: dict[str, Any], cond: str, key: str) -> int:
    for c in summary["conditions"]:
        if c["condition_code"] == cond:
            return c.get(key, 0)
    return 0


if __name__ == "__main__":
    main()
