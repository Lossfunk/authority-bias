#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.authority_steering.dynamic_parser import (
    _jaccard,
    _token_set,
    discover_model_markers,
    dynamic_parse_row,
    norm,
    strip_prompt_echo,
)


CONDITIONS = ("N0_note", "C1_note", "W1_note")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze strict vs fuzzy parser behavior on freegen dissociation rows.")
    p.add_argument("--rows-path", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    p.add_argument("--uids-from-rows", type=Path, default=None, help="Optional rows file; restrict analysis to its natural-arm UIDs.")
    p.add_argument(
        "--compare-run",
        action="append",
        default=[],
        help="Optional comparison run in the form label=/abs/path/to/_dissociation_rows.jsonl",
    )
    p.add_argument("--jaccard-threshold", type=float, default=0.60)
    return p.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("arm") != "natural":
                continue
            if r.get("condition_code") not in CONDITIONS:
                continue
            rows.append(r)
    return rows


def uid_set_from_rows(path: Path) -> set[str]:
    return {r["uid"] for r in load_rows(path)}


def source_from_uid(uid: str) -> str:
    if "::" in uid:
        return uid.split("::", 1)[0]
    return "unknown"


def row_text(row: dict[str, Any]) -> str:
    raw = row.get("generation", {}).get("raw_text", "")
    prompt = row.get("prompt_text", "")
    return strip_prompt_echo(norm(raw), norm(prompt))


def summarize(rows: list[dict[str, Any]], *, jaccard_threshold: float) -> dict[str, Any]:
    markers = discover_model_markers(rows)
    cond_strict: dict[str, Counter[str]] = defaultdict(Counter)
    cond_fuzzy: dict[str, Counter[str]] = defaultdict(Counter)
    n0_categories = Counter()
    n0_split = defaultdict(Counter)
    no_parse_overlap = Counter()

    for r in rows:
        cond = str(r["condition_code"])
        strict_label, meta = dynamic_parse_row(r, markers)
        fuzzy_label = strict_label if strict_label in {"A", "B"} else meta.get("fuzzy_label")
        src = source_from_uid(str(r["uid"]))

        cond_strict[cond]["n"] += 1
        cond_fuzzy[cond]["n"] += 1

        if strict_label in {"A", "B"}:
            cond_strict[cond]["parsed"] += 1
            cond_strict[cond]["correct"] += int(strict_label == r["correct_label"])
            cond_strict[cond]["wrong"] += int(strict_label == r["wrong_label"])
        else:
            cond_strict[cond]["no_parse"] += 1

        if fuzzy_label in {"A", "B"}:
            cond_fuzzy[cond]["parsed"] += 1
            cond_fuzzy[cond]["correct"] += int(fuzzy_label == r["correct_label"])
            cond_fuzzy[cond]["wrong"] += int(fuzzy_label == r["wrong_label"])
            cond_fuzzy[cond]["rescued_from_none"] += int(strict_label not in {"A", "B"})
        else:
            cond_fuzzy[cond]["no_parse"] += 1

        if cond != "N0_note":
            continue

        txt = row_text(r)
        ca = norm(str(r["correct_answer"]))
        wa = norm(str(r["wrong_answer"]))
        has_ca = bool(ca and ca in txt)
        has_wa = bool(wa and wa in txt)

        if strict_label == r["correct_label"]:
            bucket = "correct"
        elif strict_label == r["wrong_label"]:
            if has_ca and has_wa:
                bucket = "wrong_mentions_both"
            elif (not has_ca) and (not has_wa):
                bucket = "wrong_neither_present"
            else:
                bucket = "wrong_confidently"
        else:
            bucket = "no_parse"
            text_tokens = _token_set(txt)
            jc = _jaccard(text_tokens, _token_set(str(r["correct_answer"])))
            jw = _jaccard(text_tokens, _token_set(str(r["wrong_answer"])))
            no_parse_overlap["max_jaccard_ge_thresh"] += int(max(jc, jw) >= jaccard_threshold)
            no_parse_overlap["count"] += 1

        n0_categories[bucket] += 1
        n0_split[src][bucket] += 1
        if strict_label in {"A", "B"}:
            n0_split[src]["strict_correct"] += int(strict_label == r["correct_label"])
            n0_split[src]["strict_wrong"] += int(strict_label == r["wrong_label"])
            n0_split[src]["strict_parsed"] += 1
        n0_split[src]["n"] += 1

    def _rates(counter: Counter[str]) -> dict[str, float]:
        n = counter["n"] or 1
        parsed = counter["parsed"]
        return {
            "n": int(counter["n"]),
            "parsed": int(parsed),
            "parse_rate": parsed / n,
            "correct": int(counter["correct"]),
            "wrong": int(counter["wrong"]),
            "correct_total_rate": counter["correct"] / n,
            "wrong_total_rate": counter["wrong"] / n,
            "correct_parsed_rate": (counter["correct"] / parsed) if parsed else 0.0,
            "wrong_parsed_rate": (counter["wrong"] / parsed) if parsed else 0.0,
            "no_parse": int(counter["no_parse"]),
        }

    out = {
        "conditions_strict": {c: _rates(cond_strict[c]) for c in CONDITIONS},
        "conditions_fuzzy": {c: _rates(cond_fuzzy[c]) for c in CONDITIONS},
        "n0_categories_strict": dict(n0_categories),
        "n0_split": {k: dict(v) for k, v in n0_split.items()},
        "no_parse_overlap": dict(no_parse_overlap),
        "markers": markers,
    }
    return out


def format_md(name: str, s: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append(f"## {name}")
    lines.append("")
    lines.append("| cond | strict_parse | strict_acc_total | fuzzy_parse | fuzzy_acc_total | rescued |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for c in CONDITIONS:
        a = s["conditions_strict"][c]
        b = s["conditions_fuzzy"][c]
        lines.append(
            f"| {c} | {a['parse_rate']:.3f} | {a['correct_total_rate']:.3f} | "
            f"{b['parse_rate']:.3f} | {b['correct_total_rate']:.3f} | {int(b.get('rescued_from_none', 0))} |"
        )
    lines.append("")
    n0 = s["n0_categories_strict"]
    total = sum(n0.values()) or 1
    lines.append("N0 strict category mix:")
    for key in ("wrong_confidently", "no_parse", "correct", "wrong_mentions_both", "wrong_neither_present"):
        val = int(n0.get(key, 0))
        lines.append(f"- `{key}`: {val} ({val/total:.1%})")
    overlap = s["no_parse_overlap"]
    if overlap.get("count", 0):
        lines.append(
            f"- `no_parse` rows with max token-jaccard >= 0.60: "
            f"{int(overlap.get('max_jaccard_ge_thresh', 0))}/{int(overlap.get('count', 0))}"
        )
    lines.append("")
    lines.append("N0 by dataset source:")
    lines.append("| source | n | strict_parsed | strict_correct | strict_wrong |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for src, d in sorted(s["n0_split"].items()):
        lines.append(
            f"| {src} | {int(d.get('n',0))} | {int(d.get('strict_parsed',0))} | "
            f"{int(d.get('strict_correct',0))} | {int(d.get('strict_wrong',0))} |"
        )
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    base_rows = load_rows(args.rows_path)
    if args.uids_from_rows is not None:
        keep = uid_set_from_rows(args.uids_from_rows)
        base_rows = [r for r in base_rows if r["uid"] in keep]

    outputs: dict[str, Any] = {}
    outputs["base"] = summarize(base_rows, jaccard_threshold=args.jaccard_threshold)
    outputs["base"]["rows_path"] = str(args.rows_path)
    outputs["base"]["n_rows"] = len(base_rows)

    md_lines = format_md("base", outputs["base"])

    for spec in args.compare_run:
        if "=" not in spec:
            raise ValueError(f"Invalid --compare-run value: {spec}")
        label, path_s = spec.split("=", 1)
        path = Path(path_s)
        rows = load_rows(path)
        if args.uids_from_rows is not None:
            keep = uid_set_from_rows(args.uids_from_rows)
            rows = [r for r in rows if r["uid"] in keep]
        outputs[label] = summarize(rows, jaccard_threshold=args.jaccard_threshold)
        outputs[label]["rows_path"] = str(path)
        outputs[label]["n_rows"] = len(rows)
        md_lines.extend(format_md(label, outputs[label]))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(outputs, indent=2) + "\n")
    args.output_md.write_text("\n".join(md_lines) + "\n")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2))


if __name__ == "__main__":
    main()
