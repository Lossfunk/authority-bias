from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NEITHER_PAT = re.compile(
    r"(?i)\b("
    r"neither|none of the options|none of the above|both options are wrong|"
    r"both are wrong|not one of the options|no option|options provided do not include|"
    r"neither of the options"
    r")\b"
)
START_LABEL_PAT = re.compile(r"(?i)^\s*([AB])\s*[\)\.:,-]")
ANSWER_IS_PAT = re.compile(
    r"(?i)\b(?:the\s+correct\s+answer\s+is|correct answer is|final answer is|"
    r"answer is|therefore[,]?\s+the correct answer is|so[,]?\s+the correct answer is)\s*([AB])\b"
)
STANDALONE_PAT = re.compile(r"(?i)^\s*(?:the\s+correct\s+answer\s+is\s+)?([AB])\b")
ALL_LABEL_PAT = re.compile(r"(?i)\b([AB])\s*[\)\.:,-]")
FINAL_MARKERS_DEFAULT: tuple[str, ...] = (
    "assistantfinal",
    "answer",
    "final",
    "therefore",
    "thus",
    "hence",
    "response",
    "<|assistant|>",
)


def norm(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(text)).split())


def strip_prompt_echo(nraw: str, nprompt: str) -> str:
    if not nraw:
        return nraw
    if nprompt and nprompt in nraw:
        return nraw.split(nprompt, 1)[1].strip()

    raw_tokens = nraw.split()
    prompt_tokens = nprompt.split()
    i = 0
    lim = min(len(raw_tokens), len(prompt_tokens))
    while i < lim and raw_tokens[i] == prompt_tokens[i]:
        i += 1
    if i > 0:
        return " ".join(raw_tokens[i:]).strip()
    return nraw


def parse_by_last_anywhere(text: str, ca: str, wa: str, cl: str, wl: str) -> str | None:
    pc = text.rfind(ca) if ca else -1
    pw = text.rfind(wa) if wa else -1
    if pc < 0 and pw < 0:
        return None
    if pc >= 0 and pw < 0:
        return cl
    if pw >= 0 and pc < 0:
        return wl
    if pc == pw:
        return None
    return cl if pc > pw else wl


def parse_after_last_marker(text: str, ca: str, wa: str, cl: str, wl: str) -> tuple[str | None, str]:
    best = -1
    best_marker = "none"
    for marker in FINAL_MARKERS_DEFAULT:
        idx = text.rfind(marker)
        if idx > best:
            best = idx
            best_marker = marker
    if best < 0:
        return None, "no_marker"
    sub = text[best + len(best_marker) :]
    return parse_by_last_anywhere(sub, ca, wa, cl, wl), best_marker


def dynamic_fallback_label(
    raw_text: str,
    prompt_text: str,
    correct_text: str,
    wrong_text: str,
    correct_label: str,
    wrong_label: str,
) -> tuple[str | None, str]:
    nraw = norm(raw_text)
    nprompt = norm(prompt_text)
    text = strip_prompt_echo(nraw, nprompt)

    ca = norm(correct_text)
    wa = norm(wrong_text)
    s1 = parse_by_last_anywhere(text, ca, wa, correct_label, wrong_label)
    s2, marker = parse_after_last_marker(text, ca, wa, correct_label, wrong_label)
    s3 = parse_by_last_anywhere(text[-160:], ca, wa, correct_label, wrong_label)
    s4 = parse_by_last_anywhere(text[-80:], ca, wa, correct_label, wrong_label)
    votes = [x for x in (s1, s2, s3, s4) if x in {"A", "B"}]
    if not votes:
        return None, f"dynamic:none:{marker}"

    vc = Counter(votes)
    ordered = vc.most_common()
    if len(ordered) == 1 or (len(ordered) > 1 and ordered[0][1] > ordered[1][1]):
        return ordered[0][0], f"dynamic:consensus:{marker}"
    if s2 in {"A", "B"}:
        return s2, f"dynamic:tie_marker:{marker}"
    return None, f"dynamic:tie_unresolved:{marker}"


def parse_olmo_choice(
    raw_text: str,
    *,
    prompt_text: str,
    correct_text: str,
    wrong_text: str,
    correct_label: str,
    wrong_label: str,
) -> tuple[str | None, str]:
    text = (raw_text or "").strip()
    if not text:
        return None, "empty"
    if NEITHER_PAT.search(text):
        return None, "neither"

    # OLMo often emits an explicit leading "A)" / "B)" token. Prefer that.
    match = START_LABEL_PAT.search(text)
    if match:
        return match.group(1).upper(), "start_label"

    match = ANSWER_IS_PAT.search(text)
    if match:
        return match.group(1).upper(), "answer_is"

    match = STANDALONE_PAT.search(text)
    if match:
        return match.group(1).upper(), "standalone"

    ntext = norm(text)
    ncorrect = norm(correct_text)
    nwrong = norm(wrong_text)

    # For freegen answers, prefer the opening answer phrase over later mentions in explanations.
    if ncorrect and ntext.startswith(ncorrect):
        return correct_label, "lead_correct_text"
    if nwrong and ntext.startswith(nwrong):
        return wrong_label, "lead_wrong_text"

    # TruthfulQA-like rows often reduce to yes/no/disagreement as the opening token.
    if ntext.startswith("yes"):
        if ncorrect == "yes":
            return correct_label, "lead_yes"
        if nwrong == "yes":
            return wrong_label, "lead_yes"
    if ntext.startswith("no"):
        if ncorrect == "no":
            return correct_label, "lead_no"
        if nwrong == "no":
            return wrong_label, "lead_no"
    if ntext.startswith("disagreement"):
        if ncorrect == "disagreement":
            return correct_label, "lead_disagreement"
        if nwrong == "disagreement":
            return wrong_label, "lead_disagreement"

    return dynamic_fallback_label(
        raw_text=text,
        prompt_text=prompt_text,
        correct_text=correct_text,
        wrong_text=wrong_text,
        correct_label=correct_label,
        wrong_label=wrong_label,
    )


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cfg_alpha: dict[tuple[str, float], dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "parsed": 0, "correct": 0, "wrong": 0}
    )
    method_counts: Counter[str] = Counter()
    mismatch_vs_stored = 0
    contradictory_start_label = 0
    neither_rows = 0
    dual_label_rows = 0

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        label, method = parse_olmo_choice(
            row.get("raw_text", ""),
            prompt_text=row.get("prompt_text", ""),
            correct_text=row["correct_text"],
            wrong_text=row["wrong_text"],
            correct_label=row["correct_label"],
            wrong_label=row["wrong_label"],
        )
        parsed = dict(row)
        parsed["_olmo_label"] = label
        parsed["_olmo_method"] = method
        parsed["_olmo_is_correct"] = label == row["correct_label"] if label is not None else None
        parsed["_olmo_chose_wrong"] = label == row["wrong_label"] if label is not None else None
        parsed_rows.append(parsed)

        method_counts[method] += 1
        if label != row.get("parsed_label"):
            mismatch_vs_stored += 1

        labels_seen = {m.group(1).upper() for m in ALL_LABEL_PAT.finditer(row.get("raw_text", ""))}
        if len(labels_seen) == 2:
            dual_label_rows += 1
        if method == "neither":
            neither_rows += 1

        start_match = START_LABEL_PAT.search((row.get("raw_text") or "").strip())
        if start_match and row.get("parsed_label") not in {None, start_match.group(1).upper()}:
            contradictory_start_label += 1

        bucket = by_cfg_alpha[(row["config_id"], row["alpha"])]
        bucket["n"] += 1
        if label is not None:
            bucket["parsed"] += 1
            if parsed["_olmo_is_correct"] is True:
                bucket["correct"] += 1
            if parsed["_olmo_chose_wrong"] is True:
                bucket["wrong"] += 1

    def pct(x: int, y: int) -> float:
        return (100.0 * x / y) if y else 0.0

    config_summaries = []
    for (config_id, alpha), stats in sorted(by_cfg_alpha.items()):
        config_summaries.append(
            {
                "config_id": config_id,
                "alpha": alpha,
                "n": stats["n"],
                "parsed": stats["parsed"],
                "parse_rate_pct": round(pct(stats["parsed"], stats["n"]), 2),
                "correct_parsed": stats["correct"],
                "correct_rate_parsed_pct": round(pct(stats["correct"], stats["parsed"]), 2),
                "wrong_parsed": stats["wrong"],
                "wrong_rate_parsed_pct": round(pct(stats["wrong"], stats["parsed"]), 2),
                "wrong_rate_total_pct": round(pct(stats["wrong"], stats["n"]), 2),
            }
        )

    flip_summaries = []
    for config_id in sorted({row["config_id"] for row in parsed_rows}):
        label_meta = {
            row["uid"]: (row["correct_label"], row["wrong_label"])
            for row in parsed_rows
            if row["config_id"] == config_id
        }
        baseline_labels = {
            row["uid"]: row["_olmo_label"]
            for row in parsed_rows
            if row["config_id"] == config_id and row["alpha"] == 0.0
        }
        baseline_correct = {
            uid for uid, label in baseline_labels.items() if label == label_meta[uid][0]
        }
        for alpha in sorted({row["alpha"] for row in parsed_rows if row["config_id"] == config_id}):
            alpha_labels = {
                row["uid"]: row["_olmo_label"]
                for row in parsed_rows
                if row["config_id"] == config_id and row["alpha"] == alpha
            }
            common = baseline_correct & set(alpha_labels)
            parsed_common = sum(1 for uid in common if alpha_labels[uid] is not None)
            flips = sum(1 for uid in common if alpha_labels[uid] == label_meta[uid][1])
            flip_summaries.append(
                {
                    "config_id": config_id,
                    "alpha": alpha,
                    "baseline_correct_n": len(baseline_correct),
                    "parsed_common_n": parsed_common,
                    "flips_to_wrong": flips,
                    "flip_rate_pct": round(pct(flips, parsed_common), 2),
                }
            )

    corrected_summary = []
    for row in config_summaries:
        corrected_summary.append(
            {
                "config_id": row["config_id"],
                "alpha": row["alpha"],
                "condition": "N0_note",
                "total": row["n"],
                "parsed": row["parsed"],
                "parse_rate": row["parsed"] / row["n"] if row["n"] else 0.0,
                "accuracy": row["correct_parsed"] / row["parsed"] if row["parsed"] else 0.0,
                "wrong_rate": row["wrong_parsed"] / row["parsed"] if row["parsed"] else 0.0,
                "flip_rate": None,
                "mean_fc_logit_wrong": None,
                "std_fc_logit_wrong": None,
            }
        )

    return {
        "rows_total": len(rows),
        "method_counts": dict(method_counts),
        "mismatch_vs_stored": mismatch_vs_stored,
        "contradictory_start_label_vs_stored": contradictory_start_label,
        "neither_rows": neither_rows,
        "dual_label_rows": dual_label_rows,
        "config_summaries": config_summaries,
        "flip_summaries": flip_summaries,
        "corrected_steering_summary": corrected_summary,
    }


def write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# OLMo Additive Steering Parser Audit")
    lines.append("")
    lines.append(f"- Total rows: {summary['rows_total']}")
    lines.append(f"- Mismatch vs stored parser: {summary['mismatch_vs_stored']}")
    lines.append(
        f"- Rows where stored label contradicts explicit leading `A)`/`B)` token: {summary['contradictory_start_label_vs_stored']}"
    )
    lines.append(f"- `neither` / invalid-option rows: {summary['neither_rows']}")
    lines.append(f"- Rows containing both `A)` and `B)` style labels: {summary['dual_label_rows']}")
    lines.append(f"- Parse methods: {json.dumps(summary['method_counts'], sort_keys=True)}")
    lines.append("")
    lines.append("## By Config/Alpha")
    lines.append("")
    lines.append("| Config | Alpha | N | Parsed | Parse % | Correct % (parsed) | Wrong % (parsed) | Wrong % (total) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["config_summaries"]:
        lines.append(
            f"| `{row['config_id']}` | {row['alpha']} | {row['n']} | {row['parsed']} | "
            f"{row['parse_rate_pct']:.2f}% | {row['correct_rate_parsed_pct']:.2f}% | "
            f"{row['wrong_rate_parsed_pct']:.2f}% | {row['wrong_rate_total_pct']:.2f}% |"
        )
    lines.append("")
    lines.append("## Flip Rates On OLMo Parser")
    lines.append("")
    lines.append("| Config | Alpha | Baseline-correct n | Parsed common n | Flips to wrong | Flip % |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary["flip_summaries"]:
        lines.append(
            f"| `{row['config_id']}` | {row['alpha']} | {row['baseline_correct_n']} | "
            f"{row['parsed_common_n']} | {row['flips_to_wrong']} | {row['flip_rate_pct']:.2f}% |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-path", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument("--rewrite-summary-out", type=Path, default=None)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.rows_path.open()]
    summary = build_summary(rows)
    args.json_out.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, args.md_out)
    if args.rewrite_summary_out is not None:
        args.rewrite_summary_out.write_text(
            json.dumps(summary["corrected_steering_summary"], indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
