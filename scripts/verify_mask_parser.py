"""Physical audit of label_masks.json vs raw model generations.

Re-parses the authoritative_verified rows with the same parser used at extraction
time, plus a simple "eyeball" rule (last-occurrence of correct_answer vs wrong_answer
in the generation), and reports:
  - Agreement between metadata.jsonl (the cached decisions) and a live re-parse.
  - Agreement between the parser and the eyeball rule.
  - Row-level dumps for every uid in mask_primary_w1 and mask_primary_c1 so the
    cohort can be eyeballed directly.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from src.authority_steering.dynamic_parser import (
    discover_model_markers,
    dynamic_parse_row,
    norm,
    parse_by_last_anywhere,
    strip_prompt_echo,
)

ROOT = Path(__file__).resolve().parents[1]
MECH = ROOT / "results/authority/olmo31/mechanism/authority_activations_shared_h100"
STYLE_DIR = {
    "authoritative_verified": ROOT / "results/authority/olmo31/authoritative_verified_shared_h100",
    "weak": ROOT / "results/authority/olmo31/gradient_weak_shared_h100",
    "uncertain": ROOT / "results/authority/olmo31/gradient_uncertain_shared_h100",
    "assertive": ROOT / "results/authority/olmo31/gradient_assertive_shared_h100",
}
ROWS_NAME = "allenai__OLMo-3.1-32B-Instruct_dissociation_rows.jsonl"
CONDITION_CODES = ("N0_note", "C1_note", "W1_note")


def load_rows(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("arm") != "natural":
                continue
            if row.get("condition_code") not in CONDITION_CODES:
                continue
            out.append(row)
    return out


def eyeball_label(row: dict) -> str | None:
    """Naive rule: strip the prompt echo, then pick whichever of correct_answer/
    wrong_answer appears later in the stripped generation. Returns None if neither
    appears."""
    raw = row["generation"]["raw_text"]
    prompt = row.get("prompt_text", "")
    text = strip_prompt_echo(norm(raw), norm(prompt))
    ca = norm(row["correct_answer"])
    wa = norm(row["wrong_answer"])
    return parse_by_last_anywhere(text, ca, wa, row["correct_label"], row["wrong_label"])


def main() -> None:
    masks = json.loads((MECH / "label_masks.json").read_text())
    meta_path = MECH / "metadata.jsonl"
    meta_by_uid_style_cond: dict = {}
    with meta_path.open() as f:
        for line in f:
            rec = json.loads(line)
            meta_by_uid_style_cond.setdefault(rec["uid"], {}).setdefault(rec["style"], {})[
                rec["condition_code"]
            ] = rec

    rows_by_style_uid_cond: dict = {}
    markers_by_style: dict = {}
    for style, dir_path in STYLE_DIR.items():
        rows = load_rows(dir_path / ROWS_NAME)
        markers_by_style[style] = discover_model_markers(rows)
        ns: dict = {}
        for r in rows:
            ns.setdefault(r["uid"], {})[r["condition_code"]] = r
        rows_by_style_uid_cond[style] = ns

    w1_uids = masks["mask_primary_w1"]
    c1_uids = masks["mask_primary_c1"]
    print(f"Masks: primary_w1={len(w1_uids)}, primary_c1={len(c1_uids)}")

    disagreements_meta = []
    disagreements_eyeball = []
    parse_null_count = Counter()
    per_row_records = []

    cohorts = [("primary_w1", w1_uids), ("primary_c1", c1_uids)]
    for cohort_name, uids in cohorts:
        for uid in uids:
            for cond in CONDITION_CODES:
                row = rows_by_style_uid_cond["authoritative_verified"].get(uid, {}).get(cond)
                if row is None:
                    continue
                live_label, info = dynamic_parse_row(
                    row, markers_by_style["authoritative_verified"]
                )
                meta = meta_by_uid_style_cond.get(uid, {}).get("authoritative_verified", {}).get(cond)
                meta_label = meta["final_answer_label"] if meta else None

                eye_label = eyeball_label(row)

                if live_label != meta_label:
                    disagreements_meta.append(
                        {
                            "uid": uid,
                            "cond": cond,
                            "meta_label": meta_label,
                            "live_label": live_label,
                            "method": info.get("method"),
                            "marker": info.get("marker"),
                            "raw": row["generation"]["raw_text"][-200:],
                        }
                    )

                if live_label is None:
                    parse_null_count[cond] += 1

                if eye_label is not None and live_label is not None and eye_label != live_label:
                    disagreements_eyeball.append(
                        {
                            "uid": uid,
                            "cond": cond,
                            "parser": live_label,
                            "eyeball": eye_label,
                            "correct": row["correct_label"],
                            "wrong": row["wrong_label"],
                            "raw": row["generation"]["raw_text"][-250:],
                        }
                    )

                per_row_records.append(
                    {
                        "cohort": cohort_name,
                        "uid": uid,
                        "cond": cond,
                        "correct_label": row["correct_label"],
                        "wrong_label": row["wrong_label"],
                        "correct_answer": row["correct_answer"],
                        "wrong_answer": row["wrong_answer"],
                        "parser_label": live_label,
                        "meta_label": meta_label,
                        "eyeball_label": eye_label,
                        "final_is_correct_meta": meta.get("final_is_correct") if meta else None,
                        "w1_flip_meta": meta.get("w1_flip") if meta else None,
                        "c1_correction_meta": meta.get("c1_correction") if meta else None,
                        "n0_correct_meta": meta.get("n0_correct") if meta else None,
                        "raw_tail": row["generation"]["raw_text"][-200:],
                    }
                )

    # aggregate mask consistency: per-uid, recompute n0_correct, c1_correction, w1_flip
    # from metadata, and verify the mask membership matches.
    def n0_correct(uid: str) -> bool:
        meta = (
            meta_by_uid_style_cond.get(uid, {})
            .get("authoritative_verified", {})
            .get("N0_note")
        )
        return bool(meta and meta.get("n0_correct"))

    def c1_correction(uid: str) -> bool:
        meta = (
            meta_by_uid_style_cond.get(uid, {})
            .get("authoritative_verified", {})
            .get("C1_note")
        )
        return bool(meta and meta.get("c1_correction"))

    def w1_flip(uid: str) -> bool:
        meta = (
            meta_by_uid_style_cond.get(uid, {})
            .get("authoritative_verified", {})
            .get("W1_note")
        )
        return bool(meta and meta.get("w1_flip"))

    w1_stats = Counter()
    for uid in w1_uids:
        w1_stats["n0_correct_true"] += int(n0_correct(uid))
        w1_stats["w1_flip_true"] += int(w1_flip(uid))
    c1_stats = Counter()
    for uid in c1_uids:
        c1_stats["n0_correct_true"] += int(n0_correct(uid))
        c1_stats["c1_correction_true"] += int(c1_correction(uid))

    print("\n--- W1 cohort (mask_primary_w1, n={}):".format(len(w1_uids)))
    print("  n0_correct=True   count:", w1_stats["n0_correct_true"], "  (should equal cohort size; definition)")
    print("  w1_flip=True      count:", w1_stats["w1_flip_true"])
    print("\n--- C1 cohort (mask_primary_c1, n={}):".format(len(c1_uids)))
    print("  n0_correct=True   count:", c1_stats["n0_correct_true"], "  (should be 0; cohort = N0 wrong)")
    print("  c1_correction=True count:", c1_stats["c1_correction_true"])

    print("\n--- Parser live-vs-metadata disagreements:", len(disagreements_meta))
    for d in disagreements_meta[:20]:
        print(" ", d)

    print("\n--- Parser-vs-eyeball disagreements (both non-null):", len(disagreements_eyeball))
    for d in disagreements_eyeball[:25]:
        print(" ", json.dumps(d, ensure_ascii=False))

    print("\n--- Null parse counts by condition (cohort = w1 ∪ c1):")
    for k, v in parse_null_count.items():
        print(f"  {k}: {v}")

    out_path = MECH / "parser_audit.jsonl"
    with out_path.open("w") as f:
        for rec in per_row_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nWrote row-level audit to {out_path}")

    summary = {
        "n_w1_uids": len(w1_uids),
        "n_c1_uids": len(c1_uids),
        "disagreements_meta_vs_live": len(disagreements_meta),
        "disagreements_parser_vs_eyeball": len(disagreements_eyeball),
        "null_parse_counts": dict(parse_null_count),
        "w1_cohort_stats": dict(w1_stats),
        "c1_cohort_stats": dict(c1_stats),
    }
    (MECH / "parser_audit_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
