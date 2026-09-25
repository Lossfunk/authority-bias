from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.evidence_quality.conditions import Exp14Condition, format_prompt as format_exp14_prompt
from src.evidence_quality.reasons_dataset import ReasonEntry, load_reasons_dataset
from src.lexical_controls.dataset_mc import MCExample
from src.mechanism.common import (
    CORE_EVIDENCE_LEVELS,
    CORE_INSTRUCTION_KEYS,
    CORE_TAGS,
    PanelRow,
    answer_length_bucket,
    confidence_slice,
    margin_bin,
    model_tag,
    normalize_tag,
    prompt_length_bucket,
    read_jsonl,
    stable_run_id,
    write_json,
    write_panel_parquet,
)


@dataclass(frozen=True)
class ItemProfile:
    uid: str
    dataset: str
    topic_domain: str
    authority: str
    evidence_strength: str
    prior_margin: float
    abs_prior_margin: float
    prior_state: str
    confidence_slice: str
    answer_format: str
    label_layout: str
    prompt_length_bucket: int
    matched_control_uids: Tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical mechanistic panel from Exp14 results")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("new-phase-results/qwen3-4b-results/evidence_quality/Qwen__Qwen3-4B-Instruct-2507_results.jsonl"),
    )
    parser.add_argument("--reasons-path", type=Path, default=Path("data/exp14_reasons.jsonl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("new-phase-results/mechanism/qwen_panel"),
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--tags", type=str, default=",".join(CORE_TAGS))
    parser.add_argument("--evidence-levels", type=str, default=",".join(CORE_EVIDENCE_LEVELS))
    parser.add_argument("--controls-per-item", type=int, default=2)
    return parser.parse_args()


def _parse_csv(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _load_results(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(path)


def _to_mc_example(record: Mapping[str, Any]) -> MCExample:
    correct_label = str(record["correct_label"])
    wrong_label = str(record["wrong_label"])
    correct_answer = str(record["correct_answer"])
    wrong_answer = str(record["wrong_answer"])
    if correct_label == "A":
        option_a, option_b = correct_answer, wrong_answer
    else:
        option_a, option_b = wrong_answer, correct_answer
    return MCExample(
        uid=str(record["uid"]),
        question=str(record["question"]),
        correct_answer=correct_answer,
        wrong_answer=wrong_answer,
        correct_label=correct_label,
        wrong_label=wrong_label,
        option_a=option_a,
        option_b=option_b,
        endorsed_label=wrong_label,
        metadata=dict(record.get("metadata") or {}),
    )


def _topic_domain(record: Mapping[str, Any], reason_entry: Optional[ReasonEntry]) -> str:
    dataset = str((record.get("metadata") or {}).get("dataset", "unknown"))
    if reason_entry and reason_entry.topic_phrases:
        return f"{dataset}:{reason_entry.topic_phrases[0]}"
    return dataset


def _condition(tag: str, instruction_key: str, endorse_type: str, evidence_level: str) -> Exp14Condition:
    return Exp14Condition(
        endorse_type=endorse_type,  # type: ignore[arg-type]
        instruction_key=instruction_key,
        tag=tag,  # type: ignore[arg-type]
        evidence_level=evidence_level,  # type: ignore[arg-type]
    )


def _prior_code(tag: str, evidence_level: str) -> str:
    return f"N0_{normalize_tag(tag)}_{evidence_level}"


def _fc_correct(record: Mapping[str, Any], code: str) -> float:
    fc = record.get("fc_correct") or {}
    if code not in fc:
        raise KeyError(f"Missing fc_correct[{code}]")
    return float(fc[code])


def _logit_correct(record: Mapping[str, Any], code: str) -> float:
    logits = record.get("logit_correct") or {}
    if code not in logits:
        raise KeyError(f"Missing logit_correct[{code}]")
    return float(logits[code])


def _build_profiles(
    records: Sequence[Mapping[str, Any]],
    reasons_by_uid: Mapping[str, ReasonEntry],
    *,
    tags: Sequence[str],
    evidence_levels: Sequence[str],
) -> Dict[Tuple[str, str, str], ItemProfile]:
    grouped_abs: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    base_rows: List[Tuple[Tuple[str, str, str], Dict[str, Any]]] = []

    for record in records:
        uid = str(record["uid"])
        reason_entry = reasons_by_uid.get(uid)
        topic = _topic_domain(record, reason_entry)
        mc_example = _to_mc_example(record)
        label_layout = f"correct={mc_example.correct_label}"
        for tag in tags:
            for level in evidence_levels:
                code = _prior_code(tag, level)
                if code not in (record.get("logit_correct") or {}):
                    continue
                margin = _logit_correct(record, code)
                abs_margin = abs(margin)
                neutral_prompt = format_exp14_prompt(
                    mc_example,
                    _condition(tag, "0", "neutral", level),
                    reason_entry.to_prompt_payload() if reason_entry else {
                        "topic_phrase_1": topic,
                        "topic_phrase_2": topic,
                        "reason_correct_1": "",
                        "reason_correct_2": "",
                        "reason_wrong_1": "",
                        "reason_wrong_2": "",
                        "data_correct": "",
                        "data_wrong": "",
                    },
                )
                grouped_abs[(tag, level)].append(abs_margin)
                base_rows.append(
                    (
                        (uid, tag, level),
                        {
                            "uid": uid,
                            "dataset": str((record.get("metadata") or {}).get("dataset", "unknown")),
                            "topic_domain": topic,
                            "authority": tag,
                            "evidence_strength": level,
                            "prior_margin": margin,
                            "abs_prior_margin": abs_margin,
                            "prior_state": "prior_wrong" if margin < 0 else "prior_correct",
                            "answer_format": "ab",
                            "label_layout": label_layout,
                            "prompt_length_bucket": prompt_length_bucket(neutral_prompt),
                        },
                    )
                )

    quantiles: Dict[Tuple[str, str], Tuple[float, Optional[float]]] = {}
    for key, values in grouped_abs.items():
        sorted_values = sorted(values)
        if not sorted_values:
            continue
        q75 = sorted_values[int(0.75 * (len(sorted_values) - 1))]
        q90 = sorted_values[int(0.9 * (len(sorted_values) - 1))] if len(sorted_values) >= 10 else None
        quantiles[key] = (q75, q90)

    profiles: Dict[Tuple[str, str, str], ItemProfile] = {}
    for key, payload in base_rows:
        q75, q90 = quantiles[(payload["authority"], payload["evidence_strength"])]
        slice_name = confidence_slice(payload["abs_prior_margin"], q75, q90)
        profiles[key] = ItemProfile(
            uid=payload["uid"],
            dataset=payload["dataset"],
            topic_domain=payload["topic_domain"],
            authority=payload["authority"],
            evidence_strength=payload["evidence_strength"],
            prior_margin=float(payload["prior_margin"]),
            abs_prior_margin=float(payload["abs_prior_margin"]),
            prior_state=str(payload["prior_state"]),
            confidence_slice=slice_name,
            answer_format=str(payload["answer_format"]),
            label_layout=str(payload["label_layout"]),
            prompt_length_bucket=int(payload["prompt_length_bucket"]),
        )
    return profiles


def _match_controls(
    profiles: Mapping[Tuple[str, str, str], ItemProfile],
    *,
    controls_per_item: int,
) -> Dict[Tuple[str, str, str], Tuple[str, ...]]:
    by_group: Dict[Tuple[str, str], List[ItemProfile]] = defaultdict(list)
    for profile in profiles.values():
        by_group[(profile.authority, profile.evidence_strength)].append(profile)

    matched: Dict[Tuple[str, str, str], Tuple[str, ...]] = {}
    for key, profile in profiles.items():
        if profile.prior_state != "prior_wrong":
            continue
        candidates = [
            candidate
            for candidate in by_group[(profile.authority, profile.evidence_strength)]
            if candidate.prior_state == "prior_correct"
            and candidate.uid != profile.uid
            and candidate.answer_format == profile.answer_format
            and candidate.label_layout == profile.label_layout
        ]
        if not candidates:
            matched[key] = tuple()
            continue

        def rank(candidate: ItemProfile) -> Tuple[int, int, int, float, str]:
            same_topic = 0 if candidate.topic_domain == profile.topic_domain else 1
            same_dataset = 0 if candidate.dataset == profile.dataset else 1
            same_prompt_bucket = abs(candidate.prompt_length_bucket - profile.prompt_length_bucket)
            margin_distance = abs(candidate.abs_prior_margin - profile.abs_prior_margin)
            return (
                same_topic,
                same_dataset,
                same_prompt_bucket,
                margin_distance,
                candidate.uid,
            )

        selected = sorted(candidates, key=rank)[:controls_per_item]
        matched[key] = tuple(candidate.uid for candidate in selected)
    return matched


def build_mechanistic_panel(
    *,
    model_name: str,
    results_path: Path,
    reasons_path: Path,
    output_dir: Path,
    tags: Sequence[str],
    evidence_levels: Sequence[str],
    controls_per_item: int,
) -> Dict[str, Any]:
    records = _load_results(results_path)
    reasons_by_uid = load_reasons_dataset(reasons_path)
    profiles = _build_profiles(records, reasons_by_uid, tags=tags, evidence_levels=evidence_levels)
    matched_controls = _match_controls(profiles, controls_per_item=controls_per_item)

    selected_keys = {
        key
        for key, profile in profiles.items()
        if profile.prior_state == "prior_wrong" and profile.confidence_slice in {"high_conf_top25", "top10"}
    }
    for key, matched_uids in matched_controls.items():
        if key in selected_keys:
            authority, level = key[1], key[2]
            for matched_uid in matched_uids:
                selected_keys.add((matched_uid, authority, level))

    rows: List[PanelRow] = []
    for record in records:
        uid = str(record["uid"])
        reason_entry = reasons_by_uid.get(uid)
        mc_example = _to_mc_example(record)
        if reason_entry is None:
            continue
        payload = reason_entry.to_prompt_payload()
        for tag in tags:
            for level in evidence_levels:
                key = (uid, tag, level)
                if key not in selected_keys:
                    continue
                profile = profiles[key]
                matched_uids = matched_controls.get(key, tuple())
                for instruction_key in CORE_INSTRUCTION_KEYS:
                    for endorse_type, direction in (("neutral", "neutral"), ("wrong", "misleading"), ("correct", "corrective")):
                        condition = _condition(tag, instruction_key, endorse_type, level)
                        code = condition.code
                        prompt = format_exp14_prompt(mc_example, condition, payload)
                        metadata = {
                            "model": model_name,
                            "source_results": str(results_path),
                            "prior_code": _prior_code(tag, level),
                            "logit_correct": _logit_correct(record, code),
                            "fc_correct": _fc_correct(record, code),
                            "margin_bin": margin_bin(profile.prior_margin),
                            "answer_length_bucket": answer_length_bucket(mc_example.correct_answer),
                        }
                        rows.append(
                            PanelRow(
                                uid=uid,
                                dataset=profile.dataset,
                                topic_domain=profile.topic_domain,
                                question=mc_example.question,
                                correct_answer=mc_example.correct_answer,
                                wrong_answer=mc_example.wrong_answer,
                                answer_format=profile.answer_format,
                                correct_label=mc_example.correct_label,
                                wrong_label=mc_example.wrong_label,
                                prior_margin=profile.prior_margin,
                                prior_state=profile.prior_state,
                                confidence_slice=profile.confidence_slice,
                                authority=tag,
                                instruction=instruction_key,
                                evidence_direction=direction,
                                evidence_strength=level,
                                condition_id=code,
                                prompt_text=prompt,
                                target_metric_metadata=metadata,
                                matched_control_uids=matched_uids,
                            )
                        )

    run_id = stable_run_id(
        "panel",
        {
            "model": model_name,
            "results_path": str(results_path),
            "tags": list(tags),
            "evidence_levels": list(evidence_levels),
            "controls_per_item": controls_per_item,
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / f"{model_tag(model_name)}_mechanistic_panel.parquet"
    write_panel_parquet(panel_path, rows)

    summary = {
        "run_id": run_id,
        "model": model_name,
        "results_path": str(results_path),
        "reasons_path": str(reasons_path),
        "panel_path": str(panel_path),
        "n_rows": len(rows),
        "n_item_profiles": len(profiles),
        "n_selected_keys": len(selected_keys),
        "n_prior_wrong_high_conf": sum(
            1 for key in selected_keys if profiles.get(key) and profiles[key].prior_state == "prior_wrong"
        ),
        "tags": list(tags),
        "evidence_levels": list(evidence_levels),
        "controls_per_item": controls_per_item,
    }
    write_json(output_dir / f"{model_tag(model_name)}_mechanistic_panel_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = build_mechanistic_panel(
        model_name=args.model,
        results_path=args.results_path,
        reasons_path=args.reasons_path,
        output_dir=args.output_dir,
        tags=_parse_csv(args.tags),
        evidence_levels=_parse_csv(args.evidence_levels),
        controls_per_item=args.controls_per_item,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
