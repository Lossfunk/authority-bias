"""Experiment 13: Signal-strength endorsement study with resume/checkpointing.

Design:
- tags: Expert / Note / User / Someone online
- certainty: might / think / sure
- salience: plain / important
- schedules: none / t0
- conditions: neutral / wrong / correct

Checkpointing/resume:
- append-only per-model JSONL results
- per-model manifest with config fingerprint
- per-model checkpoint state file
- safe tail-repair when last JSONL line is truncated

Example smoke run:
    uv run python -m src.exp13.run_signal_strength --max-examples 32 --batch-size 8

Example full run:
    uv run python -m src.exp13.run_signal_strength \\
      --models meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-4B-Instruct-2507 \\
      --tags Expert,Note,User,Someone\\ online \\
      --certainty-levels might,think,sure \\
      --salience-levels plain,important \\
      --instruction-schedules none,t0 \\
      --batch-size 32 \\
      --output-dir new-phase-results/exp13
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import torch
from tqdm import tqdm

from src.exp7.dataset_mc import MCExample, create_mc_examples, load_mc_dataset, save_mc_dataset
from src.exp7.scoring import ForcedChoiceResult, get_ab_token_ids, score_prompts_forced_choice_batch
from src.exp10.conditions import INSTRUCTION_TEXT
from src.exp13.conditions import (
    DEFAULT_CERTAINTY_LEVELS,
    DEFAULT_SALIENCE_LEVELS,
    DEFAULT_TAGS,
    METRIC_NAMES,
    compute_variant_metrics,
    format_all_conditions,
    generate_all_conditions,
    instruction_key,
    normalize_tag,
    parse_certainty_levels,
    parse_instruction_schedules,
    parse_salience_levels,
    parse_tags,
    variant_keys,
)
from src.models.llama_loader import load_model_and_tokenizer


EXP13_CODE_VERSION = "2026-02-12-v1"
DEFAULT_MODELS: Tuple[str, ...] = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
)
RESUME_CHOICES = ("auto", "never", "require")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp13: signal-strength endorsement experiment with resume/checkpointing"
    )
    parser.add_argument("--models", type=str, nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--data-path", type=Path, default=Path("data/answer.jsonl"))
    parser.add_argument("--mc-dataset-path", type=Path, default=Path("data/exp7_mc_dataset.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("new-phase-results/exp13"))
    parser.add_argument("--max-examples", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--tags",
        type=str,
        default=",".join(DEFAULT_TAGS),
        help="Comma-separated tags. Allowed: Expert,Note,User,Someone online",
    )
    parser.add_argument(
        "--certainty-levels",
        type=str,
        default=",".join(DEFAULT_CERTAINTY_LEVELS),
        help="Comma-separated certainty levels: might,think,sure",
    )
    parser.add_argument(
        "--salience-levels",
        type=str,
        default=",".join(DEFAULT_SALIENCE_LEVELS),
        help="Comma-separated salience levels: plain,important",
    )
    parser.add_argument(
        "--instruction-schedules",
        type=str,
        default="none,t0",
        help="Comma-separated instruction schedules: none,t0",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--save-prompts", action="store_true")

    parser.add_argument(
        "--resume",
        type=str,
        choices=RESUME_CHOICES,
        default="auto",
        help="Resume behavior: auto|never|require",
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--flush-every", type=int, default=5)
    parser.add_argument("--allow-config-mismatch", action="store_true")
    return parser.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_data_path(path: Path) -> Path:
    if path.exists():
        return path
    fallback = Path("external/sycophancy-eval/datasets/answer.jsonl")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Dataset not found at '{path}' or '{fallback}'.")


def _model_tag(model_id: str) -> str:
    return model_id.replace("/", "__")


def _order_seed(uid: str, seed: int) -> int:
    digest = hashlib.md5(f"{seed}:{uid}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _summarize_values(values: Iterable[float]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n": 0,
            "positive_frac": 0.0,
        }
    n_pos = sum(1 for v in vals if v > 0)
    return {
        "mean": float(mean(vals)),
        "median": float(median(vals)),
        "std": float(stdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
        "n": len(vals),
        "positive_frac": float(n_pos / len(vals)),
    }


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _dataset_id(examples: Sequence[MCExample]) -> str:
    joined = "\n".join(ex.uid for ex in examples)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _stable_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _fingerprint(config: Dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(config).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _load_and_repair_results(path: Path) -> Tuple[List[Dict[str, Any]], bool, int]:
    """Load JSONL results and repair only a corrupt trailing line if present."""
    if not path.exists():
        return [], False, 0

    with path.open("rb") as f:
        lines = f.readlines()

    records: List[Dict[str, Any]] = []
    truncate_at: int | None = None
    offset = 0
    for idx, raw in enumerate(lines):
        decoded = raw.decode("utf-8", errors="strict")
        stripped = decoded.strip()
        if not stripped:
            offset += len(raw)
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            if idx == len(lines) - 1:
                truncate_at = offset
                break
            raise ValueError(
                f"Corrupt JSONL at {path}:{idx + 1} (not trailing line); aborting resume."
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid record type at {path}:{idx + 1}; expected object.")
        records.append(payload)
        offset += len(raw)

    repaired = False
    if truncate_at is not None:
        with path.open("r+b") as f:
            f.truncate(truncate_at)
            f.flush()
            os.fsync(f.fileno())
        repaired = True

    return records, repaired, len(lines)


def _dedupe_records_by_uid(records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen: Set[str] = set()
    unique: List[Dict[str, Any]] = []
    duplicates = 0
    for record in records:
        uid = str(record.get("uid", ""))
        if not uid:
            continue
        if uid in seen:
            duplicates += 1
            continue
        seen.add(uid)
        unique.append(record)
    return unique, duplicates


def _record_metric_accumulators(
    records: Sequence[Dict[str, Any]],
    condition_codes: Sequence[str],
    tag_keys: Sequence[str],
    variants: Sequence[str],
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, Dict[str, List[float]]]], Set[str]]:
    per_condition_fc_correct = {code: [] for code in condition_codes}
    metric_values: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        tag: {
            variant: {metric: [] for metric in METRIC_NAMES}
            for variant in variants
        }
        for tag in tag_keys
    }
    seen_uids: Set[str] = set()

    for record in records:
        uid = str(record.get("uid", ""))
        if uid:
            seen_uids.add(uid)

        fc_correct = record.get("fc_correct", {})
        if isinstance(fc_correct, dict):
            for code in condition_codes:
                if code in fc_correct:
                    per_condition_fc_correct[code].append(float(fc_correct[code]))

        signal_metrics = record.get("signal_metrics", {})
        if not isinstance(signal_metrics, dict):
            continue
        for tag in tag_keys:
            tag_payload = signal_metrics.get(tag, {})
            if not isinstance(tag_payload, dict):
                continue
            for variant in variants:
                variant_payload = tag_payload.get(variant, {})
                if not isinstance(variant_payload, dict):
                    continue
                for metric in METRIC_NAMES:
                    if metric in variant_payload:
                        metric_values[tag][variant][metric].append(float(variant_payload[metric]))

    return per_condition_fc_correct, metric_values, seen_uids


def _update_accumulators_from_record(
    *,
    record: Dict[str, Any],
    condition_codes: Sequence[str],
    per_condition_fc_correct: Dict[str, List[float]],
    metric_values: Dict[str, Dict[str, Dict[str, List[float]]]],
) -> None:
    fc_correct = record["fc_correct"]
    for code in condition_codes:
        per_condition_fc_correct[code].append(float(fc_correct[code]))

    for tag_key, variant_payload in record["signal_metrics"].items():
        for variant, metrics in variant_payload.items():
            for metric in METRIC_NAMES:
                metric_values[tag_key][variant][metric].append(float(metrics[metric]))


def _build_summary(
    *,
    model_id: str,
    n_examples: int,
    tags: Sequence[str],
    certainty_levels: Sequence[str],
    salience_levels: Sequence[str],
    instruction_schedules: Sequence[str],
    token_id_a: int,
    token_id_b: int,
    tokenizer,
    per_condition_fc_correct: Dict[str, List[float]],
    metric_values: Dict[str, Dict[str, Dict[str, List[float]]]],
    requested_batch_size: int,
    effective_batch_size: int,
    resumed_from_records: int,
    checkpoint_enabled: bool,
) -> Dict[str, Any]:
    return {
        "experiment": "exp13_signal_strength",
        "code_version": EXP13_CODE_VERSION,
        "model": model_id,
        "n_examples": n_examples,
        "tags": list(tags),
        "certainty_levels": list(certainty_levels),
        "salience_levels": list(salience_levels),
        "instruction_schedules": list(instruction_schedules),
        "instruction_text": INSTRUCTION_TEXT,
        "token_ids": {
            "a": token_id_a,
            "b": token_id_b,
            "a_str": tokenizer.decode([token_id_a]),
            "b_str": tokenizer.decode([token_id_b]),
        },
        "metrics": {
            tag_key: {
                variant: {
                    metric: _summarize_values(values)
                    for metric, values in metric_values[tag_key][variant].items()
                }
                for variant in metric_values[tag_key]
            }
            for tag_key in metric_values
        },
        "per_condition_fc_correct": {
            code: _summarize_values(values)
            for code, values in per_condition_fc_correct.items()
        },
        "runtime": {
            "requested_batch_size": requested_batch_size,
            "effective_batch_size": effective_batch_size,
            "resumed_from_records": resumed_from_records,
            "checkpoint_enabled": checkpoint_enabled,
        },
    }


def _score_prompts_with_fallback(
    *,
    model,
    tokenizer,
    device: torch.device,
    token_id_a: int,
    token_id_b: int,
    prompts: List[str],
    active_batch_size: int,
    max_length: int,
) -> Tuple[List[ForcedChoiceResult], int]:
    def _is_cuda_oom(exc: Exception) -> bool:
        if isinstance(exc, torch.OutOfMemoryError):
            return True
        msg = str(exc).lower()
        return "out of memory" in msg and "cuda" in msg

    current_bs = active_batch_size
    while True:
        try:
            scored = score_prompts_forced_choice_batch(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=prompts,
                device=device,
                token_id_a=token_id_a,
                token_id_b=token_id_b,
                batch_size=current_bs,
                max_length=max_length if max_length > 0 else None,
            )
            return scored, current_bs
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            if current_bs <= 4:
                raise
            next_bs = max(4, current_bs // 2)
            if next_bs == current_bs:
                next_bs = current_bs - 1
            print(f"WARNING: CUDA OOM at batch_size={current_bs}; retrying with batch_size={next_bs}.")
            current_bs = next_bs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _write_checkpoint(
    *,
    path: Path,
    model_id: str,
    fingerprint: str,
    total_examples: int,
    processed_examples: int,
    started_at: str,
    active_batch_size: int,
    resume_mode: str,
    completed: bool,
    last_uid: str | None,
) -> None:
    payload = {
        "experiment": "exp13_signal_strength",
        "code_version": EXP13_CODE_VERSION,
        "model": model_id,
        "fingerprint": fingerprint,
        "started_at": started_at,
        "updated_at": _utc_now_iso(),
        "total_examples": total_examples,
        "processed_examples": processed_examples,
        "remaining_examples": max(total_examples - processed_examples, 0),
        "effective_batch_size": active_batch_size,
        "resume_mode": resume_mode,
        "completed": completed,
        "last_uid": last_uid,
    }
    _atomic_write_json(path, payload)


def run_for_model(
    *,
    model_id: str,
    examples: Sequence[MCExample],
    output_dir: Path,
    seed: int,
    tags: Sequence[str],
    certainty_levels: Sequence[str],
    salience_levels: Sequence[str],
    instruction_schedules: Sequence[str],
    batch_size: int,
    max_length: int,
    save_prompts: bool,
    resume_mode: str,
    checkpoint_every: int,
    flush_every: int,
    allow_config_mismatch: bool,
) -> Dict[str, Any]:
    model_tag = _model_tag(model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_tag}_results.jsonl"
    summary_path = output_dir / f"{model_tag}_summary.json"
    checkpoint_path = output_dir / f"{model_tag}_checkpoint.json"
    manifest_path = output_dir / f"{model_tag}_run_manifest.json"

    tag_keys = [normalize_tag(t) for t in tags]
    variants = variant_keys(certainty_levels, salience_levels)
    i_keys = [instruction_key(s) for s in instruction_schedules]

    run_config = {
        "experiment": "exp13_signal_strength",
        "code_version": EXP13_CODE_VERSION,
        "model": model_id,
        "seed": seed,
        "dataset_id": _dataset_id(examples),
        "n_examples": len(examples),
        "tags": list(tags),
        "certainty_levels": list(certainty_levels),
        "salience_levels": list(salience_levels),
        "instruction_schedules": list(instruction_schedules),
        "instruction_keys": list(i_keys),
    }
    fingerprint = _fingerprint(run_config)

    has_existing = out_path.exists() or checkpoint_path.exists() or manifest_path.exists()
    if resume_mode == "never" and has_existing:
        raise RuntimeError(
            f"Existing run artifacts found for {model_id} in {output_dir}. "
            "Use --resume auto/require or move to a new output directory."
        )
    if resume_mode == "require" and not has_existing:
        raise RuntimeError(
            f"Resume required but no existing artifacts found for model {model_id} in {output_dir}."
        )

    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        existing_fingerprint = manifest.get("fingerprint")
        if existing_fingerprint != fingerprint and not allow_config_mismatch:
            raise RuntimeError(
                "Run config fingerprint mismatch.\n"
                f"  Existing: {existing_fingerprint}\n"
                f"  Current : {fingerprint}\n"
                "Use --allow-config-mismatch only if you intentionally want to continue with changed config."
            )
    else:
        if resume_mode == "require":
            raise RuntimeError(f"Missing manifest for resume-required run: {manifest_path}")
        _atomic_write_json(
            manifest_path,
            {
                "experiment": "exp13_signal_strength",
                "code_version": EXP13_CODE_VERSION,
                "created_at": _utc_now_iso(),
                "fingerprint": fingerprint,
                "config": run_config,
            },
        )

    existing_records, repaired_tail, loaded_lines = _load_and_repair_results(out_path)
    if repaired_tail:
        print(f"WARNING: Repaired truncated trailing JSONL line in {out_path}.")
    if loaded_lines > 0:
        print(f"Loaded {len(existing_records)} raw records from {out_path}.")

    deduped_records, duplicate_count = _dedupe_records_by_uid(existing_records)
    if duplicate_count > 0:
        print(f"WARNING: Ignored {duplicate_count} duplicate UID records in existing JSONL.")

    conditions = generate_all_conditions(tags, certainty_levels, salience_levels, i_keys)
    condition_codes = [condition.code for condition in conditions]

    per_condition_fc_correct, metric_values, seen_uids = _record_metric_accumulators(
        deduped_records,
        condition_codes,
        tag_keys,
        variants,
    )

    pending_examples = [ex for ex in examples if ex.uid not in seen_uids]
    print(f"\n{'=' * 70}")
    print(f"Running Exp13 model: {model_id}")
    print(f"Tags: {list(tags)}")
    print(f"Variants: {variants}")
    print(f"Instruction schedules: {list(instruction_schedules)}")
    print(f"Total examples: {len(examples)}")
    print(f"Already complete: {len(seen_uids)}")
    print(f"Pending: {len(pending_examples)}")
    print(f"{'=' * 70}")

    model, tokenizer = load_model_and_tokenizer(model_name=model_id, device="auto", dtype="auto")
    model.eval()
    device = next(model.parameters()).device
    token_id_a, token_id_b = get_ab_token_ids(tokenizer)
    print(f"Token A: {tokenizer.decode([token_id_a])!r} (id={token_id_a})")
    print(f"Token B: {tokenizer.decode([token_id_b])!r} (id={token_id_b})")

    active_batch_size = batch_size
    checkpoint_enabled = checkpoint_every > 0
    started_at = _utc_now_iso()
    if checkpoint_path.exists():
        ckpt = _load_json(checkpoint_path)
        started_at = str(ckpt.get("started_at", started_at))

    completed_new = 0
    last_uid: str | None = None
    with out_path.open("a") as out_file:
        for ex in tqdm(pending_examples, desc=f"Exp13 {model_id}"):
            prompts_by_code = format_all_conditions(ex, conditions)
            ordered_codes = list(prompts_by_code.keys())
            ordered_prompts = [prompts_by_code[code] for code in ordered_codes]

            scored, active_batch_size = _score_prompts_with_fallback(
                model=model,
                tokenizer=tokenizer,
                device=device,
                token_id_a=token_id_a,
                token_id_b=token_id_b,
                prompts=ordered_prompts,
                active_batch_size=active_batch_size,
                max_length=max_length,
            )

            condition_results: Dict[str, Dict[str, float]] = {}
            fc_correct: Dict[str, float] = {}
            fc_wrong: Dict[str, float] = {}
            for code, result in zip(ordered_codes, scored):
                condition_results[code] = {
                    "logit_a": float(result.logit_a),
                    "logit_b": float(result.logit_b),
                    "fc_a": float(result.fc_a),
                    "fc_b": float(result.fc_b),
                }
                if ex.correct_label == "A":
                    fc_correct[code] = float(result.fc_a)
                    fc_wrong[code] = float(result.fc_b)
                else:
                    fc_correct[code] = float(result.fc_b)
                    fc_wrong[code] = float(result.fc_a)

            signal_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
            for tag in tags:
                tag_key = normalize_tag(tag)
                signal_metrics[tag_key] = {}
                for certainty in certainty_levels:
                    for salience in salience_levels:
                        variant = f"{certainty}_{salience}"
                        signal_metrics[tag_key][variant] = compute_variant_metrics(
                            fc_correct, tag, certainty, salience
                        )

            record: Dict[str, Any] = {
                "model": model_id,
                "uid": ex.uid,
                "question": ex.question,
                "correct_answer": ex.correct_answer,
                "wrong_answer": ex.wrong_answer,
                "correct_label": ex.correct_label,
                "wrong_label": ex.wrong_label,
                "order_seed": _order_seed(ex.uid, seed=seed),
                "condition_results": condition_results,
                "fc_correct": fc_correct,
                "fc_wrong": fc_wrong,
                "signal_metrics": signal_metrics,
                "metadata": ex.metadata,
            }
            if save_prompts:
                record["prompts"] = prompts_by_code

            out_file.write(json.dumps(record) + "\n")
            completed_new += 1
            seen_uids.add(ex.uid)
            last_uid = ex.uid
            _update_accumulators_from_record(
                record=record,
                condition_codes=condition_codes,
                per_condition_fc_correct=per_condition_fc_correct,
                metric_values=metric_values,
            )

            if flush_every > 0 and completed_new % flush_every == 0:
                out_file.flush()
                os.fsync(out_file.fileno())

            if checkpoint_enabled and completed_new % checkpoint_every == 0:
                summary = _build_summary(
                    model_id=model_id,
                    n_examples=len(examples),
                    tags=tags,
                    certainty_levels=certainty_levels,
                    salience_levels=salience_levels,
                    instruction_schedules=instruction_schedules,
                    token_id_a=token_id_a,
                    token_id_b=token_id_b,
                    tokenizer=tokenizer,
                    per_condition_fc_correct=per_condition_fc_correct,
                    metric_values=metric_values,
                    requested_batch_size=batch_size,
                    effective_batch_size=active_batch_size,
                    resumed_from_records=len(deduped_records),
                    checkpoint_enabled=True,
                )
                _atomic_write_json(summary_path, summary)
                _write_checkpoint(
                    path=checkpoint_path,
                    model_id=model_id,
                    fingerprint=fingerprint,
                    total_examples=len(examples),
                    processed_examples=len(seen_uids),
                    started_at=started_at,
                    active_batch_size=active_batch_size,
                    resume_mode=resume_mode,
                    completed=False,
                    last_uid=last_uid,
                )

        out_file.flush()
        os.fsync(out_file.fileno())

    summary = _build_summary(
        model_id=model_id,
        n_examples=len(examples),
        tags=tags,
        certainty_levels=certainty_levels,
        salience_levels=salience_levels,
        instruction_schedules=instruction_schedules,
        token_id_a=token_id_a,
        token_id_b=token_id_b,
        tokenizer=tokenizer,
        per_condition_fc_correct=per_condition_fc_correct,
        metric_values=metric_values,
        requested_batch_size=batch_size,
        effective_batch_size=active_batch_size,
        resumed_from_records=len(deduped_records),
        checkpoint_enabled=checkpoint_enabled,
    )
    _atomic_write_json(summary_path, summary)
    _write_checkpoint(
        path=checkpoint_path,
        model_id=model_id,
        fingerprint=fingerprint,
        total_examples=len(examples),
        processed_examples=len(seen_uids),
        started_at=started_at,
        active_batch_size=active_batch_size,
        resume_mode=resume_mode,
        completed=(len(seen_uids) == len(examples)),
        last_uid=last_uid,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nCompleted Exp13 for {model_id}: {len(seen_uids)}/{len(examples)} examples")
    print(f"Summary written to {summary_path}")
    return summary


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.max_examples < 0:
        raise ValueError("--max-examples must be >= 0")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")
    if args.flush_every < 0:
        raise ValueError("--flush-every must be >= 0")

    tags = parse_tags(args.tags)
    certainty_levels = parse_certainty_levels(args.certainty_levels)
    salience_levels = parse_salience_levels(args.salience_levels)
    instruction_schedules = parse_instruction_schedules(args.instruction_schedules)

    print("Exp13: Signal-Strength Endorsement Experiment")
    print(f"Models: {args.models}")
    print(f"Tags: {tags}")
    print(f"Certainty levels: {certainty_levels}")
    print(f"Salience levels: {salience_levels}")
    print(f"Instruction schedules: {instruction_schedules}")
    print(f"Resume mode: {args.resume}")
    print(f"Checkpoint every: {args.checkpoint_every}")
    print(f"Flush every: {args.flush_every}")

    mc_path = args.mc_dataset_path
    if mc_path.exists():
        print(f"Loading existing MC dataset from {mc_path}")
        examples = load_mc_dataset(mc_path)
    else:
        raw_path = _resolve_data_path(args.data_path)
        print(f"Creating MC dataset from {raw_path}")
        examples = create_mc_examples(raw_path, seed=args.seed)
        save_mc_dataset(examples, mc_path)
        print(f"Saved MC dataset to {mc_path}")

    if args.max_examples > 0 and len(examples) > args.max_examples:
        rng = random.Random(args.seed)
        shuffled = list(examples)
        rng.shuffle(shuffled)
        examples = shuffled[: args.max_examples]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    item_table_path = args.output_dir / "item_table.jsonl"
    with item_table_path.open("w") as f:
        for ex in examples:
            row = {
                "uid": ex.uid,
                "dataset": ex.metadata.get("dataset", "unknown"),
                "correct_label": ex.correct_label,
                "wrong_label": ex.wrong_label,
                "opt_a_text": ex.option_a,
                "opt_b_text": ex.option_b,
                "order_seed": _order_seed(ex.uid, seed=args.seed),
            }
            f.write(json.dumps(row) + "\n")

    summaries: List[Dict[str, Any]] = []
    for model_id in args.models:
        summary = run_for_model(
            model_id=model_id,
            examples=examples,
            output_dir=args.output_dir,
            seed=args.seed,
            tags=tags,
            certainty_levels=certainty_levels,
            salience_levels=salience_levels,
            instruction_schedules=instruction_schedules,
            batch_size=args.batch_size,
            max_length=args.max_length,
            save_prompts=args.save_prompts,
            resume_mode=args.resume,
            checkpoint_every=args.checkpoint_every,
            flush_every=args.flush_every,
            allow_config_mismatch=args.allow_config_mismatch,
        )
        summaries.append(summary)

    combined = {
        "experiment": "exp13_signal_strength",
        "code_version": EXP13_CODE_VERSION,
        "description": "Certainty x salience signal-strength experiment with resume/checkpointing.",
        "design": "3x2 phrasing variants over wrong/correct endorsement with neutral baseline under none/t0 schedules",
        "n_examples": len(examples),
        "tags_tested": tags,
        "certainty_levels": certainty_levels,
        "salience_levels": salience_levels,
        "instruction_schedules": instruction_schedules,
        "instruction_text": INSTRUCTION_TEXT,
        "metrics": {
            "effect_wrong_I0": "N0 - W0 (positive means wrong endorsement moved model toward wrong answer)",
            "effect_wrong_I1": "N1 - W1",
            "effect_correct_I0": "C0 - N0 (positive means correct endorsement moved model toward correct answer)",
            "effect_correct_I1": "C1 - N1",
            "efficacy_wrong": "effect_wrong_I0 - effect_wrong_I1",
            "efficacy_correct": "effect_correct_I0 - effect_correct_I1",
            "selectivity": "efficacy_wrong - efficacy_correct",
            "baseline_shift": "N1 - N0",
        },
        "models": summaries,
    }
    _atomic_write_json(args.output_dir / "summary.json", combined)

    print(f"\n{'=' * 70}")
    print("Exp13 run complete.")
    print(f"Results written to {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

