"""Experiment 14A: Evidence-quality endorsement runner with resume/checkpointing.

Design factors:
- tags: Expert / Note (default; expandable)
- evidence levels: bare / reason1 / reason2 / reason_data
- instruction schedules: none / t0
- condition types: neutral / wrong endorsement / correct endorsement

This runner follows the Exp13 execution model:
- append-only per-model JSONL outputs
- per-model manifest with config fingerprint
- per-model checkpoint state
- truncated-tail JSONL repair for interrupted writes
- adaptive batch-size fallback on CUDA OOM
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

from src.lexical_controls.dataset_mc import MCExample, create_mc_examples, load_mc_dataset, save_mc_dataset
from src.lexical_controls.scoring import ForcedChoiceResult, get_ab_token_ids, score_prompts_forced_choice_batch
from src.evidence_quality.conditions import (
    DEFAULT_EVIDENCE_LEVELS,
    DEFAULT_TAGS,
    METRIC_NAMES,
    EvidenceLevel,
    compute_level_metrics,
    format_all_conditions,
    generate_all_conditions,
    instruction_key,
    normalize_tag,
    parse_evidence_levels,
    parse_instruction_schedules,
    parse_tags,
)
from src.evidence_quality.reasons_dataset import load_reasons_dataset, match_reason_rows
from src.models.llama_loader import load_model_and_tokenizer


EXP14_CODE_VERSION = "2026-02-15-v1"
DEFAULT_MODELS: Tuple[str, ...] = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507",
)
RESUME_CHOICES = ("auto", "never", "require")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp14A: evidence-quality endorsement experiment with resume/checkpointing"
    )
    parser.add_argument("--models", type=str, nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--data-path", type=Path, default=Path("data/answer.jsonl"))
    parser.add_argument("--mc-dataset-path", type=Path, default=Path("data/exp7_mc_dataset.jsonl"))
    parser.add_argument("--reasons-path", type=Path, default=Path("data/exp14_reasons.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("new-phase-results/evidence_quality"))
    parser.add_argument("--max-examples", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--tags",
        type=str,
        default=",".join(DEFAULT_TAGS),
        help="Comma-separated tags. Allowed: Expert,Note,User,Someone online",
    )
    parser.add_argument(
        "--evidence-levels",
        type=str,
        default=",".join(DEFAULT_EVIDENCE_LEVELS),
        help="Comma-separated evidence levels: bare,reason1,reason2,reason_data",
    )
    parser.add_argument(
        "--instruction-schedules",
        type=str,
        default="none,t0",
        help="Comma-separated instruction schedules: none,t0",
    )
    parser.add_argument(
        "--missing-reasons",
        type=str,
        choices=["skip", "error"],
        default="skip",
        help="How to handle MC examples that do not have a reason row.",
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

    parser.add_argument(
        "--use-torch-compile",
        action="store_true",
        help="Compile model forward graph for faster inference where supported.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        help="torch.compile mode (e.g., reduce-overhead, max-autotune).",
    )
    parser.add_argument(
        "--fullgraph-compile",
        action="store_true",
        help="Pass fullgraph=True to torch.compile.",
    )
    parser.add_argument(
        "--matmul-precision",
        type=str,
        choices=["highest", "high", "medium"],
        default="high",
        help="torch.set_float32_matmul_precision value.",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable CUDA TF32 kernels.",
    )
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


def _reason_dataset_id(examples: Sequence[MCExample], reason_payload_digest: Dict[str, str]) -> str:
    rows = [f"{ex.uid}:{reason_payload_digest[ex.uid]}" for ex in examples if ex.uid in reason_payload_digest]
    joined = "\n".join(rows)
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


def _init_metric_values(
    tag_keys: Sequence[str],
    evidence_levels: Sequence[EvidenceLevel],
    instruction_keys: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]:
    return {
        tag_key: {
            level: {
                f"I{i_key}": {metric: [] for metric in METRIC_NAMES}
                for i_key in instruction_keys
            }
            for level in evidence_levels
        }
        for tag_key in tag_keys
    }


def _record_metric_accumulators(
    records: Sequence[Dict[str, Any]],
    condition_codes: Sequence[str],
    tag_keys: Sequence[str],
    evidence_levels: Sequence[EvidenceLevel],
    instruction_keys: Sequence[str],
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]], Set[str]]:
    per_condition_fc_correct = {code: [] for code in condition_codes}
    metric_values = _init_metric_values(tag_keys, evidence_levels, instruction_keys)
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

        evidence_metrics = record.get("evidence_metrics", {})
        if not isinstance(evidence_metrics, dict):
            continue

        for tag_key in tag_keys:
            tag_payload = evidence_metrics.get(tag_key, {})
            if not isinstance(tag_payload, dict):
                continue
            for level in evidence_levels:
                level_payload = tag_payload.get(level, {})
                if not isinstance(level_payload, dict):
                    continue
                for i_key in instruction_keys:
                    instr_key = f"I{i_key}"
                    instr_payload = level_payload.get(instr_key, {})
                    if not isinstance(instr_payload, dict):
                        continue
                    for metric in METRIC_NAMES:
                        if metric in instr_payload:
                            metric_values[tag_key][level][instr_key][metric].append(
                                float(instr_payload[metric])
                            )

    return per_condition_fc_correct, metric_values, seen_uids


def _update_accumulators_from_record(
    *,
    record: Dict[str, Any],
    condition_codes: Sequence[str],
    per_condition_fc_correct: Dict[str, List[float]],
    metric_values: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
) -> None:
    fc_correct = record["fc_correct"]
    for code in condition_codes:
        per_condition_fc_correct[code].append(float(fc_correct[code]))

    for tag_key, tag_payload in record["evidence_metrics"].items():
        for level, level_payload in tag_payload.items():
            for instr_key, metrics in level_payload.items():
                for metric in METRIC_NAMES:
                    metric_values[tag_key][level][instr_key][metric].append(float(metrics[metric]))


def _build_summary(
    *,
    model_id: str,
    n_examples: int,
    n_skipped_missing_reasons: int,
    tags: Sequence[str],
    evidence_levels: Sequence[EvidenceLevel],
    instruction_schedules: Sequence[str],
    token_id_a: int,
    token_id_b: int,
    tokenizer,
    per_condition_fc_correct: Dict[str, List[float]],
    metric_values: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
    requested_batch_size: int,
    effective_batch_size: int,
    resumed_from_records: int,
    checkpoint_enabled: bool,
    use_torch_compile: bool,
    tf32_enabled: bool,
    matmul_precision: str,
) -> Dict[str, Any]:
    return {
        "experiment": "exp14_evidence_quality",
        "code_version": EXP14_CODE_VERSION,
        "model": model_id,
        "n_examples": n_examples,
        "n_skipped_missing_reasons": n_skipped_missing_reasons,
        "tags": list(tags),
        "evidence_levels": list(evidence_levels),
        "instruction_schedules": list(instruction_schedules),
        "token_ids": {
            "a": token_id_a,
            "b": token_id_b,
            "a_str": tokenizer.decode([token_id_a]),
            "b_str": tokenizer.decode([token_id_b]),
        },
        "metrics": {
            tag_key: {
                level: {
                    instr_key: {
                        metric: _summarize_values(values)
                        for metric, values in metric_values[tag_key][level][instr_key].items()
                    }
                    for instr_key in metric_values[tag_key][level]
                }
                for level in metric_values[tag_key]
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
            "use_torch_compile": use_torch_compile,
            "tf32_enabled": tf32_enabled,
            "matmul_precision": matmul_precision,
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
        "experiment": "exp14_evidence_quality",
        "code_version": EXP14_CODE_VERSION,
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


def _configure_torch_runtime(*, tf32: bool, matmul_precision: str) -> None:
    torch.set_float32_matmul_precision(matmul_precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32


def _maybe_compile_model(model, *, enabled: bool, mode: str, fullgraph: bool):
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        print("WARNING: torch.compile is unavailable in this runtime; using eager mode.")
        return model
    try:
        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph)
    except Exception as exc:
        print(f"WARNING: torch.compile failed ({exc}); falling back to eager mode.")
        return model
    print(f"Enabled torch.compile(mode={mode}, fullgraph={fullgraph}).")
    return compiled


def run_for_model(
    *,
    model_id: str,
    examples: Sequence[MCExample],
    reasons_payload_by_uid: Dict[str, Dict[str, str]],
    skipped_missing_reasons: int,
    output_dir: Path,
    seed: int,
    tags: Sequence[str],
    evidence_levels: Sequence[EvidenceLevel],
    instruction_schedules: Sequence[str],
    batch_size: int,
    max_length: int,
    save_prompts: bool,
    resume_mode: str,
    checkpoint_every: int,
    flush_every: int,
    allow_config_mismatch: bool,
    use_torch_compile: bool,
    compile_mode: str,
    fullgraph_compile: bool,
    tf32: bool,
    matmul_precision: str,
) -> Dict[str, Any]:
    model_tag = _model_tag(model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_tag}_results.jsonl"
    summary_path = output_dir / f"{model_tag}_summary.json"
    checkpoint_path = output_dir / f"{model_tag}_checkpoint.json"
    manifest_path = output_dir / f"{model_tag}_run_manifest.json"

    tag_keys = [normalize_tag(t) for t in tags]
    i_keys = [instruction_key(s) for s in instruction_schedules]

    reason_digest = {
        uid: hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        for uid, payload in reasons_payload_by_uid.items()
    }

    run_config = {
        "experiment": "exp14_evidence_quality",
        "code_version": EXP14_CODE_VERSION,
        "model": model_id,
        "seed": seed,
        "dataset_id": _dataset_id(examples),
        "reason_dataset_id": _reason_dataset_id(examples, reason_digest),
        "n_examples": len(examples),
        "tags": list(tags),
        "evidence_levels": list(evidence_levels),
        "instruction_schedules": list(instruction_schedules),
        "instruction_keys": list(i_keys),
        "tf32": tf32,
        "matmul_precision": matmul_precision,
        "use_torch_compile": use_torch_compile,
        "compile_mode": compile_mode,
        "fullgraph_compile": fullgraph_compile,
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
                "experiment": "exp14_evidence_quality",
                "code_version": EXP14_CODE_VERSION,
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

    conditions = generate_all_conditions(tags, evidence_levels, i_keys)
    condition_codes = [condition.code for condition in conditions]

    per_condition_fc_correct, metric_values, seen_uids = _record_metric_accumulators(
        deduped_records,
        condition_codes,
        tag_keys,
        evidence_levels,
        i_keys,
    )

    pending_examples = [ex for ex in examples if ex.uid not in seen_uids]
    print(f"\n{'=' * 74}")
    print(f"Running Exp14A model: {model_id}")
    print(f"Tags: {list(tags)}")
    print(f"Evidence levels: {list(evidence_levels)}")
    print(f"Instruction schedules: {list(instruction_schedules)}")
    print(f"Total examples: {len(examples)}")
    print(f"Already complete: {len(seen_uids)}")
    print(f"Pending: {len(pending_examples)}")
    print(f"{'=' * 74}")

    _configure_torch_runtime(tf32=tf32, matmul_precision=matmul_precision)

    model, tokenizer = load_model_and_tokenizer(model_name=model_id, device="auto", dtype="auto")
    model = _maybe_compile_model(
        model,
        enabled=use_torch_compile,
        mode=compile_mode,
        fullgraph=fullgraph_compile,
    )
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
        for ex in tqdm(pending_examples, desc=f"Exp14A {model_id}"):
            reason_payload = reasons_payload_by_uid[ex.uid]
            prompts_by_code = format_all_conditions(ex, conditions, reason_payload)
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
            logit_correct: Dict[str, float] = {}
            logit_wrong: Dict[str, float] = {}

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
                    logit_correct[code] = float(result.logit_a)
                    logit_wrong[code] = float(result.logit_b)
                else:
                    fc_correct[code] = float(result.fc_b)
                    fc_wrong[code] = float(result.fc_a)
                    logit_correct[code] = float(result.logit_b)
                    logit_wrong[code] = float(result.logit_a)

            evidence_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
            for tag in tags:
                tag_key = normalize_tag(tag)
                evidence_metrics[tag_key] = {}
                for level in evidence_levels:
                    evidence_metrics[tag_key][level] = {}
                    for i_key in i_keys:
                        instr_key = f"I{i_key}"
                        evidence_metrics[tag_key][level][instr_key] = compute_level_metrics(
                            fc_correct=fc_correct,
                            logit_correct=logit_correct,
                            tag=tag,
                            evidence_level=level,
                            instruction_key=i_key,
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
                "logit_correct": logit_correct,
                "logit_wrong": logit_wrong,
                "evidence_metrics": evidence_metrics,
                "reason_metadata": {
                    "reason_digest": reason_digest.get(ex.uid, ""),
                },
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
                    n_skipped_missing_reasons=skipped_missing_reasons,
                    tags=tags,
                    evidence_levels=evidence_levels,
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
                    use_torch_compile=use_torch_compile,
                    tf32_enabled=tf32,
                    matmul_precision=matmul_precision,
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
        n_skipped_missing_reasons=skipped_missing_reasons,
        tags=tags,
        evidence_levels=evidence_levels,
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
        use_torch_compile=use_torch_compile,
        tf32_enabled=tf32,
        matmul_precision=matmul_precision,
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

    print(f"\nCompleted Exp14A for {model_id}: {len(seen_uids)}/{len(examples)} examples")
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
    evidence_levels = parse_evidence_levels(args.evidence_levels)
    instruction_schedules = parse_instruction_schedules(args.instruction_schedules)

    print("Exp14A: Evidence-Quality Endorsement Experiment")
    print(f"Models: {args.models}")
    print(f"Tags: {tags}")
    print(f"Evidence levels: {evidence_levels}")
    print(f"Instruction schedules: {instruction_schedules}")
    print(f"Resume mode: {args.resume}")
    print(f"Checkpoint every: {args.checkpoint_every}")
    print(f"Flush every: {args.flush_every}")
    print(f"torch.compile: {args.use_torch_compile}")
    print(f"TF32: {args.tf32}, matmul precision: {args.matmul_precision}")

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

    reasons_by_uid = load_reasons_dataset(args.reasons_path)
    selected_uids = [ex.uid for ex in examples]
    available_uids, missing_uids = match_reason_rows(
        uids=selected_uids,
        reasons_by_uid=reasons_by_uid,
        policy=args.missing_reasons,
    )
    available_set = set(available_uids)
    examples = [ex for ex in examples if ex.uid in available_set]
    skipped_missing = len(missing_uids)
    if skipped_missing > 0:
        print(
            f"Skipped {skipped_missing} examples without reason rows "
            f"(policy={args.missing_reasons})."
        )

    reasons_payload_by_uid = {
        uid: reasons_by_uid[uid].to_prompt_payload()
        for uid in available_uids
    }

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
                "has_reason_row": ex.uid in reasons_payload_by_uid,
            }
            f.write(json.dumps(row) + "\n")

    summaries: List[Dict[str, Any]] = []
    for model_id in args.models:
        summary = run_for_model(
            model_id=model_id,
            examples=examples,
            reasons_payload_by_uid=reasons_payload_by_uid,
            skipped_missing_reasons=skipped_missing,
            output_dir=args.output_dir,
            seed=args.seed,
            tags=tags,
            evidence_levels=evidence_levels,
            instruction_schedules=instruction_schedules,
            batch_size=args.batch_size,
            max_length=args.max_length,
            save_prompts=args.save_prompts,
            resume_mode=args.resume,
            checkpoint_every=args.checkpoint_every,
            flush_every=args.flush_every,
            allow_config_mismatch=args.allow_config_mismatch,
            use_torch_compile=args.use_torch_compile,
            compile_mode=args.compile_mode,
            fullgraph_compile=args.fullgraph_compile,
            tf32=args.tf32,
            matmul_precision=args.matmul_precision,
        )
        summaries.append(summary)

    combined = {
        "experiment": "exp14_evidence_quality",
        "code_version": EXP14_CODE_VERSION,
        "description": "Evidence-quality scaling with crossed endorsement direction and matched neutral controls.",
        "design": (
            "3x4x2xT factorial: condition_type (neutral/wrong/correct) x evidence_level "
            "(bare/reason1/reason2/reason_data) x instruction (none/t0) x tag"
        ),
        "n_examples": len(examples),
        "n_skipped_missing_reasons": skipped_missing,
        "tags_tested": tags,
        "evidence_levels": evidence_levels,
        "instruction_schedules": instruction_schedules,
        "metrics": {
            "effect_wrong": "N - W (positive means wrong endorsement moved model toward wrong answer)",
            "effect_correct": "C - N (positive means correct endorsement moved model toward correct answer)",
            "asymmetry": "effect_correct - effect_wrong (positive means evidence helps truth more than falsehood)",
            "logit_shift_wrong": "logit_correct(N) - logit_correct(W)",
            "logit_shift_correct": "logit_correct(C) - logit_correct(N)",
            "logit_asymmetry": "logit_shift_correct - logit_shift_wrong",
        },
        "models": summaries,
    }
    _atomic_write_json(args.output_dir / "summary.json", combined)

    print(f"\n{'=' * 74}")
    print("Exp14A run complete.")
    print(f"Results written to {args.output_dir}")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    main()
