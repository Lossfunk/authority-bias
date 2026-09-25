"""Exp14E runner for closed-source APIs without logit access.

Uses repeated sampling to estimate P(correct | condition).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

from src.lexical_controls.dataset_mc import MCExample, create_mc_examples, load_mc_dataset, save_mc_dataset
from src.evidence_quality.conditions import (
    Exp14Condition,
    EvidenceLevel,
    Tag,
    format_prompt,
    parse_evidence_levels,
    parse_tags,
)
from src.evidence_quality.reasons_dataset import load_reasons_dataset, match_reason_rows


RESUME_CHOICES = ("auto", "never", "require")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp14E closed-source repeated-sampling runner")
    parser.add_argument("--models", type=str, nargs="+", default=["gpt-4o", "claude-3-5-sonnet-latest"])
    parser.add_argument("--data-path", type=Path, default=Path("data/answer.jsonl"))
    parser.add_argument("--mc-dataset-path", type=Path, default=Path("data/exp7_mc_dataset.jsonl"))
    parser.add_argument("--reasons-path", type=Path, default=Path("data/exp14_reasons.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("new-phase-results/exp14e"))
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tags", type=str, default="Expert,Note")
    parser.add_argument(
        "--evidence-levels",
        type=str,
        default="bare,reason2",
        help="Comma-separated subset, usually bare,reason2",
    )
    parser.add_argument(
        "--include-matched-neutrals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include neutral conditions for each tested evidence level",
    )

    parser.add_argument("--samples-per-condition", type=int, default=100)
    parser.add_argument("--samples-per-request", type=int, default=10)
    parser.add_argument("--temperatures", type=str, default="0.7")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=8)

    parser.add_argument("--api-base", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--api-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=4)

    parser.add_argument("--resume", type=str, choices=RESUME_CHOICES, default="auto")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    return parser.parse_args()


def _parse_csv(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _resolve_data_path(path: Path) -> Path:
    if path.exists():
        return path
    fallback = Path("external/sycophancy-eval/datasets/answer.jsonl")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Dataset not found at '{path}' or '{fallback}'.")


def _model_tag(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def _stable_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _fingerprint(config: Dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(config).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _load_and_repair_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        lines = f.readlines()

    rows: List[Dict[str, Any]] = []
    truncate_at = None
    offset = 0
    for idx, raw in enumerate(lines):
        text = raw.decode("utf-8", errors="strict").strip()
        if not text:
            offset += len(raw)
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            if idx == len(lines) - 1:
                truncate_at = offset
                break
            raise
        if isinstance(payload, dict):
            rows.append(payload)
        offset += len(raw)

    if truncate_at is not None:
        with path.open("r+b") as f:
            f.truncate(truncate_at)
    return rows


def _normalize_tag(tag: str) -> str:
    return tag.lower().replace(" ", "_").replace("-", "_")


def _parse_temperatures(raw: str) -> List[float]:
    out = []
    for token in _parse_csv(raw):
        out.append(float(token))
    if not out:
        raise ValueError("No temperatures provided")
    return out


def _build_conditions(
    *,
    tags: Sequence[Tag],
    evidence_levels: Sequence[EvidenceLevel],
    include_matched_neutrals: bool,
) -> List[Exp14Condition]:
    conditions: List[Exp14Condition] = []
    for tag in tags:
        for level in evidence_levels:
            if include_matched_neutrals or level == "bare":
                conditions.append(
                    Exp14Condition(
                        endorse_type="neutral",
                        instruction_key="0",
                        tag=tag,
                        evidence_level=level,
                    )
                )
            conditions.append(
                Exp14Condition(
                    endorse_type="wrong",
                    instruction_key="0",
                    tag=tag,
                    evidence_level=level,
                )
            )
            conditions.append(
                Exp14Condition(
                    endorse_type="correct",
                    instruction_key="0",
                    tag=tag,
                    evidence_level=level,
                )
            )
    return conditions


def _task_key(uid: str, code: str, temperature: float) -> str:
    return f"{uid}::{code}::T{temperature:.3f}"


def _extract_choice(text: str) -> str:
    text = text.strip()
    if not text:
        return "abstain"
    match = re.search(r"\b([AB])\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    stripped = text.lstrip()
    if stripped and stripped[0].upper() in {"A", "B"}:
        return stripped[0].upper()
    return "abstain"


def _request_samples(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    n: int,
    max_output_tokens: int,
    timeout_seconds: float,
) -> List[str]:
    url = f"{api_base.rstrip('/')}/chat/completions"
    body = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "max_tokens": max_output_tokens,
        "messages": [
            {
                "role": "system",
                "content": "Respond with exactly one token: A or B.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }
    req = urllib.request.Request(
        url,
        method="POST",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise ValueError("No choices returned from API")

    outputs: List[str] = []
    for choice in choices:
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        outputs.append(str(content))
    return outputs


def _sample_condition(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    samples_total: int,
    samples_per_request: int,
    max_output_tokens: int,
    timeout_seconds: float,
    max_retries: int,
) -> Dict[str, int]:
    counts = {"A": 0, "B": 0, "abstain": 0}
    collected = 0

    while collected < samples_total:
        request_n = min(samples_per_request, samples_total - collected)
        last_exc: Exception | None = None
        outputs: List[str] | None = None

        for attempt in range(1, max_retries + 1):
            try:
                outputs = _request_samples(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    n=request_n,
                    max_output_tokens=max_output_tokens,
                    timeout_seconds=timeout_seconds,
                )
                break
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError) as exc:
                last_exc = exc
                if attempt >= max_retries:
                    break
                time.sleep(1.2 * attempt)

        if outputs is None:
            raise RuntimeError(f"Sampling failed after retries: {last_exc}")

        for text in outputs:
            choice = _extract_choice(text)
            counts[choice] = counts.get(choice, 0) + 1
            collected += 1
            if collected >= samples_total:
                break

    return counts


def _load_examples(args: argparse.Namespace) -> List[MCExample]:
    if args.mc_dataset_path.exists():
        examples = load_mc_dataset(args.mc_dataset_path)
    else:
        raw = _resolve_data_path(args.data_path)
        examples = create_mc_examples(raw, seed=args.seed)
        save_mc_dataset(examples, args.mc_dataset_path)

    if args.max_examples > 0 and len(examples) > args.max_examples:
        rng = random.Random(args.seed)
        shuffled = list(examples)
        rng.shuffle(shuffled)
        examples = shuffled[: args.max_examples]
    return examples


def _prepare_manifest(
    manifest_path: Path,
    *,
    run_config: Dict[str, Any],
    resume: str,
    allow_config_mismatch: bool,
) -> str:
    fingerprint = _fingerprint(run_config)
    if manifest_path.exists():
        existing = _load_json(manifest_path)
        if existing.get("fingerprint") != fingerprint and not allow_config_mismatch:
            raise RuntimeError(
                "Run config fingerprint mismatch. Use --allow-config-mismatch to continue intentionally."
            )
    else:
        if resume == "require":
            raise RuntimeError(f"Resume required but missing manifest: {manifest_path}")
        _atomic_write_json(
            manifest_path,
            {
                "experiment": "exp14_closed_source_sampling",
                "fingerprint": fingerprint,
                "config": run_config,
                "created_at": time.time(),
            },
        )
    return fingerprint


def run_model(args: argparse.Namespace, *, model_name: str, examples: Sequence[MCExample], reasons_payload: Dict[str, Dict[str, str]], conditions: Sequence[Exp14Condition]) -> Dict[str, Any]:
    model_tag = _model_tag(model_name)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{model_tag}_samples.jsonl"
    summary_path = out_dir / f"{model_tag}_summary.json"
    checkpoint_path = out_dir / f"{model_tag}_checkpoint.json"
    manifest_path = out_dir / f"{model_tag}_run_manifest.json"

    temperatures = _parse_temperatures(args.temperatures)
    run_config = {
        "experiment": "exp14_closed_source_sampling",
        "model": model_name,
        "n_examples": len(examples),
        "tags": _parse_csv(args.tags),
        "evidence_levels": _parse_csv(args.evidence_levels),
        "include_matched_neutrals": args.include_matched_neutrals,
        "samples_per_condition": args.samples_per_condition,
        "samples_per_request": args.samples_per_request,
        "temperatures": temperatures,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
    }
    fingerprint = _prepare_manifest(
        manifest_path,
        run_config=run_config,
        resume=args.resume,
        allow_config_mismatch=args.allow_config_mismatch,
    )

    if args.resume == "never" and (out_path.exists() or checkpoint_path.exists()):
        raise RuntimeError("Existing outputs detected with --resume never")
    if args.resume == "require" and not out_path.exists():
        raise RuntimeError("Resume required but result file is missing")

    existing_rows = _load_and_repair_jsonl(out_path)
    seen_tasks: Set[str] = set()
    for row in existing_rows:
        key = row.get("task_key")
        if isinstance(key, str):
            seen_tasks.add(key)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Missing API key env var '{args.api_key_env}'")

    tasks_total = len(examples) * len(conditions) * len(temperatures)
    tasks_done = len(seen_tasks)
    tasks_new = 0

    print(f"\n{'=' * 72}")
    print(f"Closed-source sampling model: {model_name}")
    print(f"Examples: {len(examples)}, conditions: {len(conditions)}, temperatures: {temperatures}")
    print(f"Total tasks: {tasks_total}, already complete: {tasks_done}")
    print(f"{'=' * 72}")

    with out_path.open("a") as out_file:
        for ex in examples:
            reason_payload = reasons_payload[ex.uid]
            for cond in conditions:
                prompt = format_prompt(ex, cond, reason_payload)
                for temperature in temperatures:
                    task_key = _task_key(ex.uid, cond.code, temperature)
                    if task_key in seen_tasks:
                        continue

                    counts = _sample_condition(
                        api_base=args.api_base,
                        api_key=api_key,
                        model=model_name,
                        prompt=prompt,
                        temperature=temperature,
                        top_p=args.top_p,
                        samples_total=args.samples_per_condition,
                        samples_per_request=args.samples_per_request,
                        max_output_tokens=args.max_output_tokens,
                        timeout_seconds=args.api_timeout_seconds,
                        max_retries=args.max_retries,
                    )

                    row = {
                        "experiment": "exp14_closed_source_sampling",
                        "model": model_name,
                        "uid": ex.uid,
                        "condition_code": cond.code,
                        "condition": {
                            "endorse_type": cond.endorse_type,
                            "instruction_key": cond.instruction_key,
                            "tag": cond.tag,
                            "evidence_level": cond.evidence_level,
                        },
                        "temperature": temperature,
                        "top_p": args.top_p,
                        "samples_requested": args.samples_per_condition,
                        "counts": counts,
                        "correct_label": ex.correct_label,
                        "wrong_label": ex.wrong_label,
                        "task_key": task_key,
                        "metadata": ex.metadata,
                    }
                    out_file.write(json.dumps(row) + "\n")
                    out_file.flush()

                    seen_tasks.add(task_key)
                    tasks_new += 1

                    if tasks_new % 25 == 0:
                        _atomic_write_json(
                            checkpoint_path,
                            {
                                "experiment": "exp14_closed_source_sampling",
                                "model": model_name,
                                "fingerprint": fingerprint,
                                "tasks_total": tasks_total,
                                "tasks_done": len(seen_tasks),
                                "tasks_new": tasks_new,
                                "updated_at": time.time(),
                            },
                        )
                        print(f"Progress {len(seen_tasks)}/{tasks_total} tasks")

    summary = {
        "experiment": "exp14_closed_source_sampling",
        "model": model_name,
        "tasks_total": tasks_total,
        "tasks_done": len(seen_tasks),
        "tasks_new": tasks_new,
        "samples_per_condition": args.samples_per_condition,
        "temperatures": temperatures,
        "conditions": [cond.code for cond in conditions],
    }
    _atomic_write_json(summary_path, summary)
    _atomic_write_json(
        checkpoint_path,
        {
            "experiment": "exp14_closed_source_sampling",
            "model": model_name,
            "fingerprint": fingerprint,
            "tasks_total": tasks_total,
            "tasks_done": len(seen_tasks),
            "tasks_new": tasks_new,
            "updated_at": time.time(),
            "completed": len(seen_tasks) == tasks_total,
        },
    )

    print(f"Completed {model_name}: {len(seen_tasks)}/{tasks_total} tasks")
    return summary


def main() -> None:
    args = parse_args()

    tags = parse_tags(args.tags, allow_extended=True)
    evidence_levels = parse_evidence_levels(args.evidence_levels)
    conditions = _build_conditions(
        tags=tags,
        evidence_levels=evidence_levels,
        include_matched_neutrals=args.include_matched_neutrals,
    )

    examples = _load_examples(args)
    reasons_by_uid = load_reasons_dataset(args.reasons_path)
    available_uids, missing_uids = match_reason_rows(
        uids=[ex.uid for ex in examples],
        reasons_by_uid=reasons_by_uid,
        policy="skip",
    )
    if missing_uids:
        print(f"Skipping {len(missing_uids)} items without reason rows")
    available_set = set(available_uids)
    examples = [ex for ex in examples if ex.uid in available_set]
    reasons_payload = {uid: reasons_by_uid[uid].to_prompt_payload() for uid in available_uids}

    summaries = []
    for model_name in args.models:
        summary = run_model(
            args,
            model_name=model_name,
            examples=examples,
            reasons_payload=reasons_payload,
            conditions=conditions,
        )
        summaries.append(summary)

    _atomic_write_json(args.output_dir / "summary.json", {"models": summaries})
    print(f"Wrote closed-source sampling outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
