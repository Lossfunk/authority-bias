from __future__ import annotations

import argparse
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.authority_steering.run_steering_test import (
    TargetSpec,
    _dynamic_batches,
    _generate_steered_batch,
    _load_direction_file,
)
from src.direction_controls.common import default_model_specs, parse_csv, write_json
from src.mitigation.common import direction_for_layer, save_jsonl
from src.mechanism.hooks import get_component_modules, get_transformer_layers
from src.mechanism.hooks_v2 import setup_h100_optimizations, try_compile_model
from src.models.llama_loader import load_model_and_tokenizer


LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MMLU_PRO_DATASET = "TIGER-Lab/MMLU-Pro"
GSM8K_DATASET = "openai/gsm8k"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capability eval under Exp20 interventions.")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--model-name", required=True, help="Key from exp19.default_model_specs")
    p.add_argument("--direction-path", type=Path, required=True)
    p.add_argument("--direction-key", default="vector_by_layer")
    p.add_argument("--task", choices=("mmlu_pro", "mmlu", "gsm8k"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--layers", required=True, help="Comma-separated layers")
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    p.add_argument("--mode", choices=("project_out_direction", "subtract", "add"), default="project_out_direction")
    p.add_argument("--position-mode", default="answer_position")
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-batch-tokens", type=int, default=65536)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--answer-suffix", default=None)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _sample(ds, max_samples: int, seed: int):
    if max_samples <= 0 or max_samples >= len(ds):
        return list(ds)
    return list(ds.shuffle(seed=seed).select(range(max_samples)))


def _task_answer_suffix(task: str, answer_suffix: str | None) -> str:
    if answer_suffix is not None:
        return answer_suffix
    if task in {"mmlu_pro", "mmlu"}:
        return "Do not explain. Do not show reasoning. Output exactly one option letter and nothing else."
    if task == "gsm8k":
        return "Do not explain. Do not show reasoning. Output exactly the final numeric answer and nothing else."
    raise ValueError(task)


def _format_mc_prompt(question: str, choices: Sequence[str], answer_suffix: str) -> str:
    options = "\n".join(f"({LABELS[i]}) {choice}" for i, choice in enumerate(choices))
    return f"{question.strip()}\n{options}\n{answer_suffix}".strip()


def load_mmlu(max_samples: int, seed: int, answer_suffix: str) -> list[dict[str, Any]]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    rows = []
    for i, item in enumerate(_sample(ds, max_samples, seed)):
        choices = list(item["choices"])
        answer_idx = int(item["answer"])
        prompt = _format_mc_prompt(item["question"], choices, answer_suffix)
        rows.append(
            {
                "uid": f"mmlu::{i}",
                "task": "mmlu",
                "prompt_text": prompt,
                "correct_label": LABELS[answer_idx],
                "choices": choices,
                "subject": item.get("subject"),
            }
        )
    return rows


def _mmlu_pro_choices(item: dict[str, Any]) -> list[str]:
    raw = item.get("options") or item.get("choices")
    if raw is None:
        raw = [item[k] for k in sorted(item) if re.fullmatch(r"option_[a-zA-Z0-9]+", str(k)) and item[k]]
    choices = list(raw)
    if len(choices) < 2:
        raise ValueError(f"MMLU-Pro row has <2 choices: keys={sorted(item)}")
    return [str(x) for x in choices]


def _mmlu_pro_answer_idx(item: dict[str, Any], choices: Sequence[str]) -> int:
    for key in ("answer_index", "answer_idx", "label"):
        if key in item and item[key] is not None:
            val = item[key]
            if isinstance(val, int):
                return int(val)
            if str(val).strip().isdigit():
                return int(str(val).strip())
    ans = str(item.get("answer", "")).strip()
    if len(ans) == 1 and ans.upper() in LABELS:
        return LABELS.index(ans.upper())
    if ans in choices:
        return list(choices).index(ans)
    raise ValueError(f"Cannot infer MMLU-Pro answer index from answer={ans!r}")


def load_mmlu_pro(max_samples: int, seed: int, answer_suffix: str) -> list[dict[str, Any]]:
    ds = load_dataset(MMLU_PRO_DATASET, split="test")
    rows = []
    for i, item in enumerate(_sample(ds, max_samples, seed)):
        choices = _mmlu_pro_choices(item)
        answer_idx = _mmlu_pro_answer_idx(item, choices)
        prompt = _format_mc_prompt(str(item["question"]), choices, answer_suffix)
        rows.append(
            {
                "uid": f"mmlu_pro::{i}",
                "task": "mmlu_pro",
                "prompt_text": prompt,
                "correct_label": LABELS[answer_idx],
                "choices": choices,
                "subject": item.get("category") or item.get("subject"),
            }
        )
    return rows


def _gsm_answer(text: str) -> str | None:
    m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"final answer\s*[:is-]*\s*([-+]?\d[\d,]*(?:\.\d+)?)", text, flags=re.I)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def _normalize_number(text: str | None) -> str | None:
    if text is None:
        return None
    try:
        dec = Decimal(str(text).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return str(text).replace(",", "").strip()
    return format(dec.normalize(), "f").rstrip("0").rstrip(".") if "." in format(dec.normalize(), "f") else str(dec)


def load_gsm8k(max_samples: int, seed: int, answer_suffix: str) -> list[dict[str, Any]]:
    ds = load_dataset(GSM8K_DATASET, "main", split="test")
    rows = []
    for i, item in enumerate(_sample(ds, max_samples, seed)):
        rows.append(
            {
                "uid": f"gsm8k::{i}",
                "task": "gsm8k",
                "prompt_text": f"{item['question'].strip()}\n{answer_suffix}",
                "correct_label": _normalize_number(_gsm_answer(item["answer"])),
                "choices": None,
            }
        )
    return rows


def parse_mc_letter(text: str, labels: str) -> str | None:
    text = text.strip()
    escaped = re.escape(labels)
    patterns = [
        rf"(?:final\s+answer|answer|option|choice)\s*(?:is|:|-)?\s*\(?([{escaped}])\)?\b",
        rf"^\s*\(?([{escaped}])\)?[\).:\s]",
        rf"\b([{escaped}])\)",
        rf"\b([{escaped}])\b",
    ]
    matches: list[str] = []
    for pat in patterns:
        matches.extend(m.group(1).upper() for m in re.finditer(pat, text, flags=re.I | re.M))
        if matches:
            return matches[-1]
    return None


def parse_gsm8k(text: str) -> str | None:
    return _normalize_number(_gsm_answer(text))


def build_jobs(rows: Sequence[dict[str, Any]], tokenizer) -> list[dict[str, Any]]:
    jobs = []
    for row in rows:
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            model_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt_text"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            model_prompt = row["prompt_text"]
        jobs.append(
            {
                **row,
                "model_prompt": model_prompt,
                "condition_code": row["task"],
                "wrong_label": "",
                "correct_text": str(row["correct_label"]),
                "wrong_text": "",
                "positions_unpadded": {"answer_position": len(tokenizer(model_prompt, add_special_tokens=False).input_ids) - 1},
            }
        )
    return jobs


def summarize(rows: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    for row in rows:
        if task in {"mmlu_pro", "mmlu"}:
            n_choices = len(row.get("choices") or [])
            labels = LABELS[:n_choices] if n_choices else (LABELS[:10] if task == "mmlu_pro" else LABELS[:4])
            pred = parse_mc_letter(row["raw_text"], labels)
        else:
            pred = parse_gsm8k(row["raw_text"])
        row["parsed_answer"] = pred
        row["is_correct"] = pred == str(row["correct_label"]) if pred is not None else None
    summary = []
    for cfg in sorted({row["config_id"] for row in rows}):
        for alpha in sorted({float(row["alpha"]) for row in rows if row["config_id"] == cfg}):
            group = [row for row in rows if row["config_id"] == cfg and float(row["alpha"]) == alpha]
            parsed = [row for row in group if row["parsed_answer"] is not None]
            correct = [row for row in parsed if row["is_correct"]]
            summary.append(
                {
                    "config_id": cfg,
                    "alpha": alpha,
                    "task": task,
                    "total": len(group),
                    "parsed": len(parsed),
                    "parse_rate": len(parsed) / len(group) if group else 0.0,
                    "accuracy": len(correct) / len(parsed) if parsed else None,
                }
            )
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_h100_optimizations()
    spec = default_model_specs(args.repo_root)[args.model_name]
    layers = [int(x) for x in parse_csv(args.layers)]
    alphas = [float(x) for x in parse_csv(args.alphas)]
    answer_suffix = _task_answer_suffix(args.task, args.answer_suffix)
    direction_obj, _, _ = _load_direction_file(args.direction_path, args.direction_key)
    model, tokenizer = load_model_and_tokenizer(spec.model_id, device="auto")
    model.eval()
    if not args.no_compile:
        model = try_compile_model(model)
    _ = get_transformer_layers(model)
    component_modules = get_component_modules(model)
    if args.task == "mmlu_pro":
        rows = load_mmlu_pro(args.max_samples, args.seed, answer_suffix)
    elif args.task == "mmlu":
        rows = load_mmlu(args.max_samples, args.seed, answer_suffix)
    else:
        rows = load_gsm8k(args.max_samples, args.seed, answer_suffix)
    jobs = build_jobs(rows, tokenizer)
    rows_path = args.output_dir / f"{args.task}_rows.jsonl"
    if rows_path.exists() and not args.force:
        all_rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        print(f"[resume] loaded {len(all_rows)} existing rows from {rows_path}")
    else:
        all_rows = []
    expected_per_alpha = len(jobs)
    t0 = time.time()
    for layer in layers:
        direction = direction_for_layer(direction_obj, layer)
        cfg = f"{args.task}_{args.mode}_L{layer}_{args.position_mode}"
        target = [TargetSpec(layer=layer, component="block_output", direction=direction, patch_mean=None)]
        for alpha in alphas:
            existing = [
                row for row in all_rows
                if row.get("config_id") == cfg and float(row.get("alpha", 0.0)) == float(alpha)
            ]
            if len(existing) >= expected_per_alpha and not args.force:
                print(f"[resume] skipped complete {cfg} alpha={alpha} ({len(existing)} rows)")
                continue
            alpha_rows: list[dict[str, Any]] = []
            batches = _dynamic_batches(jobs, max_batch_size=args.batch_size, max_batch_tokens=args.max_batch_tokens)
            for batch_idx, chunk in enumerate(tqdm(batches, desc=f"{cfg}:a={alpha}")):
                generated = _generate_steered_batch(
                    chunk,
                    model=model,
                    tokenizer=tokenizer,
                    component_modules=component_modules,
                    target_specs=target,
                    alpha=alpha,
                    position_mode=args.position_mode,
                    apply_phase="prompt",
                    intervention_mode=args.mode,
                    norm_scaling="none",
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=1.0,
                    top_k=50,
                    greedy=args.temperature <= 0,
                    use_cache=True,
                    seed=args.seed + batch_idx,
                    collect_margin_diagnostics=False,
                    token_id_a=None,
                    token_id_b=None,
                )
                for row, job in zip(generated, chunk):
                    row["config_id"] = cfg
                    row["layer"] = layer
                    row["task"] = args.task
                    row["variant_mode"] = args.mode
                    if "choices" in job:
                        row["choices"] = job["choices"]
                    if "subject" in job:
                        row["subject"] = job["subject"]
                alpha_rows.extend(generated)
                all_rows = [
                    row for row in all_rows
                    if not (row.get("config_id") == cfg and float(row.get("alpha", 0.0)) == float(alpha))
                ] + alpha_rows
                save_jsonl(rows_path, all_rows)
    summary = summarize(all_rows, args.task)
    save_jsonl(rows_path, all_rows)
    write_json(
        args.output_dir / f"{args.task}_summary.json",
        {
            "summary": summary,
            "model": spec.model_id,
            "layers": layers,
            "task": args.task,
            "answer_suffix": answer_suffix,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "max_batch_tokens": args.max_batch_tokens,
            "elapsed_seconds": round(time.time() - t0, 2),
        },
    )
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
