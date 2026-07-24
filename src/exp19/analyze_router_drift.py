from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from src.exp7.dataset_mc import load_mc_dataset
from src.exp10.conditions import generate_all_conditions
from src.exp16.run_dissociation_test import DEFAULT_INSTRUCTION, _format_exp16_prompt
from src.exp19.common import load_direction_payload
from src.exp19.router_utils import diff_stats, learn_router_biases, prepare_model_prompts, run_forward
from src.mechanism.hooks import get_component_modules
from src.mechanism.hooks_v2 import setup_h100_optimizations, try_compile_model
from src.models.llama_loader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze MoE router drift under hidden-state steering and router-bias interventions.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--dataset-path", type=Path, required=True)
    p.add_argument("--output-path", type=Path, required=True)
    p.add_argument("--max-items", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batch-tokens", type=int, default=8192)
    p.add_argument("--prep-n-jobs", type=int, default=1)
    p.add_argument("--torch-threads", type=int, default=0)
    p.add_argument("--plain-prompt", action="store_true")
    p.add_argument("--direction-path", type=Path, default=None)
    p.add_argument("--direction-key", type=str, default="vector")
    p.add_argument("--patch-alpha", type=float, default=0.5)
    p.add_argument("--patch-component", type=str, default="block_output")
    p.add_argument("--patch-position", choices=("all_prompt_tokens", "last_prompt_token"), default="all_prompt_tokens")
    p.add_argument("--router-name-pattern", type=str, default="router,gate")
    p.add_argument("--mode", choices=("drift", "router_bias"), default="drift")
    p.add_argument("--router-bias-alpha", type=float, default=1.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint-path", type=Path, default=None)
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args()


def _condition_map() -> Dict[str, object]:
    conds = generate_all_conditions(
        tags=["Expert", "Note", "User", "Someone online"],
        instruction_text=DEFAULT_INSTRUCTION,
    )
    return {c.code: c for c in conds}


def _build_prompts(dataset_path: Path, condition_code: str, n: int) -> List[str]:
    exs = load_mc_dataset(dataset_path)[:n]
    cond = _condition_map()[condition_code]
    return [_format_exp16_prompt(ex=ex, condition=cond, endorsement_style="authoritative_verified").rstrip() for ex in exs]


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    setup_h100_optimizations()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    model, tokenizer = load_model_and_tokenizer(model_name=args.model, device="auto", dtype="auto")
    model.eval()
    if not args.no_compile:
        model = try_compile_model(model)

    patterns = [p.strip().lower() for p in args.router_name_pattern.split(",") if p.strip()]
    checkpoint_path = args.checkpoint_path or args.output_path.with_suffix(".checkpoint.json")
    if args.resume and checkpoint_path.exists():
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    else:
        payload = {}

    def _save() -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    prepared_n0 = prepare_model_prompts(
        _build_prompts(args.dataset_path, "N0_note", args.max_items),
        tokenizer,
        args.plain_prompt,
        args.prep_n_jobs,
    )
    if "baseline" not in payload:
        baseline = run_forward(
            model=model,
            tokenizer=tokenizer,
            prepared_prompts=prepared_n0,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
            router_patterns=patterns,
        )
        payload["baseline"] = baseline
        _save()
    else:
        baseline = payload["baseline"]

    if args.mode == "drift":
        if args.direction_path is None:
            raise SystemExit("--direction-path required for drift mode.")
        vec, layer, _pos, _meta = load_direction_payload(args.direction_path, args.direction_key)
        module = get_component_modules(model)[(layer, args.patch_component)]
        if "patched" not in payload:
            patched = run_forward(
                model=model,
                tokenizer=tokenizer,
                prepared_prompts=prepared_n0,
                batch_size=args.batch_size,
                max_batch_tokens=args.max_batch_tokens,
                router_patterns=patterns,
                hidden_patch=(module, vec, args.patch_alpha, args.patch_position),
            )
            payload["patched"] = patched
            _save()
        else:
            patched = payload["patched"]
        payload = {"mode": "drift", "baseline": baseline, "patched": patched, "diff": diff_stats(baseline, patched)}
    else:
        prepared_c1 = prepare_model_prompts(
            _build_prompts(args.dataset_path, "C1_note", args.max_items), tokenizer, args.plain_prompt, args.prep_n_jobs
        )
        prepared_w1 = prepare_model_prompts(
            _build_prompts(args.dataset_path, "W1_note", args.max_items), tokenizer, args.plain_prompt, args.prep_n_jobs
        )
        if "baseline_c1" not in payload:
            c1_stats = run_forward(
                model=model,
                tokenizer=tokenizer,
                prepared_prompts=prepared_c1,
                batch_size=args.batch_size,
                max_batch_tokens=args.max_batch_tokens,
                router_patterns=patterns,
            )
            payload["baseline_c1"] = c1_stats
            _save()
        else:
            c1_stats = payload["baseline_c1"]
        if "baseline_w1" not in payload:
            w1_stats = run_forward(
                model=model,
                tokenizer=tokenizer,
                prepared_prompts=prepared_w1,
                batch_size=args.batch_size,
                max_batch_tokens=args.max_batch_tokens,
                router_patterns=patterns,
            )
            payload["baseline_w1"] = w1_stats
            _save()
        else:
            w1_stats = payload["baseline_w1"]
        biases = learn_router_biases(c1_stats, w1_stats)
        if "nudged_n0" not in payload:
            nudged = run_forward(
                model=model,
                tokenizer=tokenizer,
                prepared_prompts=prepared_n0,
                batch_size=args.batch_size,
                max_batch_tokens=args.max_batch_tokens,
                router_patterns=patterns,
                router_biases={k: v * args.router_bias_alpha for k, v in biases.items()},
            )
            payload["nudged_n0"] = nudged
            _save()
        else:
            nudged = payload["nudged_n0"]
        payload = {
            "mode": "router_bias",
            "baseline_n0": baseline,
            "baseline_c1": c1_stats,
            "baseline_w1": w1_stats,
            "nudged_n0": nudged,
            "diff_n0_vs_nudged": diff_stats(baseline, nudged),
        }

    args.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if checkpoint_path.exists():
        checkpoint_path.unlink(missing_ok=True)
    print(f"[done] wrote {args.output_path}")


if __name__ == "__main__":
    main()
