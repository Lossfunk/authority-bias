"""
Minimal diagnostic to verify whether head patching changes the final hidden
state and logits for the first answer token on a single example/head.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.data.syc_dataset import ensure_splits_exist, load_user_wrong_split, make_neutral_prompt
from src.hooks.head_patch_hooks import ensure_head_patching, cache_mode, patch_mode
from src.metrics.syc_metric import compute_D_syc
from src.models.llama_loader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-example head patch diagnostic")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/exp2_path_patching.yaml"),
        help="YAML config (reuses Exp2 paths/model defaults)",
    )
    parser.add_argument("--example-idx", type=int, default=0, help="Index into split_B examples")
    parser.add_argument("--layer", type=int, default=0, help="Layer to patch")
    parser.add_argument("--head", type=int, default=0, help="Head to patch")
    parser.add_argument(
        "--pos-mode",
        choices=["prompt_last", "first_answer"],
        default="prompt_last",
        help="Token position to patch: last prompt token or first answer token.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path (default is updated-results/exp2/diagnostics/...)",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def tokenize_prompt(tokenizer, text: str, device: torch.device, add_special_tokens: bool = True):
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
    return {k: v.to(device) for k, v in enc.items()}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 42))

    dataset_cfg = cfg["dataset"]
    ensure_splits_exist(
        raw_path=Path(dataset_cfg["raw_path"]),
        split_a_path=Path(dataset_cfg["split_a_path"]),
        split_b_path=Path(dataset_cfg["split_b_path"]),
        split_c_path=Path(dataset_cfg["split_c_path"]),
        seed=cfg.get("seed", 42),
    )
    examples = load_user_wrong_split(Path(dataset_cfg["split_b_path"]))
    if args.example_idx >= len(examples):
        raise IndexError(f"example-idx {args.example_idx} >= {len(examples)} available examples")
    ex = examples[args.example_idx]

    model_cfg = cfg["model"]
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_cfg["name"],
        device=model_cfg.get("device", "auto"),
        dtype=model_cfg.get("dtype", "auto"),
    )
    device = next(model.parameters()).device
    registry = ensure_head_patching(model, force=True)

    # Prompts and positions
    prompt_wrong = ex.user_wrong_prompt
    prompt_neutral = make_neutral_prompt(ex)

    prompt_inputs = tokenize_prompt(tokenizer, prompt_wrong, device, add_special_tokens=True)
    prompt_len = prompt_inputs["input_ids"].shape[1]
    # Position to patch on the wrong run
    if args.pos_mode == "prompt_last":
        pos_wrong = prompt_len - 1
    else:  # first_answer
        pos_wrong = prompt_len  # index of first answer token after concatenation

    answer_ids = tokenizer(" " + ex.wrong_answer, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    input_ids = torch.cat([prompt_inputs["input_ids"], answer_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        baseline = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    base_hidden = baseline.hidden_states[-1][:, pos_wrong, :].detach()
    base_logits = baseline.logits[:, pos_wrong, :].detach()
    first_token_id = answer_ids[0, 0].item()
    base_logprob_first = torch.log_softmax(base_logits, dim=-1)[0, first_token_id].item()

    # Cache neutral head output at its decision point.
    neutral_inputs = tokenize_prompt(tokenizer, prompt_neutral, device, add_special_tokens=True)
    pos_neutral = neutral_inputs["input_ids"].shape[1] - 1
    with cache_mode(registry, ex.uid, pos_neutral):
        with torch.no_grad():
            model(**neutral_inputs)

    # Patched forward on the wrong prompt + wrong answer.
    with patch_mode(registry, ex.uid, args.layer, args.head, pos_wrong):
        with torch.no_grad():
            patched = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
    patched_hidden = patched.hidden_states[-1][:, pos_wrong, :].detach()
    patched_logits = patched.logits[:, pos_wrong, :].detach()
    patched_logprob_first = torch.log_softmax(patched_logits, dim=-1)[0, first_token_id].item()

    hidden_diff_norm = (patched_hidden - base_hidden).norm().item()
    logits_diff = patched_logits - base_logits
    logits_diff_norm = logits_diff.norm().item()
    logits_diff_max = logits_diff.abs().max().item()

    # Compute D_syc effects for the example to align with core metric.
    base_metric = compute_D_syc(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_wrong,
        wrong_answer=ex.wrong_answer,
        right_answer=ex.correct_answer,
        device=device,
    )
    with patch_mode(registry, ex.uid, args.layer, args.head, pos_wrong):
        patched_metric = compute_D_syc(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_wrong,
            wrong_answer=ex.wrong_answer,
            right_answer=ex.correct_answer,
            device=device,
        )

    result = {
        "example_id": ex.uid,
        "example_idx": args.example_idx,
        "layer": args.layer,
        "head": args.head,
        "prompt_len": prompt_len,
        "pos_wrong": pos_wrong,
        "pos_neutral": pos_neutral,
        "first_wrong_token_id": first_token_id,
        "hidden_norm_base": base_hidden.norm().item(),
        "hidden_norm_patched": patched_hidden.norm().item(),
        "hidden_diff_norm": hidden_diff_norm,
        "logits_norm_base": base_logits.norm().item(),
        "logits_norm_patched": patched_logits.norm().item(),
        "logits_diff_norm": logits_diff_norm,
        "logits_diff_max_abs": logits_diff_max,
        "logprob_first_base": base_logprob_first,
        "logprob_first_patched": patched_logprob_first,
        "logprob_first_delta": patched_logprob_first - base_logprob_first,
        "D_syc_base": base_metric["D_syc"],
        "D_syc_patched": patched_metric["D_syc"],
        "D_syc_delta": patched_metric["D_syc"] - base_metric["D_syc"],
    }

    # Output
    if args.output is None:
        out_dir = Path("updated-results/exp2/diagnostics")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"patch_effect_ex{args.example_idx}_L{args.layer}H{args.head}.json"
    else:
        out_path = args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    registry.cache.clear_example(ex.uid)


if __name__ == "__main__":
    main()
