"""
Experiment 2: per-head path patching using cached neutral activations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import yaml
from tqdm import tqdm

from src.data.syc_dataset import (
    ensure_splits_exist,
    load_user_wrong_split,
    make_neutral_prompt,
)
from src.metrics.syc_metric import compute_D_syc
from src.models.llama_loader import load_model_and_tokenizer
from src.hooks.head_patch_hooks import ensure_head_patching, cache_mode, patch_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp2 path patching runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/exp2_path_patching.yaml"),
        help="YAML configuration path",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def iter_layers(model, cfg) -> Iterable[int]:
    total_layers = model.config.num_hidden_layers
    start = cfg.get("layer_start", 0)
    end = cfg.get("layer_end")
    if end is None:
        end = total_layers
    else:
        end = min(end, total_layers)
    return range(start, end)


def tokenize_text(tokenizer, text: str, device: torch.device):
    encoding = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in encoding.items()}


def run_path_patching(cfg: Dict) -> None:
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    dataset_cfg = cfg["dataset"]
    ensure_splits_exist(
        raw_path=Path(dataset_cfg["raw_path"]),
        split_a_path=Path(dataset_cfg["split_a_path"]),
        split_b_path=Path(dataset_cfg["split_b_path"]),
        split_c_path=Path(dataset_cfg["split_c_path"]),
        seed=seed,
    )
    examples = load_user_wrong_split(Path(dataset_cfg["split_b_path"]))
    max_examples = cfg["path_patching"].get("max_examples")
    if max_examples is not None:
        examples = examples[:max_examples]

    model_cfg = cfg["model"]
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_cfg["name"],
        device=model_cfg.get("device", "auto"),
        dtype=model_cfg.get("dtype", "auto"),
    )
    device = next(model.parameters()).device
    registry = ensure_head_patching(model)

    num_heads = model.config.num_attention_heads
    layer_range = list(iter_layers(model, cfg["path_patching"]))

    output_cfg = cfg["output"]
    raw_path = Path(output_cfg["raw_results"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    with raw_path.open("w") as writer:
        for ex in tqdm(examples, desc="Exp2 path patching"):
            prompt_wrong = ex.user_wrong_prompt
            prompt_neutral = make_neutral_prompt(ex)
            inputs_wrong = tokenize_text(tokenizer, prompt_wrong, device)
            pos_wrong = inputs_wrong["input_ids"].shape[1] - 1

            baseline = compute_D_syc(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_wrong,
                wrong_answer=ex.wrong_answer,
                right_answer=ex.correct_answer,
                device=device,
            )

            # Cache head contexts on neutral prompt
            inputs_neutral = tokenize_text(tokenizer, prompt_neutral, device)
            pos_neutral = inputs_neutral["input_ids"].shape[1] - 1
            with cache_mode(registry, ex.uid, pos_neutral):
                with torch.no_grad():
                    model(**inputs_neutral)

            for layer_idx in layer_range:
                for head_idx in range(num_heads):
                    with patch_mode(registry, ex.uid, layer_idx, head_idx, pos_wrong):
                        patched = compute_D_syc(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_text=prompt_wrong,
                            wrong_answer=ex.wrong_answer,
                            right_answer=ex.correct_answer,
                            device=device,
                        )
                    record = {
                        "example_id": ex.uid,
                            "layer": layer_idx,
                            "head": head_idx,
                            "pos_idx_wrong": pos_wrong,
                            "pos_idx_neutral": pos_neutral,
                            "D_syc_orig": baseline["D_syc"],
                            "D_syc_patched": patched["D_syc"],
                        "delta": baseline["D_syc"] - patched["D_syc"],
                        "logP_wrong_orig": baseline["logP_wrong"],
                        "logP_right_orig": baseline["logP_right"],
                        "logP_wrong_patched": patched["logP_wrong"],
                        "logP_right_patched": patched["logP_right"],
                        "num_heads": num_heads,
                        "num_layers": model.config.num_hidden_layers,
                    }
                    writer.write(json.dumps(record) + "\n")
            registry.cache.clear_example(ex.uid)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_path_patching(cfg)


if __name__ == "__main__":
    main()

