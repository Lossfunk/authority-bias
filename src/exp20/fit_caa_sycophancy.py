from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from tqdm import tqdm

from src.exp16.run_steering_test import _format_chat_prompt
from src.exp19.common import normalize, write_json
from src.exp20.common import MODEL_SWEEPS, csv, specs_and_sweeps, write_layerwise_payload
from src.mechanism.hooks import _extract_hidden, get_component_modules
from src.mechanism.hooks_v2 import setup_h100_optimizations, try_compile_model
from src.models.llama_loader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit CAA sycophancy vectors using Rimsky CAA contrast data.")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--caa-repo", type=Path, default=Path("/tmp/caa-readonly"))
    p.add_argument("--models", default="qwen35,gpt_oss,olmo2,olmo31")
    p.add_argument("--output-root", type=Path, default=Path("results/authority/exp20/caa_sycophancy"))
    p.add_argument("--max-pairs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16, help="Upper bound; final batches are token-dynamic.")
    p.add_argument("--max-batch-tokens", type=int, default=32768, help="Approx prompt token budget per batch.")
    p.add_argument("--save-every-batches", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--plain-prompt", action="store_true")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def load_caa_pairs(caa_repo: Path, max_pairs: int) -> list[dict[str, str]]:
    path = caa_repo / "datasets" / "generate" / "sycophancy" / "generate_dataset.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = []
    for row in data[: max_pairs if max_pairs > 0 else None]:
        pairs.append(
            {
                "question": row["question"],
                "pos": row["answer_matching_behavior"],
                "neg": row["answer_not_matching_behavior"],
            }
        )
    return pairs


def render(tokenizer, question: str, answer: str, plain_prompt: bool) -> str:
    user_text = question.strip()
    assistant_text = answer.strip()
    if plain_prompt:
        return f"{user_text}\n{assistant_text}"
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}, {"role": "assistant", "content": assistant_text}],
            tokenize=False,
            add_generation_prompt=False,
        )
    return _format_chat_prompt(tokenizer, f"{user_text}\n{assistant_text}", plain_prompt=False)


def collect_last_token_activations(model, tokenizer, prompts: Sequence[str], layers: Sequence[int]) -> dict[int, torch.Tensor]:
    modules = get_component_modules(model)
    captured: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            hidden = _extract_hidden(output)
            captured[layer].append(hidden[:, -1, :].detach().float().cpu())
            return output
        return hook

    for layer in layers:
        handles.append(modules[(layer, "block_output")].register_forward_hook(make_hook(layer)))
    try:
        inputs = tokenizer(list(prompts), padding=True, truncation=False, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {layer: torch.cat(chunks, dim=0) for layer, chunks in captured.items()}


def make_pair_batches(
    rows: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    max_batch_tokens: int,
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    batch: list[dict[str, Any]] = []
    max_len = 0
    for row in rows:
        n_tok = int(row["max_tokens"])
        if batch:
            cand_max = max(max_len, n_tok)
            if len(batch) + 1 > batch_size or cand_max * (len(batch) + 1) * 2 > max_batch_tokens:
                batches.append(batch)
                batch = []
                max_len = 0
        batch.append(row)
        max_len = max(max_len, n_tok)
    if batch:
        batches.append(batch)
    return batches


def fit_for_model(args: argparse.Namespace, model_name: str, model_id: str, layers: Sequence[int], pairs: list[dict[str, str]]) -> None:
    out_dir = args.output_root / model_name / "directions"
    direction_path = out_dir / "sycophancy.pt"
    checkpoint_path = out_dir / "sycophancy_fit_checkpoint.pt"
    if direction_path.exists() and not args.force:
        print(f"[skip] {model_name}: existing CAA direction at {direction_path}")
        return
    print(f"[fit_caa] loading {model_name}: {model_id} layers={csv(layers)} pairs={len(pairs)}")
    model, tokenizer = load_model_and_tokenizer(model_id, device=args.device)
    model.eval()
    if not args.no_compile:
        model = try_compile_model(model)

    rendered = []
    for row in pairs:
        pos_prompt = render(tokenizer, row["question"], row["pos"], args.plain_prompt)
        neg_prompt = render(tokenizer, row["question"], row["neg"], args.plain_prompt)
        rendered.append(
            {
                "pos_prompt": pos_prompt,
                "neg_prompt": neg_prompt,
                "max_tokens": max(
                    len(tokenizer(pos_prompt, add_special_tokens=False).input_ids),
                    len(tokenizer(neg_prompt, add_special_tokens=False).input_ids),
                ),
            }
        )

    pos_sum: dict[int, torch.Tensor]
    neg_sum: dict[int, torch.Tensor]
    count: int
    if checkpoint_path.exists() and not args.force:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        pos_sum = {int(k): v.float() for k, v in ckpt["pos_sum"].items()}
        neg_sum = {int(k): v.float() for k, v in ckpt["neg_sum"].items()}
        count = int(ckpt["count"])
        print(f"[resume] {model_name}: loaded checkpoint count={count} from {checkpoint_path}")
    else:
        pos_sum = {}
        neg_sum = {}
        count = 0

    batches = make_pair_batches(
        rendered[count:],
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
    )
    for batch_idx, batch in enumerate(tqdm(batches, desc=f"{model_name} CAA")):
        prompts = [row["pos_prompt"] for row in batch] + [row["neg_prompt"] for row in batch]
        acts = collect_last_token_activations(model, tokenizer, prompts, layers)
        n = len(batch)
        for layer in layers:
            pos = acts[layer][:n]
            neg = acts[layer][n:]
            pos_sum[layer] = pos_sum.get(layer, torch.zeros_like(pos[0])) + pos.sum(dim=0)
            neg_sum[layer] = neg_sum.get(layer, torch.zeros_like(neg[0])) + neg.sum(dim=0)
        count += n
        if args.save_every_batches > 0 and (batch_idx + 1) % args.save_every_batches == 0:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"count": count, "pos_sum": pos_sum, "neg_sum": neg_sum}, checkpoint_path)

    vectors = {layer: normalize((pos_sum[layer] / count) - (neg_sum[layer] / count)) for layer in layers}
    write_layerwise_payload(out_dir / "sycophancy.pt", vectors, name="caa_sycophancy_pos_minus_neg", position="answer_last_token")
    write_json(
        out_dir / "sycophancy_meta.json",
        {
            "source": "nrimsky/CAA datasets/generate/sycophancy/generate_dataset.json",
            "model": model_id,
            "layers": list(layers),
            "pairs": count,
            "batch_size": args.batch_size,
            "max_batch_tokens": args.max_batch_tokens,
            "construction": "mean(pos activation - neg activation) at final token of rendered answer",
            "raw_norms": {str(layer): float(((pos_sum[layer] / count) - (neg_sum[layer] / count)).norm()) for layer in layers},
        },
    )
    torch.save({"count": count, "pos_sum": pos_sum, "neg_sum": neg_sum}, checkpoint_path)
    del model
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    setup_h100_optimizations()
    pairs = load_caa_pairs(args.caa_repo, args.max_pairs)
    for model_name, spec, sweep in specs_and_sweeps(args.repo_root, args.models):
        fit_for_model(args, model_name, spec.model_id, sweep.layers, pairs)


if __name__ == "__main__":
    main()
