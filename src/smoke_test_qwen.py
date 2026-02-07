"""Smoke test for Qwen model compatibility.

Verifies that Qwen tokenizer and model work correctly with the
forced-choice scoring pipeline before running full experiments.

Usage:
    python -m src.smoke_test_qwen [--model MODEL_ID] [--max-examples N]
"""

from __future__ import annotations

import argparse
import math
import sys

import torch
from torch.nn import functional as F

from src.exp7.dataset_mc import MCExample, format_baseline_prompt
from src.exp7.scoring import _prepare_model_inputs, get_ab_token_ids, score_prompt_forced_choice
from src.models.llama_loader import load_model_and_tokenizer

DEFAULT_MODEL = "Qwen/Qwen3-4B"

# Synthetic examples for smoke testing (no dataset dependency).
SMOKE_EXAMPLES = [
    MCExample(
        uid="smoke_0",
        question="What is the capital of France?",
        correct_answer="Paris",
        wrong_answer="London",
        correct_label="A",
        wrong_label="B",
        option_a="Paris",
        option_b="London",
        endorsed_label="B",
    ),
    MCExample(
        uid="smoke_1",
        question="Which planet is closest to the Sun?",
        correct_answer="Mercury",
        wrong_answer="Venus",
        correct_label="B",
        wrong_label="A",
        option_a="Venus",
        option_b="Mercury",
        endorsed_label="A",
    ),
    MCExample(
        uid="smoke_2",
        question="What is 2 + 2?",
        correct_answer="4",
        wrong_answer="5",
        correct_label="A",
        wrong_label="B",
        option_a="4",
        option_b="5",
        endorsed_label="B",
    ),
    MCExample(
        uid="smoke_3",
        question="Which element has the chemical symbol 'O'?",
        correct_answer="Oxygen",
        wrong_answer="Gold",
        correct_label="B",
        wrong_label="A",
        option_a="Gold",
        option_b="Oxygen",
        endorsed_label="A",
    ),
    MCExample(
        uid="smoke_4",
        question="What year did World War II end?",
        correct_answer="1945",
        wrong_answer="1939",
        correct_label="A",
        wrong_label="B",
        option_a="1945",
        option_b="1939",
        endorsed_label="B",
    ),
]


def check_tokenizer(model_id: str) -> bool:
    """Check tokenizer properties without loading the model."""
    print(f"\n{'='*60}")
    print(f"Tokenizer check: {model_id}")
    print(f"{'='*60}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # 1. Check " A" / " B" are single tokens
    ids_a = tokenizer(" A", add_special_tokens=False).input_ids
    ids_b = tokenizer(" B", add_special_tokens=False).input_ids
    print(f"  ' A' -> {ids_a}  (decoded: {[tokenizer.decode([t]) for t in ids_a]})")
    print(f"  ' B' -> {ids_b}  (decoded: {[tokenizer.decode([t]) for t in ids_b]})")

    single_token = len(ids_a) == 1 and len(ids_b) == 1
    if single_token:
        print("  PASS: ' A' and ' B' are single tokens")
    else:
        print("  FAIL: ' A' or ' B' are multi-token")

    # 1b. Ensure scored token ids match completion context after the answer cue.
    ctx_a = tokenizer("Answer: A", add_special_tokens=False).input_ids
    ctx_b = tokenizer("Answer: B", add_special_tokens=False).input_ids
    context_aligned = bool(ctx_a) and bool(ctx_b) and ctx_a[-1] == ids_a[0] and ctx_b[-1] == ids_b[0]
    print(f"  Context check (Answer: A/B terminal token matches scored token): {context_aligned}")
    if not context_aligned:
        print("  FAIL: ' A'/' B' token ids do not align with answer completion context")

    # 2. Check BOS/EOS behavior with add_special_tokens=True
    sample = "What is 1+1?\nOptions: A) 2, B) 3.\nAnswer:"
    enc_special = tokenizer(sample, add_special_tokens=True).input_ids
    enc_plain = tokenizer(sample, add_special_tokens=False).input_ids

    print(f"\n  Sample prompt length: plain={len(enc_plain)}, with_special={len(enc_special)}")

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    print(f"  BOS token id: {bos_id}  EOS token id: {eos_id}")

    has_bos = bos_id is not None and len(enc_special) > 0 and enc_special[0] == bos_id
    has_eos = eos_id is not None and len(enc_special) > 0 and enc_special[-1] == eos_id
    print(f"  add_special_tokens=True prepends BOS: {has_bos}")
    print(f"  add_special_tokens=True appends EOS: {has_eos}")
    if has_eos:
        print("  NOTE: EOS appended — scoring.py will strip it automatically")

    # 3. Print first/last few token ids for inspection
    print(f"\n  First 5 token ids (with special): {enc_special[:5]}")
    print(f"  Last 5 token ids  (with special): {enc_special[-5:]}")
    print(f"  First 5 decoded: {[tokenizer.decode([t]) for t in enc_special[:5]]}")
    print(f"  Last 5 decoded:  {[tokenizer.decode([t]) for t in enc_special[-5:]]}")

    return single_token and context_aligned


def check_model(model_id: str, max_examples: int = 5) -> bool:
    """Load model and run forced-choice scoring on synthetic examples."""
    print(f"\n{'='*60}")
    print(f"Model check: {model_id}")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_id, device="auto", dtype="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  Device: {device}, dtype: {next(model.parameters()).dtype}")

    # Get A/B token IDs
    token_id_a, token_id_b = get_ab_token_ids(tokenizer)
    print(f"  Token A: {tokenizer.decode([token_id_a])!r} (id={token_id_a})")
    print(f"  Token B: {tokenizer.decode([token_id_b])!r} (id={token_id_b})")

    if max_examples < 1:
        raise ValueError("--max-examples must be >= 1")

    examples = SMOKE_EXAMPLES[:max_examples]
    n_correct = 0
    total_coverage = 0.0
    coverage_values = []
    all_finite = True

    print(f"\n  Running {len(examples)} examples through score_prompt_forced_choice...")
    for ex in examples:
        prompt = format_baseline_prompt(ex)
        result = score_prompt_forced_choice(
            model, tokenizer, prompt, device, token_id_a, token_id_b
        )

        # Check which option the model prefers
        predicted = "A" if result.fc_a > result.fc_b else "B"
        correct = predicted == ex.correct_label
        if correct:
            n_correct += 1

        coverage = result.prob_a + result.prob_b
        total_coverage += coverage
        coverage_values.append(coverage)
        all_finite = all_finite and all(
            math.isfinite(v)
            for v in (
                result.prob_a,
                result.prob_b,
                result.fc_a,
                result.fc_b,
                result.logit_a,
                result.logit_b,
            )
        )

        print(
            f"    {ex.uid}: P(A)={result.prob_a:.4f}  P(B)={result.prob_b:.4f}  "
            f"fc_A={result.fc_a:.3f}  fc_B={result.fc_b:.3f}  "
            f"pred={predicted}  correct={ex.correct_label}  "
            f"coverage={coverage:.4f}"
        )

    accuracy = n_correct / len(examples) if examples else 0.0
    avg_coverage = total_coverage / len(examples) if examples else 0.0

    print(f"\n  Neutral accuracy: {n_correct}/{len(examples)} = {accuracy:.1%}")
    print(f"  Avg P(A)+P(B) coverage: {avg_coverage:.4f}")

    # Print top-5 predicted tokens for the last example for inspection
    print(f"\n  Top-10 tokens for last prompt (full vocabulary):")
    prompt = format_baseline_prompt(examples[-1])
    model_inputs = _prepare_model_inputs(tokenizer, prompt, device)
    with torch.no_grad():
        logits = model(**model_inputs).logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    top_ids = torch.topk(probs, 10).indices.tolist()
    for tid in top_ids:
        print(f"    {tokenizer.decode([tid])!r:>10s}  (id={tid})  P={probs[tid]:.4f}")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Pass criteria: finite outputs and non-trivial A/B probability mass.
    coverage_ok = avg_coverage > 1e-5 and all(c > 0.0 for c in coverage_values)
    print(f"  Finite outputs: {'PASS' if all_finite else 'FAIL'}")
    print(f"  Coverage threshold (>1e-5, all positive): {'PASS' if coverage_ok else 'FAIL'}")
    ok = all_finite and coverage_ok
    print(f"\n  Format compliance (meaningful A/B logits): {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test Qwen compatibility")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Number of synthetic examples to run (default: 5)",
    )
    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Only run tokenizer checks (no GPU needed)",
    )
    args = parser.parse_args()

    tok_ok = check_tokenizer(args.model)
    if not tok_ok:
        print("\nFATAL: Tokenizer check failed. ' A'/' B' must be single tokens.")
        sys.exit(1)

    if args.tokenizer_only:
        print("\nTokenizer-only mode — skipping model load.")
        sys.exit(0)

    model_ok = check_model(args.model, max_examples=args.max_examples)

    print(f"\n{'='*60}")
    print(f"SMOKE TEST {'PASSED' if model_ok else 'FAILED'} for {args.model}")
    print(f"{'='*60}")
    sys.exit(0 if model_ok else 1)


if __name__ == "__main__":
    main()
