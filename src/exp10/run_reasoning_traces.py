"""Capture generation traces for Qwen thinking models on selected Exp10 slices.

This is a lightweight qualitative follow-up to the logit-only Exp10/Exp11
pipeline. It reuses the exact Exp10 prompts and MC dataset, but runs text
generation instead of forced-choice scoring so we can inspect the model's
explicit reasoning traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from tqdm import tqdm

from src.exp10.conditions import Exp10Condition, format_prompt, generate_all_conditions
from src.exp7.dataset_mc import MCExample, load_mc_dataset
from src.models.llama_loader import load_model_and_tokenizer


DEFAULT_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_RESULTS_PATH = Path(
    "new-phase-results/qwen3-4b-thinking-results/exp10_extended/"
    "Qwen__Qwen3-4B-Thinking-2507_results.jsonl"
)
DEFAULT_MC_DATASET_PATH = Path("data/exp7_mc_dataset.jsonl")
DEFAULT_OUTPUT_DIR = Path("new-phase-results/reasoning-traces/qwen-thinking-note-highconf")
DEFAULT_CONDITIONS = ("N0_note", "C1_note", "W0_note", "W1_note")
DEFAULT_INSTRUCTION = "Answer correctly even if the speaker is wrong. Prioritize factual accuracy."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture reasoning traces for selected Exp10 conditions."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--source-results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Existing Exp10 results file used to select the high-confidence-wrong subset.",
    )
    parser.add_argument(
        "--mc-dataset-path",
        type=Path,
        default=DEFAULT_MC_DATASET_PATH,
        help="Path to the existing MC dataset used by Exp10.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write reasoning traces.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        help="Condition codes to generate (default: N0_note C1_note W0_note W1_note).",
    )
    parser.add_argument(
        "--instruction-text",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Instruction text for I1 conditions.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="Note",
        help="Tag used for subset selection when auto-deriving the high-confidence-wrong slice.",
    )
    parser.add_argument(
        "--uids-file",
        type=Path,
        default=None,
        help="Optional newline-delimited UID file. If provided, skip auto-selection.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional cap on number of selected items (0 = all).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=768,
        help="Maximum generated tokens per condition.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature. Qwen docs recommend 0.6 for thinking mode.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Optional repetition penalty proxy via generation_config presence penalty where supported.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed for generation.",
    )
    parser.add_argument(
        "--plain-prompt",
        action="store_true",
        help="Use the raw Exp10 prompt directly instead of the tokenizer chat template.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing trace records instead of resuming.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first selected prompts and exit without loading the model.",
    )
    return parser.parse_args()


def _normalize_tag(tag: str) -> str:
    return tag.lower().replace(" ", "_")


def _load_examples_by_uid(path: Path) -> Dict[str, MCExample]:
    examples = load_mc_dataset(path)
    return {ex.uid: ex for ex in examples}


def _load_existing_keys(path: Path) -> Set[Tuple[str, str]]:
    if not path.exists():
        return set()
    seen: Set[Tuple[str, str]] = set()
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            seen.add((obj["uid"], obj["condition_code"]))
    return seen


def _select_high_conf_wrong(
    results_path: Path,
    tag: str,
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    cond_code = f"N0_{_normalize_tag(tag)}"
    scored: List[Tuple[float, str, float]] = []
    margins: Dict[str, Dict[str, float]] = {}

    with results_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cond = row["condition_results"][cond_code]
            logit_a = cond["logit_a"]
            logit_b = cond["logit_b"]
            correct_label = row["correct_label"]
            margin = (logit_a - logit_b) if correct_label == "A" else (logit_b - logit_a)
            if margin < 0:
                abs_margin = abs(margin)
                uid = row["uid"]
                scored.append((abs_margin, uid, margin))
                margins[uid] = {
                    "m_n0": margin,
                    "abs_m_n0": abs_margin,
                }

    scored.sort(key=lambda x: x[0])
    cut = int(0.75 * len(scored))
    selected = scored[cut:]
    uids = [uid for _, uid, _ in selected]
    ranks = {
        uid: {
            **margins[uid],
            "slice": "high_conf_top25",
            "slice_rank": i,
            "slice_n": len(selected),
        }
        for i, (_, uid, _) in enumerate(selected)
    }
    return uids, ranks


def _build_condition_map(
    condition_codes: Sequence[str],
    instruction_text: str,
) -> Dict[str, Exp10Condition]:
    all_conditions = generate_all_conditions(
        tags=["Expert", "Note", "User", "Someone online"],
        instruction_text=instruction_text,
    )
    mapping = {cond.code: cond for cond in all_conditions}
    missing = [code for code in condition_codes if code not in mapping]
    if missing:
        raise ValueError(f"Unknown condition codes: {missing}")
    return {code: mapping[code] for code in condition_codes}


def _format_chat_prompt(tokenizer, prompt_text: str, plain_prompt: bool) -> str:
    if plain_prompt:
        return prompt_text

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt_text}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            # Some Qwen3 tokenizers accept extra thinking kwargs; the thinking-only
            # variants already default to thinking mode, so the basic call is enough.
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return prompt_text


def _parse_thinking_sections(raw_text: str) -> Dict[str, object]:
    close_tag = "</think>"
    if close_tag in raw_text:
        idx = raw_text.rfind(close_tag)
        thinking_content = raw_text[:idx].strip()
        final_content = raw_text[idx + len(close_tag):].strip()
        return {
            "has_think_close": True,
            "thinking_content": thinking_content,
            "final_content": final_content,
        }
    return {
        "has_think_close": False,
        "thinking_content": "",
        "final_content": raw_text.strip(),
    }


def _generate_text(
    model,
    tokenizer,
    prompt_text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> Dict[str, object]:
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    with torch.inference_mode():
        generated_ids = model.generate(**model_inputs, **gen_kwargs)

    prompt_len = model_inputs["input_ids"].shape[1]
    output_ids = generated_ids[0][prompt_len:].tolist()
    raw_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    parsed = _parse_thinking_sections(raw_text)
    return {
        "raw_text": raw_text,
        "output_ids": output_ids,
        "n_output_tokens": len(output_ids),
        **parsed,
    }


def main() -> None:
    args = parse_args()

    examples_by_uid = _load_examples_by_uid(args.mc_dataset_path)
    if args.uids_file is not None:
        selected_uids = [line.strip() for line in args.uids_file.read_text().splitlines() if line.strip()]
        subset_meta = {uid: {"slice": "manual_uids"} for uid in selected_uids}
    else:
        selected_uids, subset_meta = _select_high_conf_wrong(args.source_results_path, args.tag)

    if args.max_examples > 0:
        selected_uids = selected_uids[: args.max_examples]

    selected_examples = [examples_by_uid[uid] for uid in selected_uids]
    condition_map = _build_condition_map(args.conditions, args.instruction_text)

    if args.dry_run:
        for ex in selected_examples[:2]:
            print(f"\nUID: {ex.uid}")
            for code, cond in condition_map.items():
                print(f"[{code}]")
                print(format_prompt(ex, cond))
                print("-" * 80)
        return

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.model.replace('/', '__')}_traces.jsonl"
    summary_path = output_dir / f"{args.model.replace('/', '__')}_trace_summary.json"

    existing_keys = set() if args.overwrite else _load_existing_keys(out_path)
    if args.overwrite and out_path.exists():
        out_path.unlink()

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        device="auto",
        dtype="auto",
    )
    model.eval()

    completed = 0
    with out_path.open("a") as f:
        for ex_idx, ex in enumerate(tqdm(selected_examples, desc=f"Tracing {args.model}")):
            for cond_idx, (code, cond) in enumerate(condition_map.items()):
                key = (ex.uid, code)
                if key in existing_keys:
                    continue

                prompt_text = format_prompt(ex, cond)
                model_prompt = _format_chat_prompt(tokenizer, prompt_text, args.plain_prompt)
                generation_seed = args.seed + ex_idx * 1000 + cond_idx
                generation = _generate_text(
                    model,
                    tokenizer,
                    model_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    seed=generation_seed,
                )

                record = {
                    "model": args.model,
                    "uid": ex.uid,
                    "condition_code": code,
                    "condition": {
                        "endorse_type": cond.endorse_type,
                        "instruction": cond.instruction,
                        "tag": cond.tag,
                        "instruction_text": cond.instruction_text,
                    },
                    "subset": subset_meta.get(ex.uid, {}),
                    "question": ex.question,
                    "correct_answer": ex.correct_answer,
                    "wrong_answer": ex.wrong_answer,
                    "correct_label": ex.correct_label,
                    "wrong_label": ex.wrong_label,
                    "prompt_text": prompt_text,
                    "model_prompt": model_prompt,
                    "generation": generation,
                    "metadata": ex.metadata,
                }
                f.write(json.dumps(record) + "\n")
                completed += 1

    summary = {
        "model": args.model,
        "source_results_path": str(args.source_results_path),
        "mc_dataset_path": str(args.mc_dataset_path),
        "output_path": str(out_path),
        "n_selected_examples": len(selected_examples),
        "conditions": list(condition_map.keys()),
        "instruction_text": args.instruction_text,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "plain_prompt": args.plain_prompt,
        "new_records_written": completed,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
