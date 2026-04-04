"""Extract activations for arbitrary Exp10 prompt conditions from saved results.

This is a lightweight extractor for known-direction comparison work:
given existing Exp10 result rows with fully rendered prompts, rerun a chosen
set of condition codes and save aligned hidden states at the same positions used
in the correction-gating pipeline.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from src.exp7.scoring import _tokenize_for_scoring
from src.mechanism.extract_correction_gating_activations import _resolve_positions
from src.models.llama_loader import load_model_and_tokenizer


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_RESULTS_PATH = Path(
    "new-phase-results/qwen3-4b-results/exp10_extended/"
    "Qwen__Qwen3-4B-Instruct-2507_results.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("new-phase-results/mechanism/condition_activation_extract")
POSITION_NAMES: Tuple[str, ...] = ("last_token", "instruction_last", "endorsement_last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract hidden states for arbitrary Exp10 prompt conditions."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--condition-codes",
        nargs="+",
        required=True,
        help="Prompt condition codes to rerun, e.g. N0_note W0_note W0_expert.",
    )
    parser.add_argument(
        "--uids-file",
        type=Path,
        default=None,
        help="Optional newline-delimited UID file to restrict the item set.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--loader-dtype", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _parse_condition_code(code: str) -> Tuple[str, bool, str]:
    try:
        prefix, tag_key = code.split("_", 1)
    except ValueError as exc:
        raise ValueError(f"Expected condition code like W0_note, got {code!r}") from exc
    if len(prefix) != 2:
        raise ValueError(f"Expected two-character prefix in {code!r}")
    endorse_map = {"N": "neutral", "W": "wrong", "C": "correct"}
    if prefix[0].upper() not in endorse_map or prefix[1] not in {"0", "1"}:
        raise ValueError(f"Invalid condition prefix in {code!r}")
    endorse_type = endorse_map[prefix[0].upper()]
    instruction = prefix[1] == "1"
    return endorse_type, instruction, tag_key


def _infer_instruction_text(prompt_text: str, instruction: bool) -> Optional[str]:
    if not instruction:
        return None
    first_line = prompt_text.splitlines()[0].strip()
    return first_line or None


def _infer_endorsement_line(prompt_text: str) -> str:
    lines = [line.rstrip() for line in prompt_text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Prompt is too short to infer endorsement line.")
    if lines[-1].strip() != "Answer:":
        raise ValueError("Prompt does not end with 'Answer:'.")
    return lines[-2]


def _collect_records(
    *,
    results_path: Path,
    condition_codes: Sequence[str],
    allowed_uids: Optional[set[str]],
    max_items: int,
) -> List[Dict]:
    rows: List[Dict] = []
    for row in _iter_jsonl(results_path):
        if allowed_uids is not None and row["uid"] not in allowed_uids:
            continue
        record = {
            "uid": row["uid"],
            "question": row["question"],
            "correct_label": row["correct_label"],
            "wrong_label": row["wrong_label"],
            "selection_metrics": row.get("selectivity_metrics", {}),
            "conditions": {},
            "metadata": row.get("metadata", {}),
        }
        for code in condition_codes:
            prompt_text = row["prompts"][code]
            endorse_type, instruction, tag_key = _parse_condition_code(code)
            record["conditions"][code] = {
                "prompt_text": prompt_text,
                "endorse_type": endorse_type,
                "instruction": instruction,
                "tag_key": tag_key,
                "tag": tag_key.replace("_", " ").title(),
                "instruction_text": _infer_instruction_text(prompt_text, instruction),
                "endorsement_line": _infer_endorsement_line(prompt_text),
            }
        rows.append(record)
        if max_items > 0 and len(rows) >= max_items:
            break
    return rows


def _resolve_save_dtype(raw: str) -> torch.dtype:
    if not hasattr(torch, raw):
        raise ValueError(f"Unknown torch dtype: {raw}")
    dtype = getattr(torch, raw)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{raw} is not a torch dtype")
    return dtype


def _positions_for_prompt(
    *,
    tokenizer,
    prompt_text: str,
    instruction_text: Optional[str],
    endorsement_line: str,
) -> Dict[str, Optional[int]]:
    if instruction_text:
        return _resolve_positions(
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            instruction_text=instruction_text,
            endorsement_line=endorsement_line,
        )

    tokenized = _tokenize_for_scoring(tokenizer, prompt_text)
    input_ids = tokenized["input_ids"][0].tolist()
    prompt_no_special = tokenizer(prompt_text, add_special_tokens=False).input_ids
    bos_offset = len(input_ids) - len(prompt_no_special)

    positions: Dict[str, Optional[int]] = {
        "last_token": len(input_ids) - 1,
        "instruction_last": None,
        "endorsement_last": None,
    }

    prefix = prompt_text.split(endorsement_line, 1)[0]
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    endorsement_ids = tokenizer(endorsement_line, add_special_tokens=False).input_ids
    if endorsement_ids:
        positions["endorsement_last"] = bos_offset + len(prefix_ids) + len(endorsement_ids) - 1
    return positions


def main() -> None:
    args = parse_args()
    allowed_uids = None
    if args.uids_file is not None:
        allowed_uids = {
            line.strip()
            for line in args.uids_file.read_text().splitlines()
            if line.strip()
        }
        print(f"Restricting extraction to {len(allowed_uids)} UIDs from {args.uids_file}")

    records = _collect_records(
        results_path=args.results_path,
        condition_codes=args.condition_codes,
        allowed_uids=allowed_uids,
        max_items=args.max_items,
    )
    if not records:
        raise SystemExit("No matching records found.")

    counts = Counter()
    for row in records:
        counts.update(row["conditions"].keys())
    print(f"Selected {len(records)} items from {args.results_path}")
    print(f"Condition counts: {dict(counts)}")

    summary = {
        "model": args.model,
        "results_path": str(args.results_path),
        "condition_codes": list(args.condition_codes),
        "device": args.device,
        "loader_dtype": args.loader_dtype,
        "n_items": len(records),
        "n_records": len(records) * len(args.condition_codes),
        "positions": list(POSITION_NAMES),
        "uids_file": str(args.uids_file) if args.uids_file else None,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        device=args.device,
        dtype=args.loader_dtype,
    )
    model.eval()

    save_dtype = _resolve_save_dtype(args.dtype)
    hidden_size = int(model.config.hidden_size)
    n_layers = int(model.config.num_hidden_layers) + 1  # hidden_states includes embeddings
    n_total = len(records) * len(args.condition_codes)

    activations = {
        pos: torch.empty((n_total, n_layers, hidden_size), dtype=save_dtype)
        for pos in POSITION_NAMES
    }
    metadata_rows: List[Dict] = []

    row_idx = 0
    for item in tqdm(records, desc=f"Extracting {args.model}"):
        for code in args.condition_codes:
            cond = item["conditions"][code]
            prompt_text = cond["prompt_text"]
            positions = _positions_for_prompt(
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                instruction_text=cond["instruction_text"],
                endorsement_line=cond["endorsement_line"],
            )
            model_inputs = {
                key: value.to(next(model.parameters()).device)
                for key, value in _tokenize_for_scoring(tokenizer, prompt_text).items()
            }
            with torch.inference_mode():
                outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
            hidden_states = [state[0].detach().to("cpu") for state in outputs.hidden_states]

            fallback_positions: Dict[str, str] = {}
            for pos_name in POSITION_NAMES:
                token_idx = positions[pos_name]
                if token_idx is None:
                    token_idx = positions["last_token"]
                    if token_idx is None:
                        raise ValueError(f"Position {pos_name} and last_token are undefined for {code}")
                    fallback_positions[pos_name] = "last_token"
                stacked = torch.stack([state[token_idx] for state in hidden_states], dim=0)
                activations[pos_name][row_idx] = stacked.to(save_dtype)

            metadata_rows.append(
                {
                    "row_index": row_idx,
                    "uid": item["uid"],
                    "question": item["question"],
                    "condition_code": code,
                    "endorse_type": cond["endorse_type"],
                    "instruction": cond["instruction"],
                    "tag": cond["tag"],
                    "tag_key": cond["tag_key"],
                    "instruction_text": cond["instruction_text"],
                    "endorsement_line": cond["endorsement_line"],
                    "positions": positions,
                    "fallback_positions": fallback_positions,
                    "correct_label": item["correct_label"],
                    "wrong_label": item["wrong_label"],
                    "selection_metrics": item["selection_metrics"],
                    "metadata": item["metadata"],
                }
            )
            row_idx += 1

    torch.save({"activations": activations}, args.output_dir / "activations.pt")
    with (args.output_dir / "metadata.jsonl").open("w") as f:
        for row in metadata_rows:
            f.write(json.dumps(row) + "\n")

    summary.update(
        {
            "metadata_path": str(args.output_dir / "metadata.jsonl"),
            "activations_path": str(args.output_dir / "activations.pt"),
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
