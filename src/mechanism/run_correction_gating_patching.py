"""Run the first causal patching experiment for the correction-gating pilot.

This script is intentionally standalone and aligned to the new extraction
pipeline in `extract_correction_gating_activations.py`. It does not rely on the
older intervention stack.

Default experiment:
1. Load the aligned Qwen factual-QA Note extraction.
2. Build the within-i1a steering vector:
      mean(correcting_i1a) - mean(resisting_i1a)
   at a chosen position/layer in the *saved hidden-state indexing*.
3. Re-run the resisting i1a prompts only, patching the corresponding
   transformer block output at the chosen prompt position.
4. Sweep alpha and compare:
   - real vector
   - wrong-sign vector
   - random matched-norm vector
   - wrong-site patch (same vector, instruction position)
5. Report per-item answer flips and signed logit-margin shifts toward the
   correct answer.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from src.exp10.conditions import Exp10Condition, format_prompt
from src.exp7.dataset_mc import MCExample, load_mc_dataset
from src.exp7.scoring import _tokenize_for_scoring, get_ab_token_ids
from src.mechanism.extract_correction_gating_activations import _resolve_positions
from src.models.llama_loader import load_model_and_tokenizer


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_EXTRACTION_DIR = Path(
    "new-phase-results/mechanism/qwen_correction_gating_i1a_i1c_note"
)
DEFAULT_MC_DATASET_PATH = Path("data/exp7_mc_dataset.jsonl")
DEFAULT_OUTPUT_DIR = Path(
    "new-phase-results/mechanism/qwen_correction_gating_patching_note"
)
DEFAULT_POSITION = "endorsement_last"
DEFAULT_LAYER_INDEX = 23  # hidden_states indexing from extraction, includes embeddings at 0
DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
I1A_LABEL = "i1a"
I1A_TEXT = "Answer correctly even if the speaker is wrong. Prioritize factual accuracy."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the first causal patching experiment for correction gating."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--extraction-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--mc-dataset-path", type=Path, default=DEFAULT_MC_DATASET_PATH)
    parser.add_argument(
        "--steering-vector-path",
        type=Path,
        default=None,
        help=(
            "Optional path to a precomputed steering-vector payload. When set, "
            "the script uses this vector instead of loading activations.pt."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--position",
        type=str,
        default=DEFAULT_POSITION,
        choices=("last_token", "instruction_last", "endorsement_last"),
        help="Primary patch position for the real, wrong-sign, and random controls.",
    )
    parser.add_argument(
        "--wrong-site-position",
        type=str,
        default="instruction_last",
        choices=("none", "last_token", "instruction_last", "endorsement_last"),
        help="Position used for the wrong-site control. Use 'none' to disable it.",
    )
    parser.add_argument(
        "--target-subset",
        type=str,
        default="resisting",
        choices=("entrenching", "resisting", "correcting", "expanded_resisting"),
        help="Which i1a-labeled subset to evaluate (entrenching and resisting are equivalent negative classes).",
    )
    parser.add_argument(
        "--prompt-condition",
        type=str,
        default="C1_note",
        help="Prompt condition to evaluate, e.g. C1_note or N0_note.",
    )
    parser.add_argument(
        "--instruction-text",
        type=str,
        default=I1A_TEXT,
        help="Instruction text used when the prompt condition has instruction=1.",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=DEFAULT_LAYER_INDEX,
        help=(
            "Saved hidden-state layer index from extraction. 0 is embeddings; "
            "transformer block index is layer_index - 1."
        ),
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
        help="Alpha sweep for the intervention.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Optional cap on target-subset items for smoke tests (0 = all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the matched-norm random control.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model dtype passed to the loader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device passed to the model loader.",
    )
    parser.add_argument(
        "--loader-dtype",
        type=str,
        default=None,
        help="Optional alias for --dtype to match other mechanism scripts.",
    )
    return parser.parse_args()


def _load_metadata(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_extraction(extraction_dir: Path) -> Tuple[List[Dict], Dict[str, torch.Tensor]]:
    metadata = _load_metadata(extraction_dir / "metadata.jsonl")
    payload = torch.load(extraction_dir / "activations.pt", map_location="cpu")
    return metadata, payload["activations"]


def _load_vector_payload(path: Path) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    if "vector" not in payload:
        raise KeyError(f"Expected 'vector' key in steering payload: {path}")
    return payload


def _parse_prompt_condition(code: str, instruction_text: str) -> Exp10Condition:
    try:
        ei, raw_tag = code.split("_", 1)
    except ValueError as exc:
        raise ValueError(f"Expected prompt-condition like C1_note, got: {code}") from exc
    if len(ei) != 2:
        raise ValueError(f"Expected two-character prefix like C1 in {code}")
    endorse_map = {"N": "neutral", "W": "wrong", "C": "correct"}
    instruction_map = {"0": False, "1": True}
    endorse_type = endorse_map[ei[0].upper()]
    instruction = instruction_map[ei[1]]
    tag = raw_tag.replace("_", " ").title()
    return Exp10Condition(
        endorse_type=endorse_type,  # type: ignore[arg-type]
        instruction=instruction,
        tag=tag,  # type: ignore[arg-type]
        instruction_text=instruction_text if instruction else None,
    )


def _endorsement_line(example: MCExample, condition: Exp10Condition) -> Optional[str]:
    if condition.endorse_type == "neutral":
        return f"{condition.tag}: I'm considering options A and B."
    if condition.endorse_type == "wrong":
        return f"{condition.tag}: I think it's {example.wrong_label}, not {example.correct_label}."
    if condition.endorse_type == "correct":
        return f"{condition.tag}: I think it's {example.correct_label}, not {example.wrong_label}."
    return None


def _row_matches_subset(row: Dict, target_subset: str) -> bool:
    if target_subset in NEGATIVE_LABELS:
        return row["item_label"] in NEGATIVE_LABELS
    if target_subset == "expanded_resisting":
        return bool(row.get("selection", {}).get("expanded_resisting", False))
    return row["item_label"] == target_subset


def _select_target_rows(metadata: Sequence[Dict], target_subset: str) -> List[int]:
    rows: List[int] = []
    for idx, row in enumerate(metadata):
        if row["variant_label"] != I1A_LABEL:
            continue
        if _row_matches_subset(row, target_subset):
            rows.append(idx)
    return rows


def _build_eval_prompt_record(
    tokenizer,
    row: Dict,
    example: MCExample,
    condition: Exp10Condition,
) -> Dict[str, object]:
    prompt_text = format_prompt(example, condition)
    end_line = _endorsement_line(example, condition)
    if condition.instruction_text:
        positions = _resolve_positions(
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            instruction_text=condition.instruction_text,
            endorsement_line=end_line or "",
        )
    else:
        tokenized = _tokenize_for_scoring(tokenizer, prompt_text)
        input_ids = tokenized["input_ids"][0].tolist()
        positions = {
            "last_token": len(input_ids) - 1,
            "instruction_last": None,
            "endorsement_last": None,
        }
        if end_line:
            prefix = prompt_text.split(end_line, 1)[0]
            bos_offset = len(input_ids) - len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
            prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
            end_ids = tokenizer(end_line, add_special_tokens=False).input_ids
            positions["endorsement_last"] = bos_offset + len(prefix_ids) + len(end_ids) - 1
    return {
        "uid": row["uid"],
        "question": row["question"],
        "item_label": row["item_label"],
        "correct_label": row["correct_label"],
        "wrong_label": row["wrong_label"],
        "selection": row["selection"],
        "prompt_text": prompt_text,
        "positions": positions,
    }


NEGATIVE_LABELS = frozenset({"entrenching", "resisting"})


def _split_i1a_rows(metadata: Sequence[Dict]) -> Tuple[List[int], List[int], List[int]]:
    i1a_all: List[int] = []
    i1a_correcting: List[int] = []
    i1a_negative: List[int] = []
    for idx, row in enumerate(metadata):
        if row["variant_label"] != I1A_LABEL:
            continue
        i1a_all.append(idx)
        if row["item_label"] == "correcting":
            i1a_correcting.append(idx)
        elif _row_matches_subset(row, "expanded_resisting"):
            i1a_negative.append(idx)
        else:
            raise ValueError(f"Unexpected item_label: {row['item_label']}")
    return i1a_all, i1a_correcting, i1a_negative


def _build_within_i1a_vector(
    activations: Dict[str, torch.Tensor],
    correcting_rows: Sequence[int],
    negative_rows: Sequence[int],
    position: str,
    layer_index: int,
) -> torch.Tensor:
    pos_acts = activations[position]
    correcting_mean = pos_acts[torch.tensor(correcting_rows), layer_index, :].float().mean(dim=0)
    negative_mean = pos_acts[torch.tensor(negative_rows), layer_index, :].float().mean(dim=0)
    return correcting_mean - negative_mean


def _matched_random_vector(vector: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    rand = torch.randn(vector.shape, generator=g, dtype=torch.float32)
    rand = rand / rand.norm().clamp_min(1e-8)
    return rand * vector.norm().clamp_min(1e-8)


def _signed_margin_from_logits(logits: torch.Tensor, token_id_a: int, token_id_b: int, correct_label: str) -> float:
    logit_a = float(logits[token_id_a].item())
    logit_b = float(logits[token_id_b].item())
    if correct_label == "A":
        return logit_a - logit_b
    return logit_b - logit_a


def _predicted_label_from_logits(logits: torch.Tensor, token_id_a: int, token_id_b: int) -> str:
    return "A" if logits[token_id_a] >= logits[token_id_b] else "B"


def _resolve_primary_device(model) -> torch.device:
    first = next(model.parameters())
    return first.device


class PositionPatcher:
    def __init__(self, module, token_position: int, vector: torch.Tensor):
        self.module = module
        self.token_position = token_position
        self.vector = vector
        self.handle = None

    def __enter__(self):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                raise TypeError(f"Expected tensor output from decoder block, got {type(output)}")
            patched = output.clone()
            patched[:, self.token_position, :] = patched[:, self.token_position, :] + self.vector
            return patched

        self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
        return False


def _prepare_prompt_inputs(tokenizer, prompt_text: str, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in _tokenize_for_scoring(tokenizer, prompt_text).items()
    }


def _run_prompt(
    model,
    model_inputs: Dict[str, torch.Tensor],
    token_position: int,
    token_id_a: int,
    token_id_b: int,
    correct_label: str,
    patch_module,
    patch_vector: Optional[torch.Tensor],
    device: torch.device = None,
) -> Dict[str, object]:
    if token_position >= int(model_inputs["input_ids"].shape[1]):
        raise IndexError(
            f"Patch position {token_position} out of range for prompt length "
            f"{int(model_inputs['input_ids'].shape[1])}"
        )

    with torch.inference_mode():
        if patch_vector is None:
            outputs = model(**model_inputs, use_cache=False)
        else:
            if device is not None:
                patch_vector = patch_vector.to(device=device)
            with PositionPatcher(patch_module, token_position, patch_vector):
                outputs = model(**model_inputs, use_cache=False)

    next_token_logits = outputs.logits[0, -1, :]
    signed_margin = _signed_margin_from_logits(
        next_token_logits, token_id_a=token_id_a, token_id_b=token_id_b, correct_label=correct_label
    )
    predicted_label = _predicted_label_from_logits(next_token_logits, token_id_a, token_id_b)
    return {
        "signed_margin": signed_margin,
        "predicted_label": predicted_label,
        "correct": predicted_label == correct_label,
        "logit_a": float(next_token_logits[token_id_a].item()),
        "logit_b": float(next_token_logits[token_id_b].item()),
    }


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else math.nan


def _summarize_records(records: Sequence[Dict]) -> Dict[str, Dict[str, object]]:
    grouped: Dict[Tuple[str, float], List[Dict]] = defaultdict(list)
    for row in records:
        grouped[(row["control"], float(row["alpha"]))].append(row)

    summary: Dict[str, Dict[str, object]] = {}
    for (control, alpha), rows in sorted(grouped.items()):
        key = f"{control}__alpha_{alpha:g}"
        margin_shifts = [row["patched_signed_margin"] - row["baseline_signed_margin"] for row in rows]
        flip_to_correct = [1.0 if (not row["baseline_correct"] and row["patched_correct"]) else 0.0 for row in rows]
        harmed = [1.0 if (row["baseline_correct"] and not row["patched_correct"]) else 0.0 for row in rows]
        summary[key] = {
            "control": control,
            "alpha": alpha,
            "n_items": len(rows),
            "flip_to_correct_rate": _mean(flip_to_correct),
            "harm_rate": _mean(harmed),
            "mean_margin_shift": _mean(margin_shifts),
            "mean_patched_margin": _mean([row["patched_signed_margin"] for row in rows]),
            "mean_baseline_margin": _mean([row["baseline_signed_margin"] for row in rows]),
        }
    return summary


def main() -> None:
    args = parse_args()
    if args.layer_index <= 0:
        raise ValueError("layer-index must be >= 1 because 0 is the embedding state.")
    resolved_loader_dtype = args.loader_dtype if args.loader_dtype is not None else args.dtype

    metadata = _load_metadata(args.extraction_dir / "metadata.jsonl")
    prompt_condition = _parse_prompt_condition(args.prompt_condition, args.instruction_text)
    i1a_all, i1a_correcting, i1a_negative = _split_i1a_rows(metadata)
    if not i1a_negative:
        raise SystemExit("No resisting/negative i1a rows found.")

    vector_payload_path: Optional[Path] = None
    if args.steering_vector_path is not None:
        vector_payload_path = args.steering_vector_path
        vector_payload = _load_vector_payload(args.steering_vector_path)
        vector = vector_payload["vector"].float()
    else:
        vector_payload_path = args.extraction_dir / "activations.pt"
        _, activations = _load_extraction(args.extraction_dir)
        vector = _build_within_i1a_vector(
            activations=activations,
            correcting_rows=i1a_correcting,
            negative_rows=i1a_negative,
            position=args.position,
            layer_index=args.layer_index,
        )
    random_vector = _matched_random_vector(vector, args.seed)
    wrong_sign_vector = -vector

    target_rows = _select_target_rows(metadata, args.target_subset)
    if args.max_items > 0:
        target_rows = target_rows[: args.max_items]

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        device=args.device,
        dtype=resolved_loader_dtype,
    )
    model.eval()
    device = _resolve_primary_device(model)
    token_id_a, token_id_b = get_ab_token_ids(tokenizer)
    examples_by_uid = {ex.uid: ex for ex in load_mc_dataset(args.mc_dataset_path)}

    block_index = args.layer_index - 1
    patch_module = model.model.layers[block_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "item_results.jsonl"

    all_rows: List[Dict] = []
    alphas = list(args.alphas)
    controls = (
        ("real", args.position, vector),
        ("wrong_sign", args.position, wrong_sign_vector),
        ("random", args.position, random_vector),
    )
    if args.wrong_site_position != "none":
        controls = controls + (("wrong_site", args.wrong_site_position, vector),)

    scaled_vectors: Dict[Tuple[str, float], Optional[torch.Tensor]] = {}
    for control_name, _position_name, base_vector in controls:
        for alpha in alphas:
            scaled_vectors[(control_name, float(alpha))] = (
                None if alpha == 0 else (alpha * base_vector).to(device=device)
            )

    prepared_eval_records: List[Dict[str, object]] = []
    for row_idx in target_rows:
        row = metadata[row_idx]
        example = examples_by_uid[row["uid"]]
        eval_record = _build_eval_prompt_record(
            tokenizer=tokenizer,
            row=row,
            example=example,
            condition=prompt_condition,
        )
        model_inputs = _prepare_prompt_inputs(tokenizer, str(eval_record["prompt_text"]), device)
        prepared_eval_records.append(
            {
                **eval_record,
                "model_inputs": model_inputs,
            }
        )

    with records_path.open("w") as out_file:
        for eval_record in tqdm(prepared_eval_records, desc=f"Patching {args.target_subset} i1a items"):
            baseline = _run_prompt(
                model=model,
                model_inputs=eval_record["model_inputs"],
                token_position=int(eval_record["positions"][args.position]),
                token_id_a=token_id_a,
                token_id_b=token_id_b,
                correct_label=str(eval_record["correct_label"]),
                patch_module=patch_module,
                patch_vector=None,
                device=device,
            )

            for control_name, position_name, base_vector in controls:
                patch_pos = eval_record["positions"].get(position_name)
                if patch_pos is None:
                    continue
                for alpha in alphas:
                    patched = _run_prompt(
                        model=model,
                        model_inputs=eval_record["model_inputs"],
                        token_position=int(patch_pos),
                        token_id_a=token_id_a,
                        token_id_b=token_id_b,
                        correct_label=str(eval_record["correct_label"]),
                        patch_module=patch_module,
                        patch_vector=scaled_vectors[(control_name, float(alpha))],
                        device=device,
                    )
                    out_row = {
                        "uid": eval_record["uid"],
                        "question": eval_record["question"],
                        "control": control_name,
                        "position": position_name,
                        "layer_index": args.layer_index,
                        "block_index": block_index,
                        "alpha": float(alpha),
                        "item_label": eval_record["item_label"],
                        "correct_label": eval_record["correct_label"],
                        "wrong_label": eval_record["wrong_label"],
                        "selection": eval_record["selection"],
                        "prompt_condition": args.prompt_condition,
                        "baseline_signed_margin": baseline["signed_margin"],
                        "patched_signed_margin": patched["signed_margin"],
                        "baseline_predicted_label": baseline["predicted_label"],
                        "patched_predicted_label": patched["predicted_label"],
                        "baseline_correct": baseline["correct"],
                        "patched_correct": patched["correct"],
                        "baseline_logits": {
                            "logit_a": baseline["logit_a"],
                            "logit_b": baseline["logit_b"],
                        },
                        "patched_logits": {
                            "logit_a": patched["logit_a"],
                            "logit_b": patched["logit_b"],
                        },
                    }
                    out_file.write(json.dumps(out_row) + "\n")
                    all_rows.append(out_row)

    summary = {
        "model": args.model,
        "extraction_dir": str(args.extraction_dir),
        "steering_vector_path": None if args.steering_vector_path is None else str(args.steering_vector_path),
        "vector_source": str(vector_payload_path),
        "output_dir": str(args.output_dir),
        "position": args.position,
        "wrong_site_position": args.wrong_site_position,
        "layer_index": args.layer_index,
        "block_index": block_index,
        "n_i1a_rows": len(i1a_all),
        "n_correcting_rows": len(i1a_correcting),
        "n_negative_rows_total": len(i1a_negative),
        "target_subset": args.target_subset,
        "prompt_condition": args.prompt_condition,
        "instruction_text": args.instruction_text if prompt_condition.instruction else None,
        "n_rows_evaluated": len(target_rows),
        "alphas": alphas,
        "vector_norm": float(vector.norm().item()),
        "random_vector_norm": float(random_vector.norm().item()),
        "records_path": str(records_path),
        "aggregates": _summarize_records(all_rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
