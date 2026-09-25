"""Activation steering experiment runner with multiple intervention modes."""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F
from tqdm import tqdm

from src.exp7.dataset_mc import MCExample, load_mc_dataset
from src.exp7.scoring import get_ab_token_ids
from src.exp10.conditions import Exp10Condition, generate_all_conditions
from src.exp16.dynamic_parser import discover_model_markers, dynamic_parse_row
from src.exp16.run_dissociation_test import (
    _format_chat_prompt,
    _format_exp16_prompt,
    _set_torch_seed,
    _trim_generated_output_ids,
)
from src.exp16.va_subspace import load_basis_by_layer, normalize_torch, project_out_torch
from src.mechanism.hooks import (
    _extract_hidden,
    _replace_hidden,
    get_component_modules,
    get_transformer_layers,
)
from src.mechanism.hooks_v2 import setup_h100_optimizations, try_compile_model
from src.models.llama_loader import load_model_and_tokenizer


DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_DIRECTION = Path("results/authority/mechanism/gpt_oss_compliance_analysis/primary_direction.pt")
DEFAULT_MASKS = Path("results/authority/mechanism/gpt_oss_authority_activations/label_masks.json")
DEFAULT_DATASET = Path("data/exp7_mc_dataset.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/authority/mechanism/steering")

ALPHAS_DEFAULT = "-15,-10,-5,-3,-1,0,1,3,5,10,15"
CONDITIONS_DEFAULT = "N0_note,W1_note"
TARGET_LAYERS_DEFAULT = "auto"
POSITION_MODES_DEFAULT = "note_end"
COMPONENTS_DEFAULT = "block_output"
ENDORSEMENT_STYLE = "authoritative_verified"
DEFAULT_INSTRUCTION = "Answer correctly even if the speaker is wrong. Prioritize factual accuracy."

STYLE_NON_THINK = ("weak", "uncertain", "assertive", "authoritative_verified")
NOTE_LINE_RE = re.compile(r"^Note:.*$", re.MULTILINE)


@dataclass
class TargetSpec:
    layer: int
    component: str
    direction: Optional[torch.Tensor]
    patch_mean: Optional[torch.Tensor]
    nuisance_basis: Optional[torch.Tensor] = None


class ExtractionStore:
    def __init__(self, *, activations_payload: Dict[str, Any], metadata_rows: List[Dict[str, Any]]):
        self.activations: Dict[str, torch.Tensor] = activations_payload["activations"]
        self.position_names: List[str] = list(activations_payload["position_names"])
        self.target_layers: List[int] = list(activations_payload["target_layers"])
        self.layer_to_index: Dict[int, int] = {layer: i for i, layer in enumerate(self.target_layers)}
        self.idx_by_uid_style_cond: Dict[Tuple[str, str, str], int] = {}
        for idx, row in enumerate(metadata_rows):
            self.idx_by_uid_style_cond[(row["uid"], row["style"], row["condition_code"])] = idx

    def _indices(self, *, uids: Sequence[str], style: str, condition_code: str) -> List[int]:
        out: List[int] = []
        for uid in uids:
            idx = self.idx_by_uid_style_cond.get((uid, style, condition_code))
            if idx is not None:
                out.append(idx)
        return out

    def matrix(
        self,
        *,
        position: str,
        layer_value: int,
        uids: Sequence[str],
        style: str,
        condition_code: str,
    ) -> torch.Tensor:
        indices = self._indices(uids=uids, style=style, condition_code=condition_code)
        if not indices:
            hidden = self.activations[position]
            return torch.zeros((0, hidden.shape[-1]), dtype=torch.float32)
        layer_idx = self.layer_to_index[layer_value]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        return self.activations[position][idx_tensor, layer_idx, :].float().cpu()

    def mean(
        self,
        *,
        position: str,
        layer_value: int,
        uids: Sequence[str],
        style: str,
        condition_code: str,
    ) -> Optional[torch.Tensor]:
        mat = self.matrix(
            position=position,
            layer_value=layer_value,
            uids=uids,
            style=style,
            condition_code=condition_code,
        )
        if mat.shape[0] == 0:
            return None
        return mat.mean(dim=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Activation steering with flexible intervention modes.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--direction-path", type=Path, default=DEFAULT_DIRECTION)
    p.add_argument("--direction-key", type=str, default="vector",
                   help="Tensor key in direction file (e.g., vector, shared_within_label).")
    p.add_argument("--direction-position", type=str, default="auto",
                   help="Direction position for extraction-backed vectors (or auto).")
    p.add_argument("--direction-kind", type=str, default="shared_within_label",
                   choices=(
                       "pooled_authority",
                       "content",
                       "endorsement_presence",
                       "authority_given_correct",
                       "authority_given_wrong",
                       "shared_within_label",
                       "w1_minus_c1",
                       "c1_minus_w1",
                   ),
                   help="Direction type when deriving vectors from extraction artifacts.")
    p.add_argument("--extraction-dir", type=Path, default=None,
                   help="If set, derive layer/position-specific directions from activations in this dir.")
    p.add_argument("--patch-source-style", type=str, default="authoritative_verified")
    p.add_argument("--patch-source-condition", type=str, default="W1_note")
    p.add_argument("--patch-source-position", type=str, default="auto")
    p.add_argument("--masks-path", type=Path, default=DEFAULT_MASKS)
    p.add_argument("--uids-file", type=Path, default=None,
                   help="Optional newline-delimited UID list for evaluation rows (overrides --uids-subset).")
    p.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--alphas", type=str, default=ALPHAS_DEFAULT,
                   help="Comma-separated steering multipliers.")
    p.add_argument("--conditions", type=str, default=CONDITIONS_DEFAULT,
                   help="Comma-separated condition codes to steer on.")
    p.add_argument("--uids-subset", choices=("w1", "c1", "all"), default="w1",
                   help="Which UID mask to use for evaluation rows.")
    p.add_argument("--direction-uids-subset", choices=("w1", "c1", "all"), default=None,
                   help="Optional UID mask override for direction extraction.")
    p.add_argument("--direction-uids-file", type=Path, default=None,
                   help="Optional newline-delimited UID list for direction extraction (overrides --direction-uids-subset).")
    p.add_argument("--patch-uids-subset", choices=("w1", "c1", "all"), default=None,
                   help="Optional UID mask override for patch mean vectors.")
    p.add_argument("--patch-uids-file", type=Path, default=None,
                   help="Optional newline-delimited UID list for patch means (overrides --patch-uids-subset).")
    p.add_argument("--max-items", type=int, default=0, help="Cap item count (0=all).")
    p.add_argument("--target-layers", type=str, default=TARGET_LAYERS_DEFAULT,
                   help="Comma-separated layer ids or 'auto'.")
    p.add_argument("--single-layer-sweep", action="store_true",
                   help="If target-layers has multiple values, run one config per single layer.")
    p.add_argument("--position-modes", type=str, default=POSITION_MODES_DEFAULT,
                   help="Comma-separated intervention position modes.")
    p.add_argument("--components", type=str, default=COMPONENTS_DEFAULT,
                   help="Comma-separated components: block_output,attention_output,mlp_output.")
    p.add_argument("--apply-phase", choices=("prompt", "decode", "both"), default="prompt")
    p.add_argument(
        "--intervention-mode",
        choices=("add", "subtract", "replace_mean", "interpolate_mean", "project_out_direction", "project_out_subspace"),
        default="add",
    )
    p.add_argument("--norm-scaling", choices=("none", "resid_norm", "resid_std"), default="none")
    p.add_argument("--project-out-subspace-path", type=Path, default=None,
                   help="Optional nuisance subspace payload with basis_by_layer; residualizes directions or patch deltas.")
    p.add_argument("--sparse-topk", type=int, default=0,
                   help="If >0, keep only top-k absolute dimensions of direction vectors.")
    p.add_argument("--collect-margin-diagnostics", action="store_true",
                   help="Compute first-token A/B logit diagnostics for each alpha.")
    p.add_argument("--batch-size", type=int, default=8, help="Upper bound; final batches are token-dynamic.")
    p.add_argument("--max-batch-tokens", type=int, default=16384, help="Approx prompt token budget per batch.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--answer-suffix", type=str, default="",
                   help="Suffix appended after the base prompt (e.g. freegen instruction).")
    p.add_argument("--greedy", action="store_true", help="Use greedy decoding (do_sample=False).")
    p.add_argument("--no-cache", action="store_true", help="Disable generation KV cache.")
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile optimization.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plain-prompt", action="store_true")
    return p.parse_args()


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_alphas(raw: str) -> List[float]:
    return [float(x) for x in _parse_csv(raw)]


def _parse_target_layers(raw: str, *, default_layer: int, n_layers: int) -> List[int]:
    if raw.strip().lower() == "auto":
        layers = [int(default_layer)]
    else:
        layers = [int(x) for x in _parse_csv(raw)]
    deduped: List[int] = []
    seen = set()
    for layer in layers:
        if layer in seen:
            continue
        seen.add(layer)
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Target layer {layer} out of range for model with {n_layers} layers.")
        deduped.append(layer)
    if not deduped:
        raise ValueError("No valid target layers provided.")
    return deduped


def _parse_position_modes(raw: str) -> List[str]:
    valid = {
        "note_start",
        "note_end",
        "endorsed_answer",
        "answer_position",
        "endorsement_span",
        "all_prompt_tokens",
        "last_prompt_token",
        "decode_last_token",
    }
    out = _parse_csv(raw)
    for mode in out:
        if mode not in valid:
            raise ValueError(f"Unknown position mode: {mode}. Valid={sorted(valid)}")
    return out


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_extraction_store(extraction_dir: Path) -> ExtractionStore:
    activations_path = extraction_dir / "activations.pt"
    metadata_path = extraction_dir / "metadata.jsonl"
    payload = torch.load(activations_path, map_location="cpu")
    metadata = _load_jsonl(metadata_path)
    return ExtractionStore(activations_payload=payload, metadata_rows=metadata)


def _load_direction_file(path: Path, key: str) -> Tuple[torch.Tensor | Dict[int, torch.Tensor], int, str]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        raw = None
        if key == "vector" and "vector" in payload:
            raw = payload["vector"]
        elif key in payload:
            raw = payload[key]
        elif "vector" in payload:
            raw = payload["vector"]
        if isinstance(raw, dict):
            vec: torch.Tensor | Dict[int, torch.Tensor] = {}
            for layer_key, value in raw.items():
                layer_val = int(layer_key)
                t = value.float() if torch.is_tensor(value) else torch.tensor(value, dtype=torch.float32)
                n = t.norm()
                if float(n) > 0.0:
                    t = t / n
                vec[layer_val] = t
            if not vec:
                raise ValueError(f"Direction mapping '{key}' in {path} is empty")
        elif torch.is_tensor(raw):
            vec = raw.float()
            n = vec.norm()
            if float(n) > 0.0:
                vec = vec / n
        else:
            raise ValueError(f"Could not find direction tensor key '{key}' in {path}")
        layer = int(payload.get("layer", 0))
        position = str(payload.get("position", "endorsement_end"))
    elif torch.is_tensor(payload):
        vec = payload.float()
        layer = 0
        position = "endorsement_end"
        n = vec.norm()
        if float(n) > 0.0:
            vec = vec / n
    else:
        raise ValueError(f"Unsupported direction payload type: {type(payload)}")
    return vec, layer, position


def _load_masks(masks_path: Path) -> Dict[str, Any]:
    return json.loads(masks_path.read_text())


def _load_uids_from_masks(masks: Dict[str, Any], subset: str) -> List[str]:
    if subset == "w1":
        return list(masks["mask_primary_w1"])
    if subset == "c1":
        return list(masks["mask_primary_c1"])
    return list(masks["mask_primary"])


def _load_uids_from_file(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_condition_map(codes: Sequence[str]) -> Dict[str, Exp10Condition]:
    all_conditions = generate_all_conditions(
        tags=["Expert", "Note", "User", "Someone online"],
        instruction_text=DEFAULT_INSTRUCTION,
    )
    mapping = {c.code: c for c in all_conditions}
    missing = [code for code in codes if code not in mapping]
    if missing:
        raise ValueError(f"Unknown condition codes: {missing}")
    return {code: mapping[code] for code in codes}


def _extract_note_line(prompt_text: str) -> Optional[str]:
    m = NOTE_LINE_RE.search(prompt_text)
    return None if m is None else m.group(0)


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> Optional[Tuple[int, int]]:
    if not needle or len(needle) > len(haystack):
        return None
    end = len(haystack) - len(needle) + 1
    for start in range(end):
        if list(haystack[start : start + len(needle)]) == list(needle):
            return start, start + len(needle) - 1
    return None


def _compute_prompt_positions(
    *,
    tokenizer,
    model_prompt: str,
    prompt_text: str,
    condition_code: str,
    correct_label: str,
    wrong_label: str,
) -> Dict[str, Any]:
    ids_full = tokenizer(model_prompt, add_special_tokens=False).input_ids
    out: Dict[str, Any] = {
        "endorsement_start": None,
        "endorsement_end": None,
        "endorsed_answer": None,
        "answer_position": len(ids_full) - 1 if ids_full else None,
    }
    note_line = _extract_note_line(prompt_text)
    if note_line:
        note_char = model_prompt.find(note_line)
        if note_char >= 0:
            prefix_ids = tokenizer(model_prompt[:note_char], add_special_tokens=False).input_ids
            note_ids = tokenizer(note_line, add_special_tokens=False).input_ids
            if note_ids:
                note_start = len(prefix_ids)
                note_end = note_start + len(note_ids) - 1
                if not (0 <= note_start <= note_end < len(ids_full)):
                    found = _find_subsequence(ids_full, note_ids)
                    if found is not None:
                        note_start, note_end = found
                out["endorsement_start"] = note_start
                out["endorsement_end"] = note_end

                if condition_code in {"C1_note", "W1_note"}:
                    endorsed = correct_label if condition_code == "C1_note" else wrong_label
                    needle = tokenizer(f" {endorsed}", add_special_tokens=False).input_ids
                    if not needle:
                        needle = tokenizer(endorsed, add_special_tokens=False).input_ids
                    if needle:
                        segment = ids_full[note_start : note_end + 1]
                        found = _find_subsequence(segment, needle)
                        if found is not None:
                            out["endorsed_answer"] = note_start + found[0]

    answer_idx = model_prompt.find("Answer:")
    if answer_idx >= 0:
        prefix_ids = tokenizer(model_prompt[: answer_idx + len("Answer:")], add_special_tokens=False).input_ids
        if prefix_ids:
            out["answer_position"] = len(prefix_ids) - 1

    return out


def _build_jobs(
    examples_by_uid: Dict[str, MCExample],
    uids: List[str],
    condition_map: Dict[str, Exp10Condition],
    conditions: List[str],
    tokenizer,
    plain_prompt: bool,
    answer_suffix: str = "",
) -> List[Dict[str, Any]]:
    jobs = []
    for uid in uids:
        if uid not in examples_by_uid:
            continue
        ex = examples_by_uid[uid]
        for code in conditions:
            cond = condition_map.get(code)
            if cond is None:
                continue
            base = _format_exp16_prompt(ex=ex, condition=cond, endorsement_style=ENDORSEMENT_STYLE).rstrip()
            if answer_suffix:
                prompt_text = f"{base}\n{answer_suffix}".strip()
            else:
                prompt_text = f"{base}\n"
            model_prompt = _format_chat_prompt(tokenizer, prompt_text, plain_prompt)
            positions = _compute_prompt_positions(
                tokenizer=tokenizer,
                model_prompt=model_prompt,
                prompt_text=prompt_text,
                condition_code=code,
                correct_label=ex.correct_label,
                wrong_label=ex.wrong_label,
            )
            jobs.append(
                {
                    "uid": uid,
                    "condition_code": code,
                    "prompt_text": prompt_text,
                    "model_prompt": model_prompt,
                    "prompt_tokens": len(tokenizer(model_prompt, add_special_tokens=False).input_ids),
                    "positions_unpadded": positions,
                    "correct_label": ex.correct_label,
                    "wrong_label": ex.wrong_label,
                    "correct_text": ex.option_a if ex.correct_label == "A" else ex.option_b,
                    "wrong_text": ex.option_b if ex.correct_label == "A" else ex.option_a,
                }
            )
    return jobs


def _dynamic_batches(
    jobs: Sequence[Dict[str, Any]],
    *,
    max_batch_size: int,
    max_batch_tokens: int,
) -> List[List[Dict[str, Any]]]:
    ordered = sorted(jobs, key=lambda row: int(row.get("prompt_tokens", 0)))
    out: List[List[Dict[str, Any]]] = []
    batch: List[Dict[str, Any]] = []
    max_len = 0
    for job in ordered:
        n_tok = int(job.get("prompt_tokens", 0))
        if not batch:
            batch = [job]
            max_len = n_tok
            continue
        cand_max = max(max_len, n_tok)
        if len(batch) + 1 > max_batch_size or cand_max * (len(batch) + 1) > max_batch_tokens:
            out.append(batch)
            batch = [job]
            max_len = n_tok
        else:
            batch.append(job)
            max_len = cand_max
    if batch:
        out.append(batch)
    return out


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if k != "parse_meta"}
            f.write(json.dumps(clean, default=str) + "\n")


def _normalize_vector(v: torch.Tensor) -> torch.Tensor:
    n = v.norm()
    if float(n) == 0.0:
        return v.float()
    return (v / n).float()


def _sparsify_vector(v: torch.Tensor, topk: int) -> torch.Tensor:
    vec = v.float().clone()
    if topk <= 0 or topk >= vec.numel():
        return _normalize_vector(vec)
    idx = torch.topk(vec.abs(), k=topk).indices
    out = torch.zeros_like(vec)
    out[idx] = vec[idx]
    return _normalize_vector(out)


def _direction_from_store(
    store: ExtractionStore,
    *,
    uids: Sequence[str],
    position: str,
    layer_value: int,
) -> Dict[str, torch.Tensor]:
    m_c1 = store.matrix(
        position=position,
        layer_value=layer_value,
        uids=uids,
        style="authoritative_verified",
        condition_code="C1_note",
    )
    m_w1 = store.matrix(
        position=position,
        layer_value=layer_value,
        uids=uids,
        style="authoritative_verified",
        condition_code="W1_note",
    )
    m_n0 = store.matrix(
        position=position,
        layer_value=layer_value,
        uids=uids,
        style="authoritative_verified",
        condition_code="N0_note",
    )

    dim = int(store.activations[position].shape[-1])
    zeros = torch.zeros(dim, dtype=torch.float32)
    if min(m_c1.shape[0], m_w1.shape[0], m_n0.shape[0]) == 0:
        return {
            "pooled_authority": zeros.clone(),
            "content": zeros.clone(),
            "endorsement_presence": zeros.clone(),
            "authority_given_correct": zeros.clone(),
            "authority_given_wrong": zeros.clone(),
            "shared_within_label": zeros.clone(),
            "w1_minus_c1": zeros.clone(),
            "c1_minus_w1": zeros.clone(),
        }

    mean_c1 = m_c1.mean(dim=0)
    mean_w1 = m_w1.mean(dim=0)
    mean_n0 = m_n0.mean(dim=0)

    c1_parts: List[torch.Tensor] = []
    w1_parts: List[torch.Tensor] = []
    for style in STYLE_NON_THINK:
        c1 = store.matrix(
            position=position,
            layer_value=layer_value,
            uids=uids,
            style=style,
            condition_code="C1_note",
        )
        w1 = store.matrix(
            position=position,
            layer_value=layer_value,
            uids=uids,
            style=style,
            condition_code="W1_note",
        )
        if c1.shape[0]:
            c1_parts.append(c1)
        if w1.shape[0]:
            w1_parts.append(w1)

    mean_all_c1 = torch.cat(c1_parts, dim=0).mean(dim=0) if c1_parts else zeros.clone()
    mean_all_w1 = torch.cat(w1_parts, dim=0).mean(dim=0) if w1_parts else zeros.clone()

    authority_given_correct = mean_c1 - mean_n0
    authority_given_wrong = mean_w1 - mean_n0
    w1_minus_c1 = mean_w1 - mean_c1
    c1_minus_w1 = mean_c1 - mean_w1
    pooled_authority = 0.5 * (authority_given_correct + authority_given_wrong)
    content = mean_all_c1 - mean_all_w1
    endorsement_presence = 0.5 * (mean_all_c1 + mean_all_w1) - mean_n0
    shared_within_label = 0.5 * (authority_given_correct + authority_given_wrong)

    return {
        "pooled_authority": _normalize_vector(pooled_authority),
        "content": _normalize_vector(content),
        "endorsement_presence": _normalize_vector(endorsement_presence),
        "authority_given_correct": _normalize_vector(authority_given_correct),
        "authority_given_wrong": _normalize_vector(authority_given_wrong),
        "shared_within_label": _normalize_vector(shared_within_label),
        "w1_minus_c1": _normalize_vector(w1_minus_c1),
        "c1_minus_w1": _normalize_vector(c1_minus_w1),
    }


def _build_positions_for_batch(
    *,
    jobs: Sequence[Dict[str, Any]],
    attention_mask: Optional[torch.Tensor],
    input_width: int,
    tokenizer,
    position_mode: str,
) -> List[List[int]]:
    padding_side = str(getattr(tokenizer, "padding_side", "right")).lower()
    out: List[List[int]] = []
    for row_idx, job in enumerate(jobs):
        if attention_mask is not None:
            valid = torch.nonzero(attention_mask[row_idx], as_tuple=False).squeeze(-1)
            if valid.numel():
                first_valid = int(valid[0].item())
                last_valid = int(valid[-1].item())
            else:
                first_valid = 0
                last_valid = input_width - 1
        else:
            first_valid = 0
            last_valid = input_width - 1

        if position_mode == "all_prompt_tokens":
            out.append(list(range(first_valid, last_valid + 1)))
            continue
        if position_mode == "last_prompt_token":
            out.append([last_valid])
            continue
        if position_mode == "decode_last_token":
            out.append([])
            continue

        pos_data = job["positions_unpadded"]
        if position_mode == "note_start":
            raw_pos = pos_data.get("endorsement_start")
        elif position_mode == "note_end":
            raw_pos = pos_data.get("endorsement_end")
        elif position_mode == "endorsed_answer":
            raw_pos = pos_data.get("endorsed_answer")
        elif position_mode == "answer_position":
            raw_pos = pos_data.get("answer_position")
        elif position_mode == "endorsement_span":
            start = pos_data.get("endorsement_start")
            end = pos_data.get("endorsement_end")
            if isinstance(start, int) and isinstance(end, int) and end >= start:
                raw_positions = list(range(start, end + 1))
            else:
                raw_positions = []
            if padding_side == "left":
                shifted = [first_valid + pos for pos in raw_positions]
            else:
                shifted = raw_positions
            valid_positions = [p for p in shifted if first_valid <= p <= last_valid]
            out.append(valid_positions if valid_positions else [last_valid])
            continue
        else:
            raise ValueError(f"Unsupported position mode: {position_mode}")

        if isinstance(raw_pos, int):
            pos = first_valid + raw_pos if padding_side == "left" else raw_pos
        else:
            pos = last_valid
        if pos < first_valid or pos > last_valid:
            pos = last_valid
        out.append([int(pos)])
    return out


def _scale_delta(base_hidden: torch.Tensor, direction: torch.Tensor, alpha: float, scaling: str) -> torch.Tensor:
    if scaling == "none":
        return alpha * direction
    if scaling == "resid_norm":
        scale = float(base_hidden.norm().item())
        return (alpha * scale) * direction
    if scaling == "resid_std":
        scale = float(base_hidden.std(unbiased=False).item())
        return (alpha * scale) * direction
    raise ValueError(f"Unknown scaling mode: {scaling}")


def _make_intervention_hook(
    *,
    spec: TargetSpec,
    alpha: float,
    intervention_mode: str,
    norm_scaling: str,
    apply_prompt: bool,
    apply_decode: bool,
    prompt_positions_by_row: Sequence[Sequence[int]],
):
    def hook(_module, _inputs, output):
        hidden = _extract_hidden(output)
        is_decode_step = hidden.shape[1] == 1

        if is_decode_step:
            if not apply_decode:
                return output
            positions_by_row = [[0] for _ in range(hidden.shape[0])]
        else:
            if not apply_prompt:
                return output
            positions_by_row = prompt_positions_by_row

        updated = hidden.clone()
        d = None
        p = None
        if spec.direction is not None:
            d = spec.direction.to(hidden.device, dtype=hidden.dtype)
        if spec.patch_mean is not None:
            p = spec.patch_mean.to(hidden.device, dtype=hidden.dtype)
        nuisance_basis = None
        if spec.nuisance_basis is not None:
            nuisance_basis = spec.nuisance_basis.to(hidden.device, dtype=hidden.dtype)

        for row_idx, row_positions in enumerate(positions_by_row):
            if row_idx >= hidden.shape[0]:
                break
            valid_positions = [int(pos) for pos in row_positions if 0 <= int(pos) < hidden.shape[1]]
            if not valid_positions:
                continue
            for pos in valid_positions:
                base = hidden[row_idx, pos, :]
                if intervention_mode in {"add", "subtract"}:
                    if d is None:
                        continue
                    direction = d if nuisance_basis is None else normalize_torch(project_out_torch(d, nuisance_basis))
                    if float(direction.norm()) == 0.0:
                        continue
                    delta = _scale_delta(base, direction, alpha, norm_scaling)
                    if intervention_mode == "subtract":
                        delta = -delta
                    updated[row_idx, pos, :] = base + delta
                elif intervention_mode == "replace_mean":
                    if p is None:
                        continue
                    target = p
                    if nuisance_basis is not None:
                        target = base + project_out_torch(p - base, nuisance_basis)
                    updated[row_idx, pos, :] = target
                elif intervention_mode == "interpolate_mean":
                    if p is None:
                        continue
                    delta_to_patch = p - base
                    if nuisance_basis is not None:
                        delta_to_patch = project_out_torch(delta_to_patch, nuisance_basis)
                    updated[row_idx, pos, :] = base + alpha * delta_to_patch
                elif intervention_mode == "project_out_direction":
                    if d is None:
                        continue
                    direction = d if nuisance_basis is None else normalize_torch(project_out_torch(d, nuisance_basis))
                    if float(direction.norm()) == 0.0:
                        continue
                    # Partial refusal-axis ablation:
                    # alpha=1 removes the full projection onto the steering direction.
                    coeff = torch.dot(base, direction)
                    updated[row_idx, pos, :] = base - alpha * coeff * direction
                elif intervention_mode == "project_out_subspace":
                    if nuisance_basis is None:
                        raise ValueError("--intervention-mode project_out_subspace requires --project-out-subspace-path")
                    residual = project_out_torch(base, nuisance_basis)
                    projection = base - residual
                    updated[row_idx, pos, :] = base - alpha * projection
                else:
                    raise ValueError(f"Unknown intervention mode: {intervention_mode}")
        return _replace_hidden(output, updated)

    return hook


def _collect_margin_stats(
    *,
    model,
    model_inputs: Dict[str, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    token_id_a: int,
    token_id_b: int,
    jobs: Sequence[Dict[str, Any]],
) -> List[Dict[str, float]]:
    with torch.inference_mode():
        outputs = model(**model_inputs, use_cache=False)
    logits = outputs.logits
    if attention_mask is not None:
        last_idx = attention_mask.sum(dim=1) - 1
    else:
        last_idx = torch.full((logits.shape[0],), logits.shape[1] - 1, dtype=torch.long, device=logits.device)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    final_logits = logits[batch_idx, last_idx, :]
    log_probs = F.log_softmax(final_logits, dim=-1)

    out: List[Dict[str, float]] = []
    for i, job in enumerate(jobs):
        logp_a = float(log_probs[i, token_id_a].item())
        logp_b = float(log_probs[i, token_id_b].item())
        pa = float(torch.exp(log_probs[i, token_id_a]).item())
        pb = float(torch.exp(log_probs[i, token_id_b]).item())
        total = pa + pb
        fc_a = pa / total if total > 0 else 0.5
        fc_b = pb / total if total > 0 else 0.5
        logit_a = logp_a - logp_b
        logit_b = -logit_a
        wrong_label = job["wrong_label"]
        out.append(
            {
                "fc_prob_a": fc_a,
                "fc_prob_b": fc_b,
                "fc_logit_a": logit_a,
                "fc_logit_b": logit_b,
                "fc_logit_wrong": logit_a if wrong_label == "A" else logit_b,
            }
        )
    return out


def _generate_steered_batch(
    jobs: Sequence[Dict[str, Any]],
    *,
    model,
    tokenizer,
    component_modules: Dict[Tuple[int, str], Any],
    target_specs: Sequence[TargetSpec],
    alpha: float,
    position_mode: str,
    apply_phase: str,
    intervention_mode: str,
    norm_scaling: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    greedy: bool,
    use_cache: bool,
    seed: int,
    collect_margin_diagnostics: bool,
    token_id_a: Optional[int],
    token_id_b: Optional[int],
) -> List[Dict[str, Any]]:
    prompts = [j["model_prompt"] for j in jobs]
    model_inputs = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to(model.device)
    input_width = int(model_inputs["input_ids"].shape[1])
    attention_mask = model_inputs.get("attention_mask")
    prompt_positions_by_row = _build_positions_for_batch(
        jobs=jobs,
        attention_mask=attention_mask,
        input_width=input_width,
        tokenizer=tokenizer,
        position_mode=position_mode,
    )

    apply_prompt = apply_phase in {"prompt", "both"}
    apply_decode = apply_phase in {"decode", "both"}
    if position_mode == "decode_last_token":
        apply_prompt = False
        apply_decode = True

    handles = []
    for spec in target_specs:
        module = component_modules[(spec.layer, spec.component)]
        handles.append(
            module.register_forward_hook(
                _make_intervention_hook(
                    spec=spec,
                    alpha=alpha,
                    intervention_mode=intervention_mode,
                    norm_scaling=norm_scaling,
                    apply_prompt=apply_prompt,
                    apply_decode=apply_decode,
                    prompt_positions_by_row=prompt_positions_by_row,
                )
            )
        )

    do_sample = (not greedy) and temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else 1.0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": use_cache,
    }
    if do_sample:
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = top_k
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

    margin_rows: Optional[List[Dict[str, float]]] = None
    try:
        if collect_margin_diagnostics and token_id_a is not None and token_id_b is not None:
            margin_rows = _collect_margin_stats(
                model=model,
                model_inputs=model_inputs,
                attention_mask=attention_mask,
                token_id_a=token_id_a,
                token_id_b=token_id_b,
                jobs=jobs,
            )

        _set_torch_seed(seed)
        with torch.inference_mode():
            generated_ids = model.generate(**model_inputs, **gen_kwargs)
    finally:
        for h in handles:
            h.remove()

    results = []
    for idx, job in enumerate(jobs):
        output_ids = _trim_generated_output_ids(
            generated_ids[idx],
            input_width=input_width,
            pad_token_id=gen_kwargs["pad_token_id"],
            eos_token_id=tokenizer.eos_token_id,
        )
        raw_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        row = {
            "uid": job["uid"],
            "condition_code": job["condition_code"],
            "alpha": alpha,
            "raw_text": raw_text,
            "correct_label": job["correct_label"],
            "wrong_label": job["wrong_label"],
            "correct_text": job["correct_text"],
            "wrong_text": job["wrong_text"],
        }
        if margin_rows is not None:
            row.update(margin_rows[idx])
        results.append(row)
    return results


def _wrap_for_parser(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **r,
        "generation": {"raw_text": r["raw_text"]},
        "correct_answer": r["correct_text"],
        "wrong_answer": r["wrong_text"],
    }


def _parse_and_score(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    wrapped = [_wrap_for_parser(r) for r in rows]
    markers = discover_model_markers(wrapped)
    for row, w in zip(rows, wrapped):
        label, meta = dynamic_parse_row(w, markers)
        row["parsed_label"] = label
        row["parse_meta"] = meta
        if label is not None:
            row["is_correct"] = label == row["correct_label"]
            row["chose_wrong"] = label == row["wrong_label"]
        else:
            row["is_correct"] = None
            row["chose_wrong"] = None

    grouped: Dict[Tuple[str, float, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row["config_id"], float(row["alpha"]), row["condition_code"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (cfg, alpha, cond), group in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        parsed = [r for r in group if r["parsed_label"] is not None]
        correct = [r for r in parsed if r["is_correct"]]
        wrong = [r for r in parsed if r["chose_wrong"]]
        margin = [float(r["fc_logit_wrong"]) for r in group if "fc_logit_wrong" in r]
        summary.append(
            {
                "config_id": cfg,
                "alpha": alpha,
                "condition": cond,
                "total": len(group),
                "parsed": len(parsed),
                "parse_rate": len(parsed) / len(group) if group else 0.0,
                "accuracy": len(correct) / len(parsed) if parsed else None,
                "wrong_rate": len(wrong) / len(parsed) if parsed else None,
                "flip_rate": len(wrong) / len(parsed) if parsed and cond == "W1_note" else None,
                "mean_fc_logit_wrong": sum(margin) / len(margin) if margin else None,
                "std_fc_logit_wrong": (
                    float(torch.tensor(margin).std(unbiased=False).item()) if len(margin) > 1 else 0.0 if margin else None
                ),
            }
        )
    return {"rows": rows, "summary": summary}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_h100_optimizations()

    alphas = _parse_alphas(args.alphas)
    conditions = _parse_csv(args.conditions)
    position_modes = _parse_position_modes(args.position_modes)
    components = _parse_csv(args.components)
    if not components:
        raise ValueError("At least one component is required.")

    masks_path = args.masks_path
    if args.extraction_dir is not None and not masks_path.exists():
        candidate = args.extraction_dir / "label_masks.json"
        if candidate.exists():
            masks_path = candidate
    masks = _load_masks(masks_path)
    if args.uids_file is not None:
        eval_uids = [line.strip() for line in args.uids_file.read_text().splitlines() if line.strip()]
    else:
        eval_uids = _load_uids_from_masks(masks, args.uids_subset)
    if args.max_items > 0:
        eval_uids = eval_uids[: args.max_items]
    if not eval_uids:
        raise SystemExit("No evaluation UIDs selected.")

    file_direction, file_layer, file_position = _load_direction_file(args.direction_path, args.direction_key)
    extraction_store: Optional[ExtractionStore] = None
    if args.extraction_dir is not None:
        extraction_store = _load_extraction_store(args.extraction_dir)

    direction_position = file_position if args.direction_position == "auto" else args.direction_position
    patch_source_position = direction_position if args.patch_source_position == "auto" else args.patch_source_position

    direction_subset = args.direction_uids_subset or args.uids_subset
    patch_subset = args.patch_uids_subset or direction_subset
    direction_uids = _load_uids_from_file(args.direction_uids_file) if args.direction_uids_file else _load_uids_from_masks(masks, direction_subset)
    patch_uids = _load_uids_from_file(args.patch_uids_file) if args.patch_uids_file else _load_uids_from_masks(masks, patch_subset)

    mc_examples = load_mc_dataset(args.dataset_path)
    examples_by_uid = {ex.uid: ex for ex in mc_examples}
    condition_map = _build_condition_map(conditions)

    model, tokenizer = load_model_and_tokenizer(model_name=args.model, device="auto", dtype="auto")
    model.eval()
    if not args.no_compile:
        model = try_compile_model(model)
    layers = get_transformer_layers(model)
    n_layers = len(layers)
    component_modules = get_component_modules(model)

    target_layers = _parse_target_layers(args.target_layers, default_layer=file_layer, n_layers=n_layers)
    layer_groups = [[layer] for layer in target_layers] if args.single_layer_sweep else [target_layers]

    jobs = _build_jobs(
        examples_by_uid=examples_by_uid,
        uids=eval_uids,
        condition_map=condition_map,
        conditions=conditions,
        tokenizer=tokenizer,
        plain_prompt=args.plain_prompt,
        answer_suffix=args.answer_suffix,
    )
    print(
        f"Model loaded: {args.model}, layers={n_layers} | jobs={len(jobs)} "
        f"({len(eval_uids)} items x {len(conditions)} conditions)"
    )
    print(
        f"Intervention: mode={args.intervention_mode}, components={components}, "
        f"position_modes={position_modes}, layer_groups={layer_groups}"
    )

    token_id_a: Optional[int] = None
    token_id_b: Optional[int] = None
    if args.collect_margin_diagnostics:
        try:
            token_id_a, token_id_b = get_ab_token_ids(tokenizer)
        except Exception as exc:
            print(f"Warning: margin diagnostics disabled ({exc})")
            args.collect_margin_diagnostics = False

    direction_cache: Dict[Tuple[int, str], torch.Tensor] = {}
    patch_cache: Dict[Tuple[int, str, str, str], torch.Tensor] = {}
    nuisance_basis_by_layer = load_basis_by_layer(args.project_out_subspace_path) if args.project_out_subspace_path else {}

    def direction_for(layer_val: int) -> torch.Tensor:
        key = (layer_val, direction_position)
        if key in direction_cache:
            return direction_cache[key]
        if extraction_store is None:
            if isinstance(file_direction, dict):
                if layer_val not in file_direction:
                    raise ValueError(f"Layer {layer_val} not present in direction mapping from {args.direction_path}")
                vec = file_direction[layer_val].clone()
            else:
                vec = file_direction.clone()
        else:
            if direction_position not in extraction_store.position_names:
                raise ValueError(
                    f"Direction position '{direction_position}' not in extraction positions {extraction_store.position_names}"
                )
            if layer_val not in extraction_store.layer_to_index:
                raise ValueError(f"Layer {layer_val} not available in extraction target layers {extraction_store.target_layers}")
            dirs = _direction_from_store(
                extraction_store,
                uids=direction_uids,
                position=direction_position,
                layer_value=layer_val,
            )
            vec = dirs[args.direction_kind]
        vec = _sparsify_vector(vec, args.sparse_topk)
        nuisance_basis = nuisance_basis_by_layer.get(layer_val)
        if nuisance_basis is not None:
            vec = normalize_torch(project_out_torch(vec, nuisance_basis))
        direction_cache[key] = vec
        return vec

    def patch_mean_for(layer_val: int) -> torch.Tensor:
        k = (layer_val, patch_source_position, args.patch_source_style, args.patch_source_condition)
        if k in patch_cache:
            return patch_cache[k]
        if extraction_store is None:
            raise ValueError("Patch modes require --extraction-dir for source activations.")
        if patch_source_position not in extraction_store.position_names:
            raise ValueError(
                f"Patch source position '{patch_source_position}' not in extraction positions {extraction_store.position_names}"
            )
        vec = extraction_store.mean(
            position=patch_source_position,
            layer_value=layer_val,
            uids=patch_uids,
            style=args.patch_source_style,
            condition_code=args.patch_source_condition,
        )
        if vec is None:
            raise ValueError(
                f"No patch source activations for layer={layer_val}, position={patch_source_position}, "
                f"style={args.patch_source_style}, condition={args.patch_source_condition}"
            )
        patch_cache[k] = vec.float()
        return patch_cache[k]

    rows_path = args.output_dir / "steering_rows.jsonl"
    if rows_path.exists():
        all_rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        print(f"[resume] loaded {len(all_rows)} existing steering rows from {rows_path}")
    else:
        all_rows: List[Dict[str, Any]] = []
    expected_per_alpha = len(jobs)
    t0 = time.time()
    config_counter = 0
    for layer_group in layer_groups:
        for position_mode in position_modes:
            config_counter += 1
            config_id = (
                f"cfg{config_counter:03d}"
                f"_L{'-'.join(str(x) for x in layer_group)}"
                f"_P{position_mode}"
                f"_C{'-'.join(components)}"
                f"_M{args.intervention_mode}"
            )
            target_specs: List[TargetSpec] = []
            for layer_val in layer_group:
                d = None
                p = None
                nuisance_basis = nuisance_basis_by_layer.get(layer_val)
                if args.intervention_mode in {"add", "subtract", "project_out_direction"}:
                    d = direction_for(layer_val)
                elif args.intervention_mode in {"replace_mean", "interpolate_mean"}:
                    p = patch_mean_for(layer_val)
                elif args.intervention_mode == "project_out_subspace":
                    if nuisance_basis is None:
                        raise ValueError("--intervention-mode project_out_subspace requires a basis for every target layer")
                else:
                    raise ValueError(f"Unknown intervention mode: {args.intervention_mode}")
                for component in components:
                    if (layer_val, component) not in component_modules:
                        raise ValueError(f"Component target not found: layer={layer_val}, component={component}")
                    target_specs.append(
                        TargetSpec(
                            layer=layer_val,
                            component=component,
                            direction=d,
                            patch_mean=p,
                            nuisance_basis=nuisance_basis,
                        )
                    )

            print(f"\n=== {config_id} ===")
            for alpha in alphas:
                existing_alpha_rows = [
                    row for row in all_rows
                    if row.get("config_id") == config_id and float(row.get("alpha", 0.0)) == float(alpha)
                ]
                if len(existing_alpha_rows) >= expected_per_alpha:
                    print(f"--- alpha={alpha} [skip existing {len(existing_alpha_rows)} rows] ---")
                    continue
                print(f"--- alpha={alpha} ---")
                alpha_rows: List[Dict[str, Any]] = []
                batches = _dynamic_batches(jobs, max_batch_size=args.batch_size, max_batch_tokens=args.max_batch_tokens)
                for batch_idx, chunk in enumerate(tqdm(batches, desc=f"{config_id}:a={alpha}")):
                    batch_rows = _generate_steered_batch(
                        chunk,
                        model=model,
                        tokenizer=tokenizer,
                        component_modules=component_modules,
                        target_specs=target_specs,
                        alpha=alpha,
                        position_mode=position_mode,
                        apply_phase=args.apply_phase,
                        intervention_mode=args.intervention_mode,
                        norm_scaling=args.norm_scaling,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        greedy=args.greedy,
                        use_cache=not args.no_cache,
                        seed=args.seed + batch_idx,
                        collect_margin_diagnostics=args.collect_margin_diagnostics,
                        token_id_a=token_id_a,
                        token_id_b=token_id_b,
                    )
                    for row in batch_rows:
                        row.update(
                            {
                                "config_id": config_id,
                                "target_layers": list(layer_group),
                                "position_mode": position_mode,
                                "components": list(components),
                                "apply_phase": args.apply_phase,
                                "intervention_mode": args.intervention_mode,
                                "norm_scaling": args.norm_scaling,
                                "direction_kind": args.direction_kind,
                                "direction_position": direction_position,
                                "direction_source": "extraction" if extraction_store is not None else "file",
                                "project_out_subspace_path": str(args.project_out_subspace_path) if args.project_out_subspace_path else None,
                            }
                        )
                    alpha_rows.extend(batch_rows)
                    all_rows = [
                        row for row in all_rows
                        if not (row.get("config_id") == config_id and float(row.get("alpha", 0.0)) == float(alpha))
                    ] + alpha_rows
                    _write_jsonl(rows_path, all_rows)
                wrapped = [_wrap_for_parser(r) for r in alpha_rows]
                markers = discover_model_markers(wrapped)
                parsed_count = sum(1 for r in wrapped if dynamic_parse_row(r, markers)[0] is not None)
                print(f"  Generated {len(alpha_rows)} rows, ~{parsed_count} parseable")

    elapsed = time.time() - t0
    print(f"\nTotal generation: {elapsed:.1f}s for {len(all_rows)} rows")

    result = _parse_and_score(all_rows)

    _write_jsonl(rows_path, result["rows"])

    summary_path = args.output_dir / "steering_summary.json"
    summary_path.write_text(json.dumps(result["summary"], indent=2))

    meta = {
        "model": args.model,
        "direction_path": str(args.direction_path),
        "direction_key": args.direction_key,
        "direction_position": direction_position,
        "direction_kind": args.direction_kind,
        "direction_source": "extraction" if extraction_store is not None else "file",
        "extraction_dir": str(args.extraction_dir) if args.extraction_dir is not None else None,
        "masks_path": str(masks_path),
        "uids_file": str(args.uids_file) if args.uids_file is not None else None,
        "uids_subset": args.uids_subset,
        "direction_uids_subset": direction_subset,
        "direction_uids_file": str(args.direction_uids_file) if args.direction_uids_file is not None else None,
        "patch_uids_subset": patch_subset,
        "patch_uids_file": str(args.patch_uids_file) if args.patch_uids_file is not None else None,
        "n_direction_uids": len(direction_uids),
        "n_patch_uids": len(patch_uids),
        "n_uids": len(eval_uids),
        "conditions": conditions,
        "alphas": alphas,
        "target_layers": target_layers,
        "single_layer_sweep": bool(args.single_layer_sweep),
        "position_modes": position_modes,
        "components": components,
        "apply_phase": args.apply_phase,
        "intervention_mode": args.intervention_mode,
        "norm_scaling": args.norm_scaling,
        "project_out_subspace_path": str(args.project_out_subspace_path) if args.project_out_subspace_path else None,
        "sparse_topk": args.sparse_topk,
        "collect_margin_diagnostics": bool(args.collect_margin_diagnostics),
        "patch_source_style": args.patch_source_style,
        "patch_source_condition": args.patch_source_condition,
        "patch_source_position": patch_source_position,
        "answer_suffix": args.answer_suffix,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "greedy": bool(args.greedy),
        "use_cache": not args.no_cache,
        "no_compile": bool(args.no_compile),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_seconds": round(elapsed, 2),
        "rows_path": str(rows_path),
        "summary_path": str(summary_path),
    }
    (args.output_dir / "steering_meta.json").write_text(json.dumps(meta, indent=2))

    print("\n=== Steering Summary ===")
    print(
        f"{'Config':>26} {'Alpha':>8} {'Condition':>10} {'N':>5} {'Parsed':>6} "
        f"{'Accuracy':>8} {'Wrong%':>8} {'Flip%':>8} {'MeanLogitW':>11}"
    )
    for s in result["summary"]:
        acc = f"{s['accuracy']:.3f}" if s["accuracy"] is not None else "  n/a"
        wr = f"{s['wrong_rate']:.3f}" if s["wrong_rate"] is not None else "  n/a"
        fr = f"{s['flip_rate']:.3f}" if s["flip_rate"] is not None else "  n/a"
        lw = f"{s['mean_fc_logit_wrong']:.4f}" if s["mean_fc_logit_wrong"] is not None else "     n/a"
        print(
            f"{s['config_id'][:26]:>26} {s['alpha']:>8.1f} {s['condition']:>10} {s['total']:>5} "
            f"{s['parsed']:>6} {acc:>8} {wr:>8} {fr:>8} {lw:>11}"
        )

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
