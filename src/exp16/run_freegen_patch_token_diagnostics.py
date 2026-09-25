from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.nn import functional as F

from src.exp16.dynamic_parser import discover_model_markers, dynamic_parse_row
from src.exp16.run_dissociation_test import _set_torch_seed
from src.exp16.run_steering_test import (
    ENDORSEMENT_STYLE,
    TargetSpec,
    _build_condition_map,
    _build_jobs,
    _build_positions_for_batch,
    _generate_steered_batch,
    _load_extraction_store,
    _load_masks,
    _load_uids_from_masks,
)
from src.exp7.dataset_mc import load_mc_dataset
from src.mechanism.hooks import _extract_hidden, get_component_modules, get_transformer_layers
from src.models.llama_loader import load_model_and_tokenizer


DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_DATASET = Path("data/exp7_mc_dataset.jsonl")
DEFAULT_EXTRACT_DIR = Path("results/authority/mechanism/gpt_oss_authority_activations")
DEFAULT_MASKS = Path("results/authority/mechanism/gpt_oss_authority_activations/label_masks.json")
DEFAULT_OUTPUT_DIR = Path("results/authority/mechanism/freegen_patch_token_diagnostics")


class FullLayerCapture:
    def __init__(self, model, target_layers: Sequence[int], save_dtype: torch.dtype = torch.float16):
        self.layers = get_transformer_layers(model)
        self.target_layers = list(target_layers)
        self.save_dtype = save_dtype
        self._captured: Dict[int, torch.Tensor] = {}
        self._handles = []

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = _extract_hidden(output)
            self._captured[layer_idx] = hidden.detach().to(dtype=self.save_dtype, device="cpu")

        return hook

    def __enter__(self):
        for layer_idx in self.target_layers:
            handle = self.layers[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)
        return self

    def __exit__(self, *_exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear(self):
        self._captured.clear()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cheap free-generation token diagnostics for authority patching.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--extraction-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    p.add_argument("--masks-path", type=Path, default=DEFAULT_MASKS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--condition-code", type=str, default="W1_note")
    p.add_argument("--uids-subset", choices=("w1", "c1", "all"), default="w1")
    p.add_argument("--max-items", type=int, default=64)
    p.add_argument("--target-layer", type=int, default=16)
    p.add_argument("--capture-layers", type=str, default="16,18,20")
    p.add_argument("--component", choices=("block_output", "attention_output", "mlp_output"), default="block_output")
    p.add_argument("--position-mode", type=str, default="note_end")
    p.add_argument("--intervention-mode", choices=("replace_mean", "interpolate_mean"), default="interpolate_mean")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--patch-source-style", type=str, default="authoritative_verified")
    p.add_argument("--patch-source-condition", type=str, default="N0_note")
    p.add_argument("--patch-source-position", type=str, default="endorsement_end")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plain-prompt", action="store_true")
    return p.parse_args()


def _parse_csv_int(raw: str) -> List[int]:
    out: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise ValueError("Expected at least one capture layer.")
    return out


def _build_target_specs(
    *,
    store,
    patch_uids: Sequence[str],
    target_layer: int,
    component: str,
    patch_source_style: str,
    patch_source_condition: str,
    patch_source_position: str,
) -> List[TargetSpec]:
    patch_mean = store.mean(
        position=patch_source_position,
        layer_value=target_layer,
        uids=patch_uids,
        style=patch_source_style,
        condition_code=patch_source_condition,
    )
    if patch_mean is None:
        raise ValueError(
            f"No patch source activations for layer={target_layer}, position={patch_source_position}, "
            f"style={patch_source_style}, condition={patch_source_condition}"
        )
    return [
        TargetSpec(
            layer=target_layer,
            component=component,
            direction=None,
            patch_mean=patch_mean.float(),
            nuisance_basis=None,
        )
    ]


def _shifted_positions(
    *,
    jobs: Sequence[Dict[str, Any]],
    tokenizer,
    attention_mask: torch.Tensor,
    input_width: int,
) -> Dict[str, List[List[int]]]:
    out: Dict[str, List[List[int]]] = {}
    for mode in ("note_end", "answer_position", "endorsement_span"):
        out[mode] = _build_positions_for_batch(
            jobs=jobs,
            attention_mask=attention_mask,
            input_width=input_width,
            tokenizer=tokenizer,
            position_mode=mode,
        )
    return out


def _collect_prompt_views(
    *,
    jobs: Sequence[Dict[str, Any]],
    model,
    tokenizer,
    target_specs: Sequence[TargetSpec],
    capture_layers: Sequence[int],
    batch_size: int,
    alpha: float,
    intervention_mode: str,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    component_modules = get_component_modules(model)
    rows: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for start in range(0, len(jobs), batch_size):
        chunk = list(jobs[start : start + batch_size])
        prompts = [job["model_prompt"] for job in chunk]
        model_inputs = tokenizer(
            prompts,
            padding=True,
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)
        attention_mask = model_inputs["attention_mask"]
        input_width = int(model_inputs["input_ids"].shape[1])
        positions = _shifted_positions(
            jobs=chunk,
            tokenizer=tokenizer,
            attention_mask=attention_mask,
            input_width=input_width,
        )

        handles = []
        for spec in target_specs:
            module = component_modules[(spec.layer, spec.component)]
            handles.append(
                module.register_forward_hook(
                    _make_prompt_only_hook(
                        spec=spec,
                        alpha=alpha,
                        intervention_mode=intervention_mode,
                        prompt_positions_by_row=positions["note_end"],
                    )
                )
            )

        capture = FullLayerCapture(model, target_layers=capture_layers, save_dtype=torch.float16)
        with capture:
            with torch.inference_mode():
                _ = model(**model_inputs, use_cache=False)

        for handle in handles:
            handle.remove()

        for row_idx, job in enumerate(chunk):
            row_views: Dict[str, Dict[str, List[float]]] = {}
            span_positions = positions["endorsement_span"][row_idx]
            note_end_positions = positions["note_end"][row_idx]
            answer_positions = positions["answer_position"][row_idx]
            for layer in capture_layers:
                hidden = capture._captured[layer][row_idx].float()
                layer_views: Dict[str, List[float]] = {}
                if note_end_positions:
                    layer_views["note_end"] = hidden[note_end_positions[0]].tolist()
                if answer_positions:
                    layer_views["answer_position"] = hidden[answer_positions[0]].tolist()
                if span_positions:
                    layer_views["endorsement_mean"] = hidden[span_positions, :].mean(dim=0).tolist()
                row_views[str(layer)] = layer_views
            rows[job["uid"]] = row_views
        capture.clear()

    return rows


def _make_prompt_only_hook(
    *,
    spec: TargetSpec,
    alpha: float,
    intervention_mode: str,
    prompt_positions_by_row: Sequence[Sequence[int]],
):
    def hook(_module, _inputs, output):
        hidden = _extract_hidden(output)
        updated = hidden.clone()
        patch_mean = spec.patch_mean.to(hidden.device, dtype=hidden.dtype) if spec.patch_mean is not None else None

        for row_idx, row_positions in enumerate(prompt_positions_by_row):
            if row_idx >= hidden.shape[0]:
                break
            valid_positions = [int(pos) for pos in row_positions if 0 <= int(pos) < hidden.shape[1]]
            if not valid_positions or patch_mean is None:
                continue
            for pos in valid_positions:
                base = hidden[row_idx, pos, :]
                if intervention_mode == "replace_mean":
                    updated[row_idx, pos, :] = patch_mean
                elif intervention_mode == "interpolate_mean":
                    updated[row_idx, pos, :] = base + alpha * (patch_mean - base)
                else:
                    raise ValueError(f"Unsupported intervention mode: {intervention_mode}")
        if torch.is_tensor(output):
            return updated
        return (updated, *output[1:])

    return hook


def _parse_generation_rows(rows: List[Dict[str, Any]]) -> None:
    wrapped = [
        {
            **row,
            "generation": {"raw_text": row["raw_text"]},
            "correct_answer": row["correct_text"],
            "wrong_answer": row["wrong_text"],
        }
        for row in rows
    ]
    markers = discover_model_markers(wrapped)
    for row, wrapped_row in zip(rows, wrapped, strict=True):
        label, meta = dynamic_parse_row(wrapped_row, markers)
        row["parsed_label"] = label
        row["parse_meta"] = meta
        row["is_correct"] = None if label is None else label == row["correct_label"]
        row["chose_wrong"] = None if label is None else label == row["wrong_label"]


def _answer_text_for_label(job: Dict[str, Any], label: str) -> str:
    return job["correct_text"] if label == job["correct_label"] else job["wrong_text"]


def _continuation_ids(tokenizer, prompt_text: str, answer_text: str) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_enc = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
    full_text = f"{prompt_text} {answer_text}"
    full_enc = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
    prompt_ids = prompt_enc["input_ids"][0]
    full_ids = full_enc["input_ids"][0]
    if full_ids.shape[0] <= prompt_ids.shape[0]:
        raise ValueError("Continuation tokenization failed to add answer tokens.")
    return prompt_ids, full_ids


def _score_candidate(
    *,
    model,
    tokenizer,
    job: Dict[str, Any],
    target_specs: Sequence[TargetSpec],
    alpha: float,
    intervention_mode: str,
    answer_text: str,
) -> Dict[str, float]:
    prompt_ids, full_ids = _continuation_ids(tokenizer, job["model_prompt"], answer_text)
    continuation_ids = full_ids[prompt_ids.shape[0] :]
    inputs = {
        "input_ids": full_ids.unsqueeze(0).to(model.device),
        "attention_mask": torch.ones((1, full_ids.shape[0]), dtype=torch.long, device=model.device),
    }

    component_modules = get_component_modules(model)
    prompt_positions = [[int(job["positions_unpadded"]["endorsement_end"])]]
    handles = []
    for spec in target_specs:
        module = component_modules[(spec.layer, spec.component)]
        handles.append(
            module.register_forward_hook(
                _make_prompt_only_hook(
                    spec=spec,
                    alpha=alpha,
                    intervention_mode=intervention_mode,
                    prompt_positions_by_row=prompt_positions,
                )
            )
        )

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)

    for handle in handles:
        handle.remove()

    logits = outputs.logits[0]
    start = int(prompt_ids.shape[0] - 1)
    stop = int(full_ids.shape[0] - 1)
    candidate_logits = logits[start:stop]
    log_probs = F.log_softmax(candidate_logits, dim=-1)
    token_indices = torch.arange(continuation_ids.shape[0], device=log_probs.device)
    token_logprobs = log_probs[token_indices, continuation_ids.to(log_probs.device)]
    return {
        "total_logprob": float(token_logprobs.sum().item()),
        "mean_logprob": float(token_logprobs.mean().item()),
        "first_token_logprob": float(token_logprobs[0].item()),
        "n_tokens": int(continuation_ids.shape[0]),
    }


def _compute_teacher_forced_metrics(
    *,
    jobs: Sequence[Dict[str, Any]],
    model,
    tokenizer,
    target_specs: Sequence[TargetSpec],
    alpha: float,
    intervention_mode: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for job in jobs:
        uid = job["uid"]
        out[uid] = {"baseline": {}, "patched": {}}
        for phase_name, phase_specs, phase_alpha in (
            ("baseline", [], 0.0),
            ("patched", list(target_specs), alpha),
        ):
            correct_scores = _score_candidate(
                model=model,
                tokenizer=tokenizer,
                job=job,
                target_specs=phase_specs,
                alpha=phase_alpha,
                intervention_mode=intervention_mode,
                answer_text=job["correct_text"],
            )
            wrong_scores = _score_candidate(
                model=model,
                tokenizer=tokenizer,
                job=job,
                target_specs=phase_specs,
                alpha=phase_alpha,
                intervention_mode=intervention_mode,
                answer_text=job["wrong_text"],
            )
            out[uid][phase_name] = {
                "correct_total_logprob": correct_scores["total_logprob"],
                "wrong_total_logprob": wrong_scores["total_logprob"],
                "correct_first_token_logprob": correct_scores["first_token_logprob"],
                "wrong_first_token_logprob": wrong_scores["first_token_logprob"],
                "total_margin_correct_minus_wrong": (
                    correct_scores["total_logprob"] - wrong_scores["total_logprob"]
                ),
                "first_token_margin_correct_minus_wrong": (
                    correct_scores["first_token_logprob"] - wrong_scores["first_token_logprob"]
                ),
            }
    return out


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _summarize(
    *,
    baseline_rows: Sequence[Dict[str, Any]],
    patched_rows: Sequence[Dict[str, Any]],
    teacher_forced: Dict[str, Dict[str, Dict[str, float]]],
    baseline_views: Dict[str, Dict[str, Dict[str, List[float]]]],
    patched_views: Dict[str, Dict[str, Dict[str, List[float]]]],
    capture_layers: Sequence[int],
) -> Dict[str, Any]:
    baseline_by_uid = {row["uid"]: row for row in baseline_rows}
    patched_by_uid = {row["uid"]: row for row in patched_rows}
    ordered_uids = [row["uid"] for row in baseline_rows if row["uid"] in patched_by_uid]

    gen_pairs = []
    margin_deltas = []
    first_margin_deltas = []
    margin_sign_flips = 0
    prompt_delta_by_layer: Dict[str, Dict[str, List[float]]] = {
        str(layer): {"note_end": [], "answer_position": [], "endorsement_mean": []}
        for layer in capture_layers
    }

    rows = []
    for uid in ordered_uids:
        b = baseline_by_uid[uid]
        p = patched_by_uid[uid]
        tf_base = teacher_forced[uid]["baseline"]
        tf_patch = teacher_forced[uid]["patched"]
        margin_delta = tf_patch["total_margin_correct_minus_wrong"] - tf_base["total_margin_correct_minus_wrong"]
        first_margin_delta = (
            tf_patch["first_token_margin_correct_minus_wrong"] - tf_base["first_token_margin_correct_minus_wrong"]
        )
        margin_deltas.append(margin_delta)
        first_margin_deltas.append(first_margin_delta)
        if (
            tf_base["total_margin_correct_minus_wrong"] > 0
            and tf_patch["total_margin_correct_minus_wrong"] < 0
        ):
            margin_sign_flips += 1

        hidden_delta_payload: Dict[str, Dict[str, float]] = {}
        for layer in capture_layers:
            layer_key = str(layer)
            hidden_delta_payload[layer_key] = {}
            for pos_name in ("note_end", "answer_position", "endorsement_mean"):
                base_vec = baseline_views.get(uid, {}).get(layer_key, {}).get(pos_name)
                patch_vec = patched_views.get(uid, {}).get(layer_key, {}).get(pos_name)
                if base_vec is None or patch_vec is None:
                    continue
                delta = torch.tensor(patch_vec) - torch.tensor(base_vec)
                norm_val = float(delta.norm().item())
                hidden_delta_payload[layer_key][pos_name] = norm_val
                prompt_delta_by_layer[layer_key][pos_name].append(norm_val)

        rows.append(
            {
                "uid": uid,
                "baseline_parsed_label": b.get("parsed_label"),
                "patched_parsed_label": p.get("parsed_label"),
                "baseline_is_correct": b.get("is_correct"),
                "patched_is_correct": p.get("is_correct"),
                "baseline_chose_wrong": b.get("chose_wrong"),
                "patched_chose_wrong": p.get("chose_wrong"),
                "baseline_raw_text": b.get("raw_text"),
                "patched_raw_text": p.get("raw_text"),
                "teacher_forced_baseline": tf_base,
                "teacher_forced_patched": tf_patch,
                "teacher_forced_total_margin_delta": margin_delta,
                "teacher_forced_first_token_margin_delta": first_margin_delta,
                "prompt_hidden_delta_norms": hidden_delta_payload,
            }
        )
        gen_pairs.append((b, p))

    def _rate(rows_in: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [row[key] for row in rows_in if row.get(key) is not None]
        if not vals:
            return None
        return float(sum(1 for value in vals if value) / len(vals))

    generation = {
        "n_rows": len(ordered_uids),
        "baseline_correct_rate": _rate(baseline_rows, "is_correct"),
        "patched_correct_rate": _rate(patched_rows, "is_correct"),
        "baseline_wrong_rate": _rate(baseline_rows, "chose_wrong"),
        "patched_wrong_rate": _rate(patched_rows, "chose_wrong"),
        "correct_to_wrong_flips": sum(
            1
            for b, p in gen_pairs
            if b.get("is_correct") is True and p.get("chose_wrong") is True
        ),
        "wrong_to_correct_flips": sum(
            1
            for b, p in gen_pairs
            if b.get("chose_wrong") is True and p.get("is_correct") is True
        ),
    }

    prompt_summary = {
        str(layer): {
            pos_name: _mean(values)
            for pos_name, values in pos_payload.items()
        }
        for layer, pos_payload in prompt_delta_by_layer.items()
    }

    teacher_summary = {
        "mean_total_margin_delta": _mean(margin_deltas),
        "mean_first_token_margin_delta": _mean(first_margin_deltas),
        "margin_sign_flips_positive_to_negative": margin_sign_flips,
    }

    return {
        "generation": generation,
        "teacher_forced": teacher_summary,
        "prompt_hidden_delta_norm_mean": prompt_summary,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capture_layers = _parse_csv_int(args.capture_layers)
    t0 = time.time()
    _set_torch_seed(args.seed)

    print(f"Loading model {args.model}...")
    model, tokenizer = load_model_and_tokenizer(model_name=args.model, device="auto", dtype="auto")
    model.eval()

    examples = load_mc_dataset(args.dataset_path)
    examples_by_uid = {ex.uid: ex for ex in examples}
    masks = _load_masks(args.masks_path)
    selected_uids = _load_uids_from_masks(masks, args.uids_subset)
    if args.max_items > 0:
        selected_uids = selected_uids[: args.max_items]
    if not selected_uids:
        raise SystemExit("No UIDs selected.")

    condition_map = _build_condition_map([args.condition_code])
    jobs = _build_jobs(
        examples_by_uid=examples_by_uid,
        uids=selected_uids,
        condition_map=condition_map,
        conditions=[args.condition_code],
        tokenizer=tokenizer,
        plain_prompt=args.plain_prompt,
    )
    if not jobs:
        raise SystemExit("No jobs built.")

    extraction_store = _load_extraction_store(args.extraction_dir)
    target_specs = _build_target_specs(
        store=extraction_store,
        patch_uids=selected_uids,
        target_layer=args.target_layer,
        component=args.component,
        patch_source_style=args.patch_source_style,
        patch_source_condition=args.patch_source_condition,
        patch_source_position=args.patch_source_position,
    )

    baseline_prompt_views = _collect_prompt_views(
        jobs=jobs,
        model=model,
        tokenizer=tokenizer,
        target_specs=[],
        capture_layers=capture_layers,
        batch_size=args.batch_size,
        alpha=0.0,
        intervention_mode=args.intervention_mode,
    )
    patched_prompt_views = _collect_prompt_views(
        jobs=jobs,
        model=model,
        tokenizer=tokenizer,
        target_specs=target_specs,
        capture_layers=capture_layers,
        batch_size=args.batch_size,
        alpha=args.alpha,
        intervention_mode=args.intervention_mode,
    )

    baseline_rows: List[Dict[str, Any]] = []
    patched_rows: List[Dict[str, Any]] = []
    component_modules = get_component_modules(model)

    for start in range(0, len(jobs), args.batch_size):
        chunk = jobs[start : start + args.batch_size]
        baseline_rows.extend(
            _generate_steered_batch(
                chunk,
                model=model,
                tokenizer=tokenizer,
                component_modules=component_modules,
                target_specs=[],
                alpha=0.0,
                position_mode=args.position_mode,
                apply_phase="prompt",
                intervention_mode=args.intervention_mode,
                norm_scaling="none",
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                greedy=True,
                use_cache=True,
                seed=args.seed,
                collect_margin_diagnostics=False,
                token_id_a=None,
                token_id_b=None,
            )
        )
        patched_rows.extend(
            _generate_steered_batch(
                chunk,
                model=model,
                tokenizer=tokenizer,
                component_modules=component_modules,
                target_specs=target_specs,
                alpha=args.alpha,
                position_mode=args.position_mode,
                apply_phase="prompt",
                intervention_mode=args.intervention_mode,
                norm_scaling="none",
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                greedy=True,
                use_cache=True,
                seed=args.seed,
                collect_margin_diagnostics=False,
                token_id_a=None,
                token_id_b=None,
            )
        )

    _parse_generation_rows(baseline_rows)
    _parse_generation_rows(patched_rows)

    teacher_forced = _compute_teacher_forced_metrics(
        jobs=jobs,
        model=model,
        tokenizer=tokenizer,
        target_specs=target_specs,
        alpha=args.alpha,
        intervention_mode=args.intervention_mode,
    )

    summary = _summarize(
        baseline_rows=baseline_rows,
        patched_rows=patched_rows,
        teacher_forced=teacher_forced,
        baseline_views=baseline_prompt_views,
        patched_views=patched_prompt_views,
        capture_layers=capture_layers,
    )
    summary["config"] = {
        "model": args.model,
        "condition_code": args.condition_code,
        "uids_subset": args.uids_subset,
        "max_items": args.max_items,
        "target_layer": args.target_layer,
        "capture_layers": capture_layers,
        "component": args.component,
        "position_mode": args.position_mode,
        "intervention_mode": args.intervention_mode,
        "alpha": args.alpha,
        "patch_source_style": args.patch_source_style,
        "patch_source_condition": args.patch_source_condition,
        "patch_source_position": args.patch_source_position,
        "max_new_tokens": args.max_new_tokens,
        "endorsement_style": ENDORSEMENT_STYLE,
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    rows_path = args.output_dir / "diagnostic_rows.jsonl"
    with rows_path.open("w") as f:
        for row in summary["rows"]:
            f.write(json.dumps(row) + "\n")
    summary_path = args.output_dir / "diagnostic_summary.json"
    summary_path.write_text(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"saved_rows={rows_path}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
