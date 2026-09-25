from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from joblib import Parallel, delayed
from tqdm import tqdm

from src.lexical_controls.dataset_mc import MCExample, load_mc_dataset, save_mc_dataset, verify_mc_dataset
from src.endorsement.conditions import Exp10Condition, generate_all_conditions
from src.authority_steering.dynamic_parser import discover_model_markers, dynamic_parse_row
from src.authority_steering.run_dissociation_test import DEFAULT_INSTRUCTION, _format_chat_prompt, _format_exp16_prompt
from src.authority_steering.run_steering_test import (
    DEFAULT_DIRECTION,
    DEFAULT_MASKS,
    ExtractionStore,
    TargetSpec,
    _compute_prompt_positions,
    _direction_from_store,
    _generate_steered_batch,
    _load_direction_file,
    _load_extraction_store,
    _load_masks,
    _load_uids_from_masks,
    _sparsify_vector,
)
from src.mechanism.hooks import get_component_modules, get_transformer_layers
from src.mechanism.hooks_v2 import setup_h100_optimizations, try_compile_model
from src.models.llama_loader import load_model_and_tokenizer


DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_PIQA_DATASET = Path("data/piqa_mc_validation.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/authority/mechanism/piqa_interventions")
DEFAULT_EXTRACT_DIR = Path("results/authority/mechanism/gpt_oss_authority_activations")


@dataclass(frozen=True)
class RunConfig:
    config_id: str
    mode: str
    alpha: float
    target_layers: Tuple[int, ...]
    components: Tuple[str, ...]
    position_mode: str
    apply_phase: str
    direction_kind: Optional[str] = None
    sparse_topk: int = 0
    norm_scaling: str = "none"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PIQA transfer interventions with the current exp16 patching stack.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--mc-dataset-path", type=Path, default=DEFAULT_PIQA_DATASET)
    p.add_argument("--rebuild-dataset-from-hf", action="store_true")
    p.add_argument("--hf-split", type=str, default="validation")
    p.add_argument("--save-rebuilt-dataset", type=Path, default=None)
    p.add_argument("--uids-file", type=Path, default=None)
    p.add_argument("--eval-uids-subset", choices=("all", "w1", "c1"), default="all")
    p.add_argument("--max-samples", type=int, default=0, help="0 = all selected samples.")
    p.add_argument("--conditions", type=str, default="W1_note")
    p.add_argument(
        "--answer-suffix",
        type=str,
        default="What is the answer? Reply with exactly one token: A or B.",
    )
    p.add_argument("--instruction-text", type=str, default=DEFAULT_INSTRUCTION)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prep-n-jobs", type=int, default=-1, help="CPU workers for data/prompt preparation.")
    p.add_argument("--parse-n-jobs", type=int, default=-1, help="CPU workers for parser scoring.")

    p.add_argument("--direction-path", type=Path, default=DEFAULT_DIRECTION)
    p.add_argument("--direction-key", type=str, default="vector")
    p.add_argument("--extraction-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    p.add_argument("--masks-path", type=Path, default=DEFAULT_MASKS)
    p.add_argument("--direction-position", type=str, default="endorsement_end")
    p.add_argument("--patch-source-position", type=str, default="endorsement_end")
    p.add_argument("--patch-source-style", type=str, default="authoritative_verified")
    p.add_argument("--patch-source-condition", type=str, default="N0_note")
    p.add_argument("--direction-uids-subset", choices=("w1", "c1", "all"), default="w1")
    p.add_argument("--direction-uids-file", type=Path, default=None,
                   help="Optional newline-delimited UID list for direction extraction (overrides --direction-uids-subset).")
    p.add_argument("--patch-uids-subset", choices=("w1", "c1", "all"), default="w1")
    p.add_argument("--patch-uids-file", type=Path, default=None,
                   help="Optional newline-delimited UID list for patch means (overrides --patch-uids-subset).")

    p.add_argument("--target-layers", type=str, default="18")
    p.add_argument("--single-layer-sweep", action="store_true")
    p.add_argument("--components", type=str, default="block_output")
    p.add_argument("--position-mode", type=str, default="endorsement_span")
    p.add_argument("--apply-phase", choices=("prompt", "decode", "both"), default="prompt")
    p.add_argument("--norm-scaling", choices=("none", "resid_norm", "resid_std"), default="none")
    p.add_argument("--additive-modes", type=str, default="add,subtract")
    p.add_argument(
        "--direction-kinds",
        type=str,
        default="shared_within_label,authority_given_wrong,pooled_authority,endorsement_presence",
    )
    p.add_argument("--additive-alphas", type=str, default="-10,-5,0,5,10")
    p.add_argument("--sparse-topk-values", type=str, default="0,64")
    p.add_argument("--patch-modes", type=str, default="interpolate_mean")
    p.add_argument("--patch-alphas", type=str, default="0,0.3,0.5,0.7,1.0")
    p.add_argument("--include-baseline", action="store_true")
    p.add_argument("--max-configs", type=int, default=0)

    p.add_argument("--batch-size", type=int, default=8, help="Upper bound; final batches are token-dynamic.")
    p.add_argument("--max-batch-tokens", type=int, default=16384, help="Approx prompt token budget per batch.")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--plain-prompt", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_floats(raw: str) -> List[float]:
    return [float(x) for x in _parse_csv(raw)]


def _parse_ints(raw: str) -> List[int]:
    return [int(x) for x in _parse_csv(raw)]


def _condition_map(codes: Sequence[str], instruction_text: str) -> Dict[str, Exp10Condition]:
    all_conditions = generate_all_conditions(
        tags=["Expert", "Note", "User", "Someone online"],
        instruction_text=instruction_text,
    )
    mapping = {c.code: c for c in all_conditions}
    missing = [code for code in codes if code not in mapping]
    if missing:
        raise ValueError(f"Unknown condition codes: {missing}")
    return {code: mapping[code] for code in codes}


def _build_piqa_example(idx: int, row: Dict[str, Any], seed: int) -> MCExample:
    correct_answer = row["sol1"] if int(row["label"]) == 0 else row["sol2"]
    wrong_answer = row["sol2"] if int(row["label"]) == 0 else row["sol1"]
    correct_is_a = random.Random(seed + idx).choice([True, False])
    if correct_is_a:
        correct_label, wrong_label = "A", "B"
        option_a, option_b = correct_answer, wrong_answer
    else:
        correct_label, wrong_label = "B", "A"
        option_a, option_b = wrong_answer, correct_answer
    return MCExample(
        uid=f"piqa::{idx}",
        question=str(row["goal"]),
        correct_answer=correct_answer,
        wrong_answer=wrong_answer,
        correct_label=correct_label,
        wrong_label=wrong_label,
        option_a=option_a,
        option_b=option_b,
        endorsed_label=wrong_label,
        metadata={"dataset": "piqa", "source_split": row.get("split", "unknown"), "source_index": idx},
    )


def _load_or_build_piqa_dataset(args: argparse.Namespace) -> List[MCExample]:
    if args.rebuild_dataset_from_hf:
        ds = load_dataset("lighteval/piqa", split=args.hf_split)
        rows = [dict(r) for r in ds]
        n_jobs = args.prep_n_jobs if args.prep_n_jobs != 0 else 1
        if n_jobs == 1:
            examples = [_build_piqa_example(i, row, args.seed) for i, row in enumerate(rows)]
        else:
            examples = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_build_piqa_example)(i, row, args.seed) for i, row in enumerate(rows)
            )
        out_path = args.save_rebuilt_dataset or args.mc_dataset_path
        if out_path is not None:
            save_mc_dataset(examples, out_path)
            print(json.dumps({"rebuilt_dataset": str(out_path), "verification": verify_mc_dataset(examples)}, indent=2))
        return examples
    return load_mc_dataset(args.mc_dataset_path)


def _parse_layers(raw: str, n_layers: int) -> List[int]:
    layers = _parse_ints(raw)
    uniq: List[int] = []
    seen = set()
    for layer in layers:
        if layer in seen:
            continue
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Layer {layer} out of range for model with {n_layers} layers.")
        uniq.append(layer)
        seen.add(layer)
    if not uniq:
        raise ValueError("No valid target layers.")
    return uniq


def _select_eval_uids(
    args: argparse.Namespace,
    examples_by_uid: Dict[str, MCExample],
    masks: Optional[Dict[str, Any]],
) -> List[str]:
    if args.uids_file is not None:
        uids = [line.strip() for line in args.uids_file.read_text().splitlines() if line.strip()]
        uids = [uid for uid in uids if uid in examples_by_uid]
    elif args.eval_uids_subset == "all":
        uids = sorted(examples_by_uid.keys())
    else:
        if masks is None:
            raise ValueError("--eval-uids-subset w1/c1 requires masks.")
        subset = "w1" if args.eval_uids_subset == "w1" else "c1"
        uids = [uid for uid in _load_uids_from_masks(masks, subset) if uid in examples_by_uid]
    if args.max_samples > 0:
        uids = uids[: args.max_samples]
    if not uids:
        raise ValueError("No PIQA UIDs selected.")
    return uids


def _load_uids_file(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prompt_skeleton(uid: str, condition_code: str, ex: MCExample, cond: Exp10Condition, answer_suffix: str) -> Dict[str, Any]:
    base = _format_exp16_prompt(ex=ex, condition=cond, endorsement_style="authoritative_verified").rstrip()
    prompt_text = f"{base}\n{answer_suffix}".strip()
    return {
        "uid": uid,
        "condition_code": condition_code,
        "prompt_text": prompt_text,
        "correct_label": ex.correct_label,
        "wrong_label": ex.wrong_label,
        "correct_text": ex.correct_answer,
        "wrong_text": ex.wrong_answer,
        "question": ex.question,
    }


def _build_jobs(
    *,
    uids: Sequence[str],
    examples_by_uid: Dict[str, MCExample],
    condition_map: Dict[str, Exp10Condition],
    tokenizer,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    skeleton_inputs: List[Tuple[str, str, MCExample, Exp10Condition]] = []
    for uid in uids:
        ex = examples_by_uid[uid]
        for condition_code, cond in condition_map.items():
            skeleton_inputs.append((uid, condition_code, ex, cond))

    n_jobs = args.prep_n_jobs if args.prep_n_jobs != 0 else 1
    if n_jobs == 1:
        skeletons = [
            _prompt_skeleton(uid, code, ex, cond, args.answer_suffix)
            for uid, code, ex, cond in skeleton_inputs
        ]
    else:
        skeletons = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_prompt_skeleton)(uid, code, ex, cond, args.answer_suffix)
            for uid, code, ex, cond in skeleton_inputs
        )

    jobs: List[Dict[str, Any]] = []
    for row in skeletons:
        model_prompt = _format_chat_prompt(tokenizer, row["prompt_text"], args.plain_prompt)
        positions = _compute_prompt_positions(
            tokenizer=tokenizer,
            model_prompt=model_prompt,
            prompt_text=row["prompt_text"],
            condition_code=row["condition_code"],
            correct_label=row["correct_label"],
            wrong_label=row["wrong_label"],
        )
        row["model_prompt"] = model_prompt
        row["positions_unpadded"] = positions
        row["prompt_token_count"] = len(tokenizer(model_prompt, add_special_tokens=False).input_ids)
        jobs.append(row)
    jobs.sort(key=lambda x: x["prompt_token_count"])
    return jobs


def _dynamic_batches(
    jobs: Sequence[Dict[str, Any]],
    *,
    max_batch_size: int,
    max_batch_tokens: int,
) -> List[List[Dict[str, Any]]]:
    out: List[List[Dict[str, Any]]] = []
    batch: List[Dict[str, Any]] = []
    max_len = 0
    for job in jobs:
        n_tok = int(job["prompt_token_count"])
        if not batch:
            batch = [job]
            max_len = n_tok
            continue
        cand_max = max(max_len, n_tok)
        cand_size = len(batch) + 1
        if cand_size > max_batch_size or (cand_max * cand_size > max_batch_tokens):
            out.append(batch)
            batch = [job]
            max_len = n_tok
        else:
            batch.append(job)
            max_len = cand_max
    if batch:
        out.append(batch)
    return out


def _build_run_configs(
    *,
    include_baseline: bool,
    additive_modes: Sequence[str],
    direction_kinds: Sequence[str],
    additive_alphas: Sequence[float],
    sparse_topk_values: Sequence[int],
    patch_modes: Sequence[str],
    patch_alphas: Sequence[float],
    layer_groups: Sequence[Sequence[int]],
    components: Sequence[str],
    position_mode: str,
    apply_phase: str,
    norm_scaling: str,
) -> List[RunConfig]:
    out: List[RunConfig] = []
    if include_baseline:
        for layer_group in layer_groups:
            out.append(
                RunConfig(
                    config_id=f"baseline_L{'-'.join(str(x) for x in layer_group)}",
                    mode="none",
                    alpha=0.0,
                    target_layers=tuple(layer_group),
                    components=tuple(components),
                    position_mode=position_mode,
                    apply_phase=apply_phase,
                    norm_scaling=norm_scaling,
                )
            )

    for layer_group in layer_groups:
        layer_tag = "-".join(str(x) for x in layer_group)
        for mode in additive_modes:
            for kind in direction_kinds:
                for topk in sparse_topk_values:
                    for alpha in additive_alphas:
                        out.append(
                            RunConfig(
                                config_id=f"{mode}_{kind}_k{topk}_a{alpha:g}_L{layer_tag}",
                                mode=mode,
                                alpha=float(alpha),
                                target_layers=tuple(layer_group),
                                components=tuple(components),
                                position_mode=position_mode,
                                apply_phase=apply_phase,
                                direction_kind=kind,
                                sparse_topk=int(topk),
                                norm_scaling=norm_scaling,
                            )
                        )
        for mode in patch_modes:
            for alpha in patch_alphas:
                out.append(
                    RunConfig(
                        config_id=f"{mode}_a{alpha:g}_L{layer_tag}",
                        mode=mode,
                        alpha=float(alpha),
                        target_layers=tuple(layer_group),
                        components=tuple(components),
                        position_mode=position_mode,
                        apply_phase=apply_phase,
                        norm_scaling=norm_scaling,
                    )
                )
    return out


def _wrap_for_parser(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **row,
        "generation": {"raw_text": row["raw_text"]},
        "correct_answer": row["correct_text"],
        "wrong_answer": row["wrong_text"],
    }


def _parse_rows(rows: List[Dict[str, Any]], parse_n_jobs: int) -> None:
    wrapped = [_wrap_for_parser(r) for r in rows]
    markers = discover_model_markers(wrapped)

    def _parse_one(w: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        return dynamic_parse_row(w, markers)

    if parse_n_jobs == 1:
        parsed = [_parse_one(w) for w in wrapped]
    else:
        parsed = Parallel(n_jobs=parse_n_jobs, prefer="threads")(delayed(_parse_one)(w) for w in wrapped)

    for row, (label, meta) in zip(rows, parsed):
        row["parsed_label"] = label
        row["parse_meta"] = meta
        row["is_correct"] = (label == row["correct_label"]) if label is not None else None
        row["chose_wrong"] = (label == row["wrong_label"]) if label is not None else None


def _summarize_config(rows: List[Dict[str, Any]], cfg: RunConfig) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["condition_code"], []).append(row)

    summary: List[Dict[str, Any]] = []
    for cond in sorted(grouped):
        group = grouped[cond]
        parsed = [r for r in group if r["parsed_label"] is not None]
        correct = [r for r in parsed if r["is_correct"]]
        wrong = [r for r in parsed if r["chose_wrong"]]
        summary.append(
            {
                "config_id": cfg.config_id,
                "condition_code": cond,
                "n": len(group),
                "parsed": len(parsed),
                "parse_rate": len(parsed) / max(len(group), 1),
                "accuracy_parsed": len(correct) / max(len(parsed), 1) if parsed else None,
                "wrong_rate_parsed": len(wrong) / max(len(parsed), 1) if parsed else None,
                "flip_rate_parsed": (len(wrong) / max(len(parsed), 1)) if (parsed and cond == "W1_note") else None,
                "mode": cfg.mode,
                "alpha": cfg.alpha,
                "direction_kind": cfg.direction_kind,
                "sparse_topk": cfg.sparse_topk,
                "target_layers": list(cfg.target_layers),
                "components": list(cfg.components),
                "position_mode": cfg.position_mode,
                "apply_phase": cfg.apply_phase,
                "norm_scaling": cfg.norm_scaling,
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_h100_optimizations()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    additive_modes = _parse_csv(args.additive_modes)
    direction_kinds = _parse_csv(args.direction_kinds)
    additive_alphas = _parse_floats(args.additive_alphas)
    sparse_topk_values = _parse_ints(args.sparse_topk_values)
    patch_modes = _parse_csv(args.patch_modes)
    patch_alphas = _parse_floats(args.patch_alphas)
    components = _parse_csv(args.components)
    conditions = _parse_csv(args.conditions)
    if not components:
        raise ValueError("No components provided.")
    if not conditions:
        raise ValueError("No conditions provided.")

    masks = _load_masks(args.masks_path) if args.masks_path.exists() else None
    examples = _load_or_build_piqa_dataset(args)
    examples_by_uid = {ex.uid: ex for ex in examples}
    eval_uids = _select_eval_uids(args, examples_by_uid, masks)

    direction_file_vec, file_layer, _ = _load_direction_file(args.direction_path, args.direction_key)

    extraction_store: Optional[ExtractionStore] = None
    if args.extraction_dir.exists():
        extraction_store = _load_extraction_store(args.extraction_dir)

    model, tokenizer = load_model_and_tokenizer(model_name=args.model, device="auto", dtype="auto")
    model.eval()
    if not args.no_compile:
        model = try_compile_model(model)

    component_modules = get_component_modules(model)
    n_layers = len(get_transformer_layers(model))
    target_layers = _parse_layers(args.target_layers, n_layers=n_layers)
    layer_groups = [[layer] for layer in target_layers] if args.single_layer_sweep else [target_layers]

    condition_map = _condition_map(conditions, args.instruction_text)
    jobs = _build_jobs(
        uids=eval_uids,
        examples_by_uid=examples_by_uid,
        condition_map=condition_map,
        tokenizer=tokenizer,
        args=args,
    )
    batches = _dynamic_batches(jobs, max_batch_size=args.batch_size, max_batch_tokens=args.max_batch_tokens)
    print(
        f"Model={args.model} | prompts={len(jobs)} ({len(eval_uids)} uids x {len(conditions)} conditions) | "
        f"batches={len(batches)} | target_layers={target_layers}"
    )

    direction_uids: List[str] = []
    patch_uids: List[str] = []
    if masks is not None:
        direction_uids = _load_uids_file(args.direction_uids_file) if args.direction_uids_file else _load_uids_from_masks(masks, args.direction_uids_subset)
        patch_uids = _load_uids_file(args.patch_uids_file) if args.patch_uids_file else _load_uids_from_masks(masks, args.patch_uids_subset)

    direction_cache: Dict[Tuple[int, str, int], torch.Tensor] = {}
    patch_cache: Dict[Tuple[int, str, str, str], torch.Tensor] = {}

    def direction_for(layer_val: int, direction_kind: str, topk: int) -> torch.Tensor:
        cache_key = (layer_val, direction_kind, topk)
        if cache_key in direction_cache:
            return direction_cache[cache_key]
        if extraction_store is not None:
            dirs = _direction_from_store(
                extraction_store,
                uids=direction_uids,
                position=args.direction_position,
                layer_value=layer_val,
            )
            vec = dirs[direction_kind]
        else:
            if isinstance(direction_file_vec, dict):
                if layer_val not in direction_file_vec:
                    raise ValueError(
                        f"Layer {layer_val} not present in direction mapping from {args.direction_path}"
                    )
                vec = direction_file_vec[layer_val].clone()
            else:
                vec = direction_file_vec.clone()
        vec = _sparsify_vector(vec, topk)
        direction_cache[cache_key] = vec
        return vec

    def patch_mean_for(layer_val: int) -> torch.Tensor:
        key = (layer_val, args.patch_source_position, args.patch_source_style, args.patch_source_condition)
        if key in patch_cache:
            return patch_cache[key]
        if extraction_store is None:
            raise ValueError("Patch interventions require --extraction-dir activations.")
        vec = extraction_store.mean(
            position=args.patch_source_position,
            layer_value=layer_val,
            uids=patch_uids,
            style=args.patch_source_style,
            condition_code=args.patch_source_condition,
        )
        if vec is None:
            raise ValueError(f"No patch source mean for layer={layer_val}, position={args.patch_source_position}.")
        patch_cache[key] = vec
        return vec

    configs = _build_run_configs(
        include_baseline=args.include_baseline,
        additive_modes=additive_modes,
        direction_kinds=direction_kinds,
        additive_alphas=additive_alphas,
        sparse_topk_values=sparse_topk_values,
        patch_modes=patch_modes,
        patch_alphas=patch_alphas,
        layer_groups=layer_groups,
        components=components,
        position_mode=args.position_mode,
        apply_phase=args.apply_phase,
        norm_scaling=args.norm_scaling,
    )
    if args.max_configs > 0:
        configs = configs[: args.max_configs]
    if not configs:
        raise ValueError("No run configs built. Check additive/patch arguments.")

    rows_path = args.output_dir / "piqa_rows.jsonl"
    summary_path = args.output_dir / "piqa_summary.json"
    existing_rows: List[Dict[str, Any]] = []
    if rows_path.exists():
        existing_rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        print(f"[resume] loaded {len(existing_rows)} existing PIQA rows from {rows_path}")
    existing_by_config: Dict[str, List[Dict[str, Any]]] = {}
    for row in existing_rows:
        existing_by_config.setdefault(str(row.get("config_id")), []).append(row)
    completed_configs = {
        cfg_id
        for cfg_id, rows in existing_by_config.items()
        if len(rows) >= len(jobs) and all("raw_text" in row for row in rows)
    }
    summary_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    with rows_path.open("w") as out_file:
        for cfg in configs:
            print(f"\n=== {cfg.config_id} ===")
            if cfg.config_id in completed_configs:
                cfg_rows = existing_by_config[cfg.config_id][: len(jobs)]
                _parse_rows(cfg_rows, args.parse_n_jobs if args.parse_n_jobs != 0 else 1)
                summary_rows.extend(_summarize_config(cfg_rows, cfg))
                for row in cfg_rows:
                    out = {k: v for k, v in row.items() if k != "parse_meta"}
                    out_file.write(json.dumps(out, default=str) + "\n")
                out_file.flush()
                print(f"[resume] skipped complete {cfg.config_id} ({len(cfg_rows)} rows)")
                continue
            target_specs: List[TargetSpec] = []
            if cfg.mode != "none":
                for layer_val in cfg.target_layers:
                    d = None
                    p = None
                    if cfg.mode in {"add", "subtract", "project_out_direction"}:
                        if cfg.direction_kind is None:
                            raise ValueError(f"Missing direction kind for config={cfg.config_id}")
                        d = direction_for(layer_val, cfg.direction_kind, cfg.sparse_topk)
                    else:
                        p = patch_mean_for(layer_val)
                    for component in cfg.components:
                        if (layer_val, component) not in component_modules:
                            raise ValueError(f"Component target not found: layer={layer_val}, component={component}")
                        target_specs.append(TargetSpec(layer=layer_val, component=component, direction=d, patch_mean=p))

            cfg_rows: List[Dict[str, Any]] = []
            for batch_idx, chunk in enumerate(tqdm(batches, desc=cfg.config_id)):
                chunk_rows = _generate_steered_batch(
                    chunk,
                    model=model,
                    tokenizer=tokenizer,
                    component_modules=component_modules,
                    target_specs=target_specs,
                    alpha=cfg.alpha,
                    position_mode=cfg.position_mode,
                    apply_phase=cfg.apply_phase,
                    intervention_mode="add" if cfg.mode == "none" else cfg.mode,
                    norm_scaling=cfg.norm_scaling,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    greedy=args.temperature == 0.0,
                    use_cache=not args.no_cache,
                    seed=args.seed + batch_idx,
                    collect_margin_diagnostics=False,
                    token_id_a=None,
                    token_id_b=None,
                )
                for row, job in zip(chunk_rows, chunk):
                    row.update(
                        {
                            "config_id": cfg.config_id,
                            "mode": cfg.mode,
                            "direction_kind": cfg.direction_kind,
                            "sparse_topk": cfg.sparse_topk,
                            "position_mode": cfg.position_mode,
                            "apply_phase": cfg.apply_phase,
                            "target_layers": list(cfg.target_layers),
                            "components": list(cfg.components),
                            "norm_scaling": cfg.norm_scaling,
                            "alpha": cfg.alpha,
                            "prompt_text": job["prompt_text"],
                            "question": job["question"],
                        }
                    )
                cfg_rows.extend(chunk_rows)

            _parse_rows(cfg_rows, args.parse_n_jobs if args.parse_n_jobs != 0 else 1)
            summary_rows.extend(_summarize_config(cfg_rows, cfg))

            for row in cfg_rows:
                out = {k: v for k, v in row.items() if k != "parse_meta"}
                out["response"] = row["raw_text"]
                out_file.write(json.dumps(out, default=str) + "\n")
            out_file.flush()

    elapsed = time.time() - t0
    summary_path.write_text(json.dumps(summary_rows, indent=2))
    meta = {
        "model": args.model,
        "mc_dataset_path": str(args.mc_dataset_path),
        "n_examples_loaded": len(examples),
        "n_eval_uids": len(eval_uids),
        "conditions": conditions,
        "n_jobs": len(jobs),
        "n_batches": len(batches),
        "n_configs": len(configs),
        "target_layers": target_layers,
        "single_layer_sweep": bool(args.single_layer_sweep),
        "components": components,
        "position_mode": args.position_mode,
        "apply_phase": args.apply_phase,
        "direction_path": str(args.direction_path),
        "direction_key": args.direction_key,
        "direction_position": args.direction_position,
        "direction_uids_subset": args.direction_uids_subset,
        "direction_uids_file": str(args.direction_uids_file) if args.direction_uids_file is not None else None,
        "patch_uids_subset": args.patch_uids_subset,
        "patch_uids_file": str(args.patch_uids_file) if args.patch_uids_file is not None else None,
        "n_direction_uids": len(direction_uids),
        "n_patch_uids": len(patch_uids),
        "extraction_dir": str(args.extraction_dir),
        "patch_source_position": args.patch_source_position,
        "patch_source_style": args.patch_source_style,
        "patch_source_condition": args.patch_source_condition,
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "use_cache": not args.no_cache,
        "elapsed_seconds": round(elapsed, 2),
        "rows_path": str(rows_path),
        "summary_path": str(summary_path),
    }
    (args.output_dir / "piqa_meta.json").write_text(json.dumps(meta, indent=2))

    print("\n=== PIQA summary ===")
    print(f"{'Config':>42} {'Cond':>8} {'N':>6} {'Parsed%':>8} {'Acc%':>8} {'Wrong%':>8}")
    for row in summary_rows:
        acc = f"{row['accuracy_parsed']:.3f}" if row["accuracy_parsed"] is not None else "  n/a"
        wrong = f"{row['wrong_rate_parsed']:.3f}" if row["wrong_rate_parsed"] is not None else "  n/a"
        print(
            f"{row['config_id'][:42]:>42} {row['condition_code']:>8} {row['n']:>6d} "
            f"{row['parse_rate']:>8.3f} {acc:>8} {wrong:>8}"
        )
    if file_layer not in target_layers:
        print(f"Note: direction file default layer={file_layer}, active target layers={target_layers}.")
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
