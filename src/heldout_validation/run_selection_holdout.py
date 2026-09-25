from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from src.authority_steering.run_steering_test import _direction_from_store, _load_extraction_store, _load_masks, _load_uids_from_masks
from src.direction_controls.common import ModelSpec, load_axis_vector, load_direction_payload, normalize, residualize, run_module, write_json
from src.mitigation.common import DEFAULT_ALPHAS, Exp20Sweep, csv, parse_csv, specs_and_sweeps


OUTPUT_ROOT = Path("results/authority/heldout_validation/selection_holdout_50_50")
FULL_ANSWER_SUFFIX = "Answer the question in one short sentence using the full answer text, not option letters."
INTERVENTION_POSITION = "endorsement_span"
ENV = {
    "USE_HUB_KERNELS": "NO",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run clean 50/50 held-out selection validation for headline cells.")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--models", default="qwen35,gpt_oss,olmo2,olmo31")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--seed", type=int, default=20260505)
    p.add_argument("--fold", choices=("fold0", "fold1", "both"), default="fold0")
    p.add_argument(
        "--tasks",
        default="forward_trivia,source_user,mitigation_trivia,mitigation_piqa",
        help="CSV from: forward_trivia,forward_piqa,source_user,mitigation_trivia,mitigation_piqa.",
    )
    p.add_argument("--execute", action="store_true", help="Actually run GPU jobs; otherwise print commands and write manifests.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--max-trivia-items", type=int, default=512)
    p.add_argument("--max-piqa-samples", type=int, default=200)
    p.add_argument("--forward-alphas", default="0,0.3,0.5,0.7,1")
    p.add_argument("--mitigation-alphas", default=DEFAULT_ALPHAS)
    p.add_argument("--source-user-alphas", default="0,1")
    p.add_argument("--mitigation-variants", default="residualized,assistant,caa")
    p.add_argument("--caa-root", type=Path, default=Path("results/authority/mitigation/caa_sycophancy"))
    p.add_argument("--forward-mode", choices=("interpolate_mean", "replace_mean", "add", "subtract"), default="interpolate_mean")
    p.add_argument("--forward-conditions", default="N0_note")
    p.add_argument("--mitigation-conditions", default="W1_note")
    p.add_argument("--source-user-fit-conditions", default="no_cue,source_C1,source_W1,user_C1,user_W1")
    p.add_argument("--source-user-eval-conditions", default="source_W1,user_W1")
    p.add_argument("--plain-prompt", action="store_true")
    return p.parse_args()


def _write_lines(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def _split_uids(spec: ModelSpec, *, out_dir: Path, seed: int, max_items: int, force: bool) -> dict[str, Path | int | list[str]]:
    split_json = out_dir / "split.json"
    train_path = out_dir / "train_uids.txt"
    eval_path = out_dir / "eval_uids.txt"
    if split_json.exists() and train_path.exists() and eval_path.exists() and not force:
        payload = json.loads(split_json.read_text(encoding="utf-8"))
        return {**payload, "train_path": train_path, "eval_path": eval_path}

    masks = _load_masks(spec.masks_path)
    uids = list(_load_uids_from_masks(masks, "w1"))
    random.Random(seed).shuffle(uids)
    if max_items > 0:
        uids = uids[:max_items]
    mid = len(uids) // 2
    train_uids = sorted(uids[:mid])
    eval_uids = sorted(uids[mid:])
    _write_lines(train_path, train_uids)
    _write_lines(eval_path, eval_uids)
    payload = {
        "seed": seed,
        "max_items": max_items,
        "source_mask": "mask_primary_w1",
        "n_total": len(uids),
        "n_train": len(train_uids),
        "n_eval": len(eval_uids),
        "train_preview": train_uids[:5],
        "eval_preview": eval_uids[:5],
    }
    write_json(split_json, payload)
    return {**payload, "train_path": train_path, "eval_path": eval_path}


def _fold_paths(base: Path, fold: str) -> tuple[Path, Path]:
    split_dir = base / "splits" / fold
    return split_dir / "train_uids.txt", split_dir / "eval_uids.txt"


def _prepare_folds(spec: ModelSpec, *, model_out: Path, seed: int, max_items: int, force: bool, fold: str) -> list[str]:
    primary_dir = model_out / "splits" / "fold0"
    split = _split_uids(spec, out_dir=primary_dir, seed=seed, max_items=max_items, force=force)
    train0 = Path(split["train_path"])
    eval0 = Path(split["eval_path"])
    fold1_dir = model_out / "splits" / "fold1"
    fold1_dir.mkdir(parents=True, exist_ok=True)
    train1 = fold1_dir / "train_uids.txt"
    eval1 = fold1_dir / "eval_uids.txt"
    if force or not train1.exists() or not eval1.exists():
        shutil.copyfile(eval0, train1)
        shutil.copyfile(train0, eval1)
        write_json(
            fold1_dir / "split.json",
            {
                "seed": seed,
                "source": "fold0 swapped",
                "n_train": len([x for x in train1.read_text(encoding="utf-8").splitlines() if x.strip()]),
                "n_eval": len([x for x in eval1.read_text(encoding="utf-8").splitlines() if x.strip()]),
            },
        )
    return ["fold0", "fold1"] if fold == "both" else [fold]


def _write_layerwise_payload(path: Path, vectors: Dict[int, torch.Tensor], *, name: str, position: str, extra: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vector_by_layer": {str(k): normalize(v).cpu() for k, v in vectors.items()},
            "layer": int(next(iter(vectors))),
            "position": position,
            "direction_name": name,
            **extra,
        },
        path,
    )


def _derive_train_payloads(
    *,
    spec: ModelSpec,
    sweep: Exp20Sweep,
    train_uids_file: Path,
    out_dir: Path,
    force: bool,
) -> tuple[dict[str, Path], str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "authority": out_dir / "authority_train.pt",
        "assistant": out_dir / "assistant.pt",
        "residualized": out_dir / "residualized_train.pt",
    }
    if all(p.exists() for p in paths.values()) and not force:
        meta_path = out_dir / "norms.json"
        if meta_path.exists():
            rows = json.loads(meta_path.read_text(encoding="utf-8")).get("rows", [])
            if rows:
                return paths, str(rows[0].get("fit_position", "endorsement_end"))
        return paths, "endorsement_end"

    train_uids = [line.strip() for line in train_uids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    store = _load_extraction_store(spec.extraction_dir)
    _, _, fit_position, _ = load_direction_payload(spec.direction_path, "vector")
    if fit_position not in store.position_names:
        raise ValueError(f"{spec.name}: {fit_position} not in extraction positions {store.position_names}")

    authority: dict[int, torch.Tensor] = {}
    assistant: dict[int, torch.Tensor] = {}
    residualized_map: dict[int, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for layer in sweep.layers:
        if layer not in store.layer_to_index:
            raise ValueError(f"{spec.name}: layer {layer} missing in extraction target layers {store.target_layers}")
        dirs = _direction_from_store(store, uids=train_uids, position=fit_position, layer_value=layer)
        auth = normalize(dirs["shared_within_label"])
        axis = load_axis_vector(spec.axis_path, layer) if spec.axis_path else torch.zeros_like(auth)
        resid = normalize(residualize(auth, axis)) if float(axis.norm()) else auth
        authority[layer] = auth
        assistant[layer] = normalize(axis) if float(axis.norm()) else axis
        residualized_map[layer] = resid
        rows.append(
            {
                "layer": layer,
                "n_train_uids": len(train_uids),
                "fit_position": fit_position,
                "authority_norm": float(auth.norm()),
                "assistant_norm": float(axis.norm()),
                "residualized_norm": float(resid.norm()),
                "cos_authority_assistant": float(torch.dot(auth, normalize(axis))) if float(axis.norm()) else None,
            }
        )

    extra = {"fit_uids_file": str(train_uids_file), "fit_protocol": "exp22_50_50_train_fold"}
    _write_layerwise_payload(paths["authority"], authority, name="authority_train_fold", position=fit_position, extra=extra)
    _write_layerwise_payload(paths["assistant"], assistant, name="assistant_axis", position=fit_position, extra=extra)
    _write_layerwise_payload(paths["residualized"], residualized_map, name="authority_residualized_train_fold", position=fit_position, extra=extra)
    write_json(out_dir / "norms.json", {"rows": rows})
    return paths, fit_position


def _maybe_run(module: str, cmd: Sequence[str], *, execute: bool) -> int:
    return run_module(module, cmd, dry_run=not execute, env=ENV)


def _task_complete(out_dir: Path, sentinels: Sequence[str], *, force: bool) -> bool:
    if force or not out_dir.exists():
        return False
    for s in sentinels:
        if (out_dir / s).exists():
            return True
    # Allow any matching glob to count as well (e.g. results files written under nested dirs)
    for s in sentinels:
        if any(out_dir.rglob(s)):
            return True
    return False


def _speed_args(sweep: Exp20Sweep, *, no_compile: bool) -> list[str]:
    args = ["--batch-size", str(sweep.trivia_batch_size), "--max-batch-tokens", str(sweep.piqa_max_batch_tokens)]
    if no_compile:
        args.append("--no-compile")
    return args


def _run_forward_trivia(
    *,
    spec: ModelSpec,
    sweep: Exp20Sweep,
    train_uids: Path,
    eval_uids: Path,
    fit_position: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> int:
    cmd = [
        "--model", spec.model_id,
        "--direction-path", str(spec.direction_path),
        "--dataset-path", str(spec.trivia_dataset_path),
        "--masks-path", str(spec.masks_path),
        "--extraction-dir", str(spec.extraction_dir),
        "--uids-file", str(eval_uids),
        "--direction-uids-file", str(train_uids),
        "--patch-uids-file", str(train_uids),
        "--conditions", args.forward_conditions,
        "--target-layers", csv(sweep.layers),
        "--single-layer-sweep",
        "--position-modes", INTERVENTION_POSITION,
        "--components", "block_output",
        "--intervention-mode", args.forward_mode,
        "--patch-source-position", fit_position,
        "--patch-source-condition", "W1_note",
        "--norm-scaling", "none",
        "--alphas", args.forward_alphas,
        "--max-new-tokens", "256",
        "--answer-suffix", FULL_ANSWER_SUFFIX,
        "--output-dir", str(out_dir),
        "--max-items", "0",
        *_speed_args(sweep, no_compile=args.no_compile),
    ]
    if args.plain_prompt:
        cmd.append("--plain-prompt")
    return _maybe_run("src.authority_steering.run_steering_test", cmd, execute=args.execute)


def _run_forward_piqa(
    *,
    spec: ModelSpec,
    sweep: Exp20Sweep,
    train_uids: Path,
    fit_position: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> int:
    cmd = [
        "--model", spec.model_id,
        "--mc-dataset-path", str(spec.piqa_dataset_path),
        "--masks-path", str(spec.masks_path),
        "--direction-path", str(spec.direction_path),
        "--extraction-dir", str(spec.extraction_dir),
        "--direction-uids-file", str(train_uids),
        "--patch-uids-file", str(train_uids),
        "--conditions", args.forward_conditions,
        "--target-layers", csv(sweep.layers),
        "--single-layer-sweep",
        "--components", "block_output",
        "--position-mode", INTERVENTION_POSITION,
        "--patch-modes", args.forward_mode if args.forward_mode in {"interpolate_mean", "replace_mean"} else "",
        "--patch-alphas", args.forward_alphas,
        "--additive-modes", args.forward_mode if args.forward_mode in {"add", "subtract"} else "",
        "--additive-alphas", args.forward_alphas,
        "--direction-kinds", "shared_within_label",
        "--sparse-topk-values", "0",
        "--patch-source-position", fit_position,
        "--patch-source-condition", "W1_note",
        "--output-dir", str(out_dir),
        "--max-new-tokens", "256",
        "--max-samples", str(args.max_piqa_samples),
        "--answer-suffix", FULL_ANSWER_SUFFIX,
        "--batch-size", str(sweep.piqa_batch_size),
        "--max-batch-tokens", str(sweep.piqa_max_batch_tokens),
    ]
    if args.no_compile:
        cmd.append("--no-compile")
    if args.plain_prompt:
        cmd.append("--plain-prompt")
    return _maybe_run("src.authority_steering.run_piqa_interventions", cmd, execute=args.execute)


def _run_source_user(
    *,
    spec: ModelSpec,
    sweep: Exp20Sweep,
    train_uids: Path,
    eval_uids: Path,
    output_root: Path,
    model_name: str,
    args: argparse.Namespace,
) -> int:
    cmd = [
        "--repo-root", str(args.repo_root),
        "--models", model_name,
        "--dataset-path", str(spec.trivia_dataset_path),
        "--masks-path", str(spec.masks_path),
        "--output-root", str(output_root),
        "--layers", csv(sweep.layers),
        "--fit-uids-file", str(train_uids),
        "--eval-uids-file", str(eval_uids),
        "--fit-conditions", args.source_user_fit_conditions,
        "--eval-conditions", args.source_user_eval_conditions,
        "--alphas", args.source_user_alphas,
        "--max-items", "0",
        "--batch-size", str(sweep.trivia_batch_size),
        "--extract-batch-size", str(min(sweep.trivia_batch_size, 64)),
        "--max-batch-tokens", str(sweep.piqa_max_batch_tokens),
        "--max-new-tokens", "256",
    ]
    if args.force:
        cmd.append("--force")
    # Source/user extraction relies on forward hooks via SelectiveLayerCapture,
    # which can drop captures when the model is wrapped by torch.compile (manifests as
    # KeyError on the target layer). Force eager mode here regardless of --no-compile.
    cmd.append("--no-compile")
    if args.plain_prompt:
        cmd.append("--plain-prompt")
    return _maybe_run("src.mitigation.run_source_user_authority_split", cmd, execute=args.execute)


def _mitigation_direction_path(paths: dict[str, Path], caa_root: Path, model_name: str, variant: str) -> tuple[Path, str] | None:
    if variant == "authority":
        return paths["authority"], "project_out_direction"
    if variant == "assistant":
        return paths["assistant"], "project_out_direction"
    if variant == "residualized":
        return paths["residualized"], "project_out_direction"
    if variant == "caa":
        path = caa_root / model_name / "directions" / "sycophancy.pt"
        if not path.exists():
            return None
        return path, "subtract"
    raise ValueError(f"Unknown mitigation variant: {variant}")


def _run_mitigation_trivia(
    *,
    spec: ModelSpec,
    sweep: Exp20Sweep,
    eval_uids: Path,
    direction_path: Path,
    mode: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> int:
    cmd = [
        "--model", spec.model_id,
        "--direction-path", str(direction_path),
        "--direction-key", "vector_by_layer",
        "--dataset-path", str(spec.trivia_dataset_path),
        "--masks-path", str(spec.masks_path),
        "--uids-file", str(eval_uids),
        "--conditions", args.mitigation_conditions,
        "--target-layers", csv(sweep.layers),
        "--single-layer-sweep",
        "--position-modes", INTERVENTION_POSITION,
        "--components", "block_output",
        "--intervention-mode", mode,
        "--norm-scaling", "none",
        "--alphas", args.mitigation_alphas,
        "--max-new-tokens", "256",
        "--answer-suffix", FULL_ANSWER_SUFFIX,
        "--output-dir", str(out_dir),
        "--max-items", "0",
        *_speed_args(sweep, no_compile=args.no_compile),
    ]
    if args.plain_prompt:
        cmd.append("--plain-prompt")
    return _maybe_run("src.authority_steering.run_steering_test", cmd, execute=args.execute)


def _run_mitigation_piqa(
    *,
    spec: ModelSpec,
    sweep: Exp20Sweep,
    direction_path: Path,
    mode: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> int:
    cmd = [
        "--model", spec.model_id,
        "--mc-dataset-path", str(spec.piqa_dataset_path),
        "--masks-path", str(spec.masks_path),
        "--direction-path", str(direction_path),
        "--direction-key", "vector_by_layer",
        "--conditions", args.mitigation_conditions,
        "--target-layers", csv(sweep.layers),
        "--single-layer-sweep",
        "--components", "block_output",
        "--position-mode", INTERVENTION_POSITION,
        "--additive-modes", mode,
        "--norm-scaling", "none",
        "--direction-kinds", "shared_within_label",
        "--additive-alphas", args.mitigation_alphas,
        "--sparse-topk-values", "0",
        "--patch-modes", "",
        "--extraction-dir", str(out_dir / "__no_extract__"),
        "--output-dir", str(out_dir),
        "--max-new-tokens", "256",
        "--max-samples", str(args.max_piqa_samples),
        "--answer-suffix", FULL_ANSWER_SUFFIX,
        "--batch-size", str(sweep.piqa_batch_size),
        "--max-batch-tokens", str(sweep.piqa_max_batch_tokens),
    ]
    if args.no_compile:
        cmd.append("--no-compile")
    if args.plain_prompt:
        cmd.append("--plain-prompt")
    return _maybe_run("src.authority_steering.run_piqa_interventions", cmd, execute=args.execute)


def _write_analysis_plan(out_root: Path, payload: dict[str, Any]) -> None:
    write_json(out_root / "analysis_plan.json", payload)


def main() -> None:
    args = parse_args()
    tasks = set(parse_csv(args.tasks))
    valid_tasks = {"forward_trivia", "forward_piqa", "source_user", "mitigation_trivia", "mitigation_piqa"}
    unknown = tasks - valid_tasks
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "experiment": "exp22_selection_holdout_50_50",
        "created_unix": time.time(),
        "execute": bool(args.execute),
        "seed": args.seed,
        "fold": args.fold,
        "tasks": sorted(tasks),
        "models": {},
        "speedups": {
            "h100": True,
            "qwen_olmo_attention": "flash_attention_2 via src.models.llama_loader when available",
            "gpt_oss_attention": "eager/normal attention for attention-sink stability",
            "torch_compile": not args.no_compile,
            "dynamic_token_batches": True,
            "env": ENV,
        },
        "clean_protocol": {
            "fit": "Fit direction/patch means only on train_uids.txt.",
            "eval": "Evaluate headline behavior only on eval_uids.txt.",
            "split_source": "mask_primary_w1, seeded shuffle, then 50/50.",
            "gemma4": "intentionally excluded by default.",
        },
    }

    for model_name, spec, sweep in specs_and_sweeps(args.repo_root, args.models):
        model_out = args.output_root / model_name
        folds = _prepare_folds(spec, model_out=model_out, seed=args.seed, max_items=args.max_trivia_items, force=args.force, fold=args.fold)
        model_manifest: dict[str, Any] = {"model_id": spec.model_id, "layers": list(sweep.layers), "folds": {}}
        for fold_name in folds:
            train_uids, eval_uids = _fold_paths(model_out, fold_name)
            fold_out = model_out / fold_name
            direction_paths, fit_position = _derive_train_payloads(
                spec=spec,
                sweep=sweep,
                train_uids_file=train_uids,
                out_dir=fold_out / "directions",
                force=args.force,
            )
            fold_manifest: dict[str, Any] = {
                "train_uids": str(train_uids),
                "eval_uids": str(eval_uids),
                "fit_position": fit_position,
                "directions": {k: str(v) for k, v in direction_paths.items()},
                "runs": [],
            }
            if "forward_trivia" in tasks:
                ft_dir = fold_out / "forward_trivia"
                if _task_complete(ft_dir, ("steering_summary.json",), force=args.force):
                    print(f"[exp22] skip forward_trivia (already done): {ft_dir}")
                    fold_manifest["runs"].append({"task": "forward_trivia", "skipped": "already_done"})
                else:
                    rc = _run_forward_trivia(
                        spec=spec,
                        sweep=sweep,
                        train_uids=train_uids,
                        eval_uids=eval_uids,
                        fit_position=fit_position,
                        out_dir=ft_dir,
                        args=args,
                    )
                    fold_manifest["runs"].append({"task": "forward_trivia", "returncode": rc})
                    if args.execute and rc:
                        print(f"[exp22] WARNING: forward_trivia failed for {model_name}/{fold_name} (rc={rc}); continuing.")
            if "forward_piqa" in tasks:
                fp_dir = fold_out / "forward_piqa"
                if _task_complete(fp_dir, ("piqa_summary.json",), force=args.force):
                    print(f"[exp22] skip forward_piqa (already done): {fp_dir}")
                    fold_manifest["runs"].append({"task": "forward_piqa", "skipped": "already_done"})
                else:
                    rc = _run_forward_piqa(
                        spec=spec,
                        sweep=sweep,
                        train_uids=train_uids,
                        fit_position=fit_position,
                        out_dir=fp_dir,
                        args=args,
                    )
                    fold_manifest["runs"].append({"task": "forward_piqa", "returncode": rc})
                    if args.execute and rc:
                        print(f"[exp22] WARNING: forward_piqa failed for {model_name}/{fold_name} (rc={rc}); continuing.")
            if "source_user" in tasks:
                source_root = fold_out / "source_user"
                if _task_complete(source_root / model_name, ("source_user_authority_summary.json",), force=args.force):
                    print(f"[exp22] skip source_user (already done): {source_root / model_name}")
                    fold_manifest["runs"].append({"task": "source_user", "skipped": "already_done", "output_dir": str(source_root / model_name)})
                else:
                    rc = _run_source_user(
                        spec=spec,
                        sweep=sweep,
                        train_uids=train_uids,
                        eval_uids=eval_uids,
                        output_root=source_root,
                        model_name=model_name,
                        args=args,
                    )
                    fold_manifest["runs"].append({"task": "source_user", "returncode": rc, "output_dir": str(source_root / model_name)})
                    if args.execute and rc:
                        print(f"[exp22] WARNING: source_user failed for {model_name}/{fold_name} (rc={rc}); continuing.")

            for variant in parse_csv(args.mitigation_variants):
                path_mode = _mitigation_direction_path(direction_paths, args.caa_root, model_name, variant)
                if path_mode is None:
                    fold_manifest["runs"].append({"task": "mitigation", "variant": variant, "skipped": "missing_caa_direction"})
                    continue
                direction_path, mode = path_mode
                if "mitigation_trivia" in tasks:
                    mt_dir = fold_out / f"mitigation_trivia_{variant}"
                    if _task_complete(mt_dir, ("steering_summary.json",), force=args.force):
                        print(f"[exp22] skip mitigation_trivia ({variant}) (already done): {mt_dir}")
                        fold_manifest["runs"].append({"task": "mitigation_trivia", "variant": variant, "skipped": "already_done"})
                    else:
                        rc = _run_mitigation_trivia(
                            spec=spec,
                            sweep=sweep,
                            eval_uids=eval_uids,
                            direction_path=direction_path,
                            mode=mode,
                            out_dir=mt_dir,
                            args=args,
                        )
                        fold_manifest["runs"].append({"task": "mitigation_trivia", "variant": variant, "mode": mode, "returncode": rc})
                        if args.execute and rc:
                            print(f"[exp22] WARNING: mitigation_trivia failed for {model_name}/{fold_name}/{variant} (rc={rc}); continuing.")
                if "mitigation_piqa" in tasks:
                    mp_dir = fold_out / f"mitigation_piqa_{variant}"
                    if _task_complete(mp_dir, ("piqa_summary.json",), force=args.force):
                        print(f"[exp22] skip mitigation_piqa ({variant}) (already done): {mp_dir}")
                        fold_manifest["runs"].append({"task": "mitigation_piqa", "variant": variant, "skipped": "already_done"})
                    else:
                        rc = _run_mitigation_piqa(
                            spec=spec,
                            sweep=sweep,
                            direction_path=direction_path,
                            mode=mode,
                            out_dir=mp_dir,
                            args=args,
                        )
                        fold_manifest["runs"].append({"task": "mitigation_piqa", "variant": variant, "mode": mode, "returncode": rc})
                        if args.execute and rc:
                            print(f"[exp22] WARNING: mitigation_piqa failed for {model_name}/{fold_name}/{variant} (rc={rc}); continuing.")
            model_manifest["folds"][fold_name] = fold_manifest
            manifest["models"][model_name] = model_manifest
            write_json(args.output_root / "manifest.json", manifest)

    _write_analysis_plan(
        args.output_root,
        {
            "headline_comparison": "For each model/task/config, compare original in-sample headline cell to exp22 held-out eval cell.",
            "primary_report": [
                "best in-sample config and effect",
                "same slot under train-fit/eval-only",
                "ratio heldout/headline",
                "median ratio across swept slots",
                "bootstrap CI over eval UIDs plus across-slot IQR",
            ],
            "acceptance_rule_of_thumb": "Headline survives if held-out effect remains within roughly 10-15 percent of original and rank order is stable.",
        },
    )
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps({"manifest": str(args.output_root / "manifest.json"), "execute": bool(args.execute)}, indent=2))


if __name__ == "__main__":
    main()
