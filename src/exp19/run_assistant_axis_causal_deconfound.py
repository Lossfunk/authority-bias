from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.exp19.common import (
    default_model_specs,
    load_axis_vector,
    load_direction_payload,
    parse_csv,
    residualize,
    run_module,
    save_direction_payload,
    write_json,
)


POSITION_TO_PATCH_MODE = {
    "endorsement_end": "note_end",
    "endorsement_start": "note_start",
    "endorsed_answer": "endorsed_answer",
    "answer_position": "answer_position",
    "endorsement_mean": "endorsement_span",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run assistant-axis causal deconfound experiments (A/B).")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--models", type=str, default="gpt_oss,gemma4,olmo2,qwen35")
    p.add_argument("--variants", type=str, default="authority,assistant,residualized")
    p.add_argument("--output-root", type=Path, default=Path("results/authority/exp19/assistant_axis_causal_deconfound"))
    p.add_argument("--trivia-alphas", type=str, default="0,0.3,0.5,1.0")
    p.add_argument("--piqa-alphas", type=str, default="0,0.3,0.5,1.0")
    p.add_argument("--max-items", type=int, default=0, help="Default item cap applied to both tasks if task-specific caps are unset.")
    p.add_argument("--trivia-max-items", type=int, default=None, help="Override --max-items for trivia only (0 = no cap).")
    p.add_argument("--piqa-max-items", type=int, default=None, help="Override --max-items for PIQA only (0 = no cap).")
    p.add_argument("--run-trivia", action="store_true")
    p.add_argument("--run-piqa", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--skip-missing-axis", action="store_true")
    p.add_argument("--trivia-batch-size", type=int, default=8)
    p.add_argument("--piqa-batch-size", type=int, default=8)
    p.add_argument("--piqa-max-batch-tokens", type=int, default=16384)
    p.add_argument("--piqa-max-new-tokens", type=int, default=256)
    p.add_argument("--trivia-conditions", type=str, default="N0_note")
    p.add_argument("--piqa-conditions", type=str, default="N0_note")
    p.add_argument(
        "--trivia-position-mode",
        type=str,
        default="auto",
        help="Token-position mode for trivia intervention. 'auto' reads the position from the direction payload "
        "(same position the direction was fit at). Pass e.g. 'endorsement_span' to override.",
    )
    p.add_argument(
        "--piqa-position-mode",
        type=str,
        default="auto",
        help="Token-position mode for PIQA intervention. 'auto' reads the position from the direction payload.",
    )
    p.add_argument(
        "--trivia-intervention-mode",
        type=str,
        default="add",
        choices=("add", "subtract", "project_out_direction"),
        help="Direction-specific intervention mode for trivia. "
        "'add' injects alpha*direction; 'project_out_direction' removes alpha*<base,dir>*dir.",
    )
    p.add_argument(
        "--piqa-intervention-mode",
        type=str,
        default="add",
        choices=("add", "subtract", "project_out_direction"),
        help="Direction-specific intervention mode for PIQA.",
    )
    p.add_argument(
        "--trivia-norm-scaling",
        type=str,
        default="none",
        choices=("none", "resid_norm", "resid_std"),
        help="If 'resid_norm', the additive delta is scaled by the base hidden's L2 norm "
        "so alpha behaves like a fraction of residual-stream magnitude.",
    )
    p.add_argument(
        "--piqa-norm-scaling",
        type=str,
        default="none",
        choices=("none", "resid_norm", "resid_std"),
    )
    p.add_argument(
        "--trivia-alpha-override",
        type=str,
        default=None,
        help="If set, overrides frozen-best trivia alpha list (still adds alpha=0 baseline). "
        "Useful for calibrating add+resid_norm or project_out_direction sweeps.",
    )
    p.add_argument(
        "--piqa-alpha-override",
        type=str,
        default=None,
        help="If set, overrides frozen-best PIQA alpha list.",
    )
    p.add_argument("--gpu-devices", type=str, default="", help="Comma list like '0,1'; assigned round-robin per model.")
    p.add_argument("--resume", action="store_true", help="Skip runs with existing completed outputs/checkpoint entries.")
    p.add_argument("--overwrite", action="store_true", help="Force rerun even if outputs/checkpoint exist.")
    p.add_argument("--checkpoint-path", type=Path, default=None, help="Optional manifest checkpoint path.")
    p.add_argument(
        "--freeze-best-settings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use best (layer, alpha) per model/task from original forward-patch summaries.",
    )
    p.add_argument(
        "--best-settings-json",
        type=Path,
        default=None,
        help="Optional manual settings override JSON: {model: {trivia|piqa: {layer, alpha, condition}}}.",
    )
    p.add_argument(
        "--allow-missing-frozen-settings",
        action="store_true",
        help="If frozen settings are missing, fall back to alpha sweeps instead of raising.",
    )
    p.add_argument("--no-compile", action="store_true", help="Pass --no-compile to downstream steering/piqa modules.")
    p.add_argument(
        "--dedupe-alpha-zero",
        action="store_true",
        help="alpha=0 produces identical outputs across authority/assistant/residualized variants (no intervention). "
        "When set, only the authority variant runs alpha=0; assistant/residualized skip alpha=0.",
    )
    return p.parse_args()


def _parse_alpha_list(raw: str) -> List[float]:
    return [float(x) for x in parse_csv(raw)]


def _alpha_with_baseline(alpha: float) -> str:
    if abs(alpha) < 1e-9:
        return "0"
    return ",".join(["0", f"{alpha:g}"])


def _row_condition(row: Dict[str, Any]) -> str:
    val = row.get("condition")
    if val is None:
        val = row.get("condition_code")
    return str(val or "")


def _row_layer(row: Dict[str, Any]) -> int | None:
    layers = row.get("target_layers")
    if isinstance(layers, list) and layers:
        try:
            return int(layers[0])
        except Exception:
            pass
    cfg = str(row.get("config_id", ""))
    m = re.search(r"_L(\d+)", cfg)
    if m:
        return int(m.group(1))
    return None


def _row_score(row: Dict[str, Any]) -> float | None:
    """
    Pick the strongest available "how much did steering move behavior" signal.

    Preference order:
      1. matched_flip_rate (if present)
      2. flip_rate (forward-patch style)
      3. wrong_rate (trivia summaries) — larger = more compliance with the endorsement
      4. 1 - accuracy (trivia/piqa fallback when accuracy is present)
    """
    for key in ("matched_flip_rate", "flip_rate", "flip_rate_parsed"):
        val = row.get(key)
        if val is not None:
            try:
                return float(val)
            except Exception:
                continue
    for key in ("wrong_rate", "wrong_rate_parsed"):
        val = row.get(key)
        if val is not None:
            try:
                return float(val)
            except Exception:
                continue
    for key in ("accuracy", "accuracy_parsed"):
        val = row.get(key)
        if val is not None:
            try:
                return 1.0 - float(val)
            except Exception:
                continue
    return None


def _best_from_summary(
    summary_path: Path,
    preferred_condition: str = "N0_note",
    *,
    layer_filter: int | None = None,
) -> Dict[str, Any] | None:
    if not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return None
    rows = [r for r in data if isinstance(r, dict)]
    if not rows:
        return None

    def _pick(candidates: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        scored: List[Dict[str, Any]] = []
        for row in candidates:
            score = _row_score(row)
            if score is None:
                continue
            try:
                alpha = float(row.get("alpha", 0.0))
            except Exception:
                continue
            # Never pick the baseline alpha=0 row as "best" — it is the no-op setting.
            if abs(alpha) < 1e-9:
                continue
            layer = _row_layer(row)
            if layer is None:
                continue
            if layer_filter is not None and int(layer) != int(layer_filter):
                continue
            scored.append(
                {
                    "alpha": alpha,
                    "layer": int(layer),
                    "condition": _row_condition(row),
                    "score": float(score),
                    "config_id": row.get("config_id"),
                    "source_summary_path": str(summary_path),
                }
            )
        if not scored:
            return None
        return max(scored, key=lambda x: (x["score"], -abs(float(x["alpha"]))))

    cond_pref = [preferred_condition, "W1_note", "N0_note", "C1_note"]
    for cond in cond_pref:
        subset = [r for r in rows if _row_condition(r) == cond]
        picked = _pick(subset)
        if picked is not None:
            return picked
    return _pick(rows)


def _safe_exists(path: Path | None) -> bool:
    return path is not None and path.exists()


def _derive_dirs(model_name: str, spec, out_root: Path) -> Dict[str, Path]:
    authority_vec, layer, position, meta = load_direction_payload(spec.direction_path, "vector")
    if not _safe_exists(spec.axis_path):
        raise FileNotFoundError(f"Assistant axis missing for {model_name}: {spec.axis_path}")
    axis_vec = load_axis_vector(spec.axis_path, layer=layer)
    residual_vec = residualize(authority_vec, axis_vec)
    if float(residual_vec.norm().item()) == 0.0:
        raise ValueError(f"Residualized direction collapsed to zero for {model_name}.")

    model_dir = out_root / model_name / "directions"
    authority_path = model_dir / "authority.pt"
    assistant_path = model_dir / "assistant_axis.pt"
    residual_path = model_dir / "authority_residualized.pt"

    save_direction_payload(
        path=authority_path,
        vector=authority_vec,
        layer=layer,
        position=position,
        direction_name="authority_baseline",
        extra={"source_direction_path": str(spec.direction_path), "source_meta": meta},
    )
    save_direction_payload(
        path=assistant_path,
        vector=axis_vec,
        layer=layer,
        position=position,
        direction_name="assistant_axis_only",
        extra={"source_axis_path": str(spec.axis_path)},
    )
    save_direction_payload(
        path=residual_path,
        vector=residual_vec,
        layer=layer,
        position=position,
        direction_name="authority_residualized_against_assistant",
        extra={"source_axis_path": str(spec.axis_path), "source_direction_path": str(spec.direction_path)},
    )
    return {"authority": authority_path, "assistant": assistant_path, "residualized": residual_path}


TRIVIA_ANSWER_SUFFIX = "Answer the question in one short sentence using the full answer text, not option letters."
PIQA_ANSWER_SUFFIX = "Answer the question in one short sentence using the full answer text, not option letters."


def _trivia_cmd(
    spec,
    direction_path: Path,
    out_dir: Path,
    condition: str,
    target_layer: int,
    position_mode: str,
    alphas: str,
    max_items: int,
    batch_size: int,
    intervention_mode: str = "add",
    norm_scaling: str = "none",
    answer_suffix: str = TRIVIA_ANSWER_SUFFIX,
    no_compile: bool = False,
) -> List[str]:
    vec, _, _, _ = load_direction_payload(direction_path, "vector")
    _ = vec
    args = [
        "--model", spec.model_id,
        "--direction-path", str(direction_path),
        "--direction-key", "vector",
        "--dataset-path", str(spec.trivia_dataset_path),
        "--masks-path", str(spec.masks_path),
        "--uids-subset", "w1",
        "--conditions", condition,
        "--target-layers", str(target_layer),
        "--position-modes", position_mode,
        "--components", "block_output",
        "--intervention-mode", intervention_mode,
        "--norm-scaling", norm_scaling,
        "--alphas", alphas,
        "--max-new-tokens", "256",
        "--batch-size", str(batch_size),
        "--answer-suffix", answer_suffix,
        "--output-dir", str(out_dir),
    ]
    if max_items > 0:
        args += ["--max-items", str(max_items)]
    if no_compile:
        args += ["--no-compile"]
    return args


def _piqa_cmd(
    spec,
    direction_path: Path,
    out_dir: Path,
    condition: str,
    target_layer: int,
    position_mode: str,
    alphas: str,
    max_items: int,
    batch_size: int,
    max_batch_tokens: int,
    max_new_tokens: int,
    intervention_mode: str = "add",
    norm_scaling: str = "none",
    answer_suffix: str = PIQA_ANSWER_SUFFIX,
    no_compile: bool = False,
) -> List[str]:
    no_extract = out_dir / "__no_extract__"
    args = [
        "--model", spec.model_id,
        "--mc-dataset-path", str(spec.piqa_dataset_path),
        "--masks-path", str(spec.masks_path),
        "--direction-path", str(direction_path),
        "--direction-key", "vector",
        "--conditions", condition,
        "--target-layers", str(target_layer),
        "--components", "block_output",
        "--position-mode", position_mode,
        "--additive-modes", intervention_mode,
        "--norm-scaling", norm_scaling,
        "--direction-kinds", "shared_within_label",
        "--additive-alphas", alphas,
        "--sparse-topk-values", "0",
        "--patch-modes", "",
        "--extraction-dir", str(no_extract),
        "--output-dir", str(out_dir),
        "--max-new-tokens", str(max_new_tokens),
        "--batch-size", str(batch_size),
        "--max-batch-tokens", str(max_batch_tokens),
        "--answer-suffix", answer_suffix,
    ]
    if max_items > 0:
        args += ["--max-samples", str(max_items)]
    if no_compile:
        args += ["--no-compile"]
    return args


def _load_overrides(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Missing best-settings JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"best-settings JSON must be an object: {path}")
    return payload


def _resolve_frozen_setting(
    *,
    model_name: str,
    task: str,
    spec,
    preferred_condition: str,
    model_override: Dict[str, Any],
    layer_filter: int | None = None,
) -> Dict[str, Any] | None:
    override = model_override.get(task)
    if isinstance(override, dict):
        return {
            "alpha": float(override["alpha"]),
            "layer": int(override["layer"]),
            "condition": str(override.get("condition", preferred_condition)),
            "source": "override",
        }
    summary_path = spec.trivia_reference_summary_path if task == "trivia" else spec.piqa_reference_summary_path
    if summary_path is None:
        return None
    best = _best_from_summary(
        summary_path,
        preferred_condition=preferred_condition,
        layer_filter=layer_filter,
    )
    if best is None:
        return None
    return {**best, "source": "reference_summary"}


def _extract_summary_row(summary_path: Path, *, condition: str, alpha: float, layer: int) -> Dict[str, Any] | None:
    if not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return None
    for row in data:
        if not isinstance(row, dict):
            continue
        if _row_condition(row) != condition:
            continue
        if _row_layer(row) != layer:
            continue
        try:
            if abs(float(row.get("alpha", 0.0)) - alpha) > 1e-9:
                continue
        except Exception:
            continue
        return row
    return None


def main() -> None:
    args = parse_args()
    if not args.run_trivia and not args.run_piqa:
        args.run_trivia = True
        args.run_piqa = True

    specs = default_model_specs(args.repo_root.resolve())
    selected = parse_csv(args.models)
    selected_variants = parse_csv(args.variants)
    gpu_pool = parse_csv(args.gpu_devices)
    overrides = _load_overrides(args.best_settings_json)
    checkpoint_path = args.checkpoint_path or (args.output_root / "assistant_axis_causal_deconfound_manifest.json")
    if args.resume and checkpoint_path.exists():
        try:
            manifest: Dict[str, object] = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            manifest["execute"] = bool(args.execute)
        except Exception:
            manifest = {"models": {}, "execute": bool(args.execute)}
    else:
        manifest = {"models": {}, "execute": bool(args.execute)}
    models_payload = manifest.setdefault("models", {})

    def _already_done(existing_runs: List[Dict[str, object]], task: str, variant: str, expected_path: Path) -> bool:
        if not expected_path.exists():
            return False
        for row in existing_runs:
            if row.get("task") == task and row.get("variant") == variant and int(row.get("returncode", 1)) == 0:
                return True
        return False

    def _run_one(name: str, gpu_id: str | None) -> Dict[str, object] | None:
        if name not in specs:
            print(f"[skip] unknown model key: {name}")
            return None
        spec = specs[name]
        if not spec.direction_path.exists():
            print(f"[skip] missing direction file: {spec.direction_path}")
            return None
        if not _safe_exists(spec.axis_path):
            msg = f"[skip] missing assistant axis for {name}: {spec.axis_path}"
            if args.skip_missing_axis:
                print(msg)
                return None
            raise SystemExit(msg)

        print(f"\n=== {name} ===")
        derived = _derive_dirs(name, spec, args.output_root)
        _, authority_layer, authority_position, _ = load_direction_payload(derived["authority"], "vector")
        # Direction payloads use extractor-side labels (endorsement_end, endorsement_mean, ...),
        # which differ from the exp16 steering runner's position-mode vocabulary.
        # POSITION_TO_PATCH_MODE maps extractor -> steering-runner names.
        mapped_position = POSITION_TO_PATCH_MODE.get(authority_position, authority_position)
        trivia_position_mode = args.trivia_position_mode if args.trivia_position_mode != "auto" else mapped_position
        piqa_position_mode = args.piqa_position_mode if args.piqa_position_mode != "auto" else mapped_position
        existing_model = models_payload.get(name, {}) if isinstance(models_payload, dict) else {}
        existing_runs = list(existing_model.get("runs", [])) if isinstance(existing_model, dict) else []
        model_manifest: Dict[str, Any] = {
            "derived_directions": {k: str(v) for k, v in derived.items()},
            "runs": existing_runs,
            "frozen_settings": {},
            "comparison": {},
            "resolved": {
                "authority_layer": int(authority_layer),
                "authority_position": str(authority_position),
                "trivia_position_mode": str(trivia_position_mode),
                "piqa_position_mode": str(piqa_position_mode),
            },
        }
        print(f"[{name}] authority layer={authority_layer} position={authority_position} -> trivia={trivia_position_mode}, piqa={piqa_position_mode}")
        run_env = {"CUDA_VISIBLE_DEVICES": gpu_id} if gpu_id is not None and gpu_id != "" else None

        model_override = overrides.get(name, {}) if isinstance(overrides.get(name, {}), dict) else {}

        frozen_trivia = None
        frozen_piqa = None
        if args.freeze_best_settings:
            if args.run_trivia:
                frozen_trivia = _resolve_frozen_setting(
                    model_name=name,
                    task="trivia",
                    spec=spec,
                    preferred_condition=args.trivia_conditions,
                    model_override=model_override,
                    layer_filter=authority_layer,
                )
                if frozen_trivia is None and not args.allow_missing_frozen_settings:
                    raise SystemExit(
                        f"Missing frozen trivia setting for {name}. "
                        f"Provide --best-settings-json or use --allow-missing-frozen-settings."
                    )
                if frozen_trivia is not None and int(frozen_trivia["layer"]) != authority_layer:
                    raise SystemExit(
                        f"Frozen trivia layer mismatch for {name}: frozen={frozen_trivia['layer']} "
                        f"authority_direction_layer={authority_layer}"
                    )
            if args.run_piqa:
                frozen_piqa = _resolve_frozen_setting(
                    model_name=name,
                    task="piqa",
                    spec=spec,
                    preferred_condition=args.piqa_conditions,
                    model_override=model_override,
                    layer_filter=authority_layer,
                )
                if frozen_piqa is None and not args.allow_missing_frozen_settings:
                    raise SystemExit(
                        f"Missing frozen PIQA setting for {name}. "
                        f"Provide --best-settings-json or use --allow-missing-frozen-settings."
                    )
                if frozen_piqa is not None and int(frozen_piqa["layer"]) != authority_layer:
                    raise SystemExit(
                        f"Frozen PIQA layer mismatch for {name}: frozen={frozen_piqa['layer']} "
                        f"authority_direction_layer={authority_layer}"
                    )

        if frozen_trivia is not None:
            model_manifest["frozen_settings"]["trivia"] = frozen_trivia
        if frozen_piqa is not None:
            model_manifest["frozen_settings"]["piqa"] = frozen_piqa

        def _variant_alphas(base: str, variant: str) -> str:
            if not args.dedupe_alpha_zero or variant == "authority":
                return base
            kept = [a for a in _parse_alpha_list(base) if abs(float(a)) > 1e-9]
            return ",".join(f"{a:g}" for a in kept) if kept else base

        for variant_name in selected_variants:
            if variant_name not in {"authority", "assistant", "residualized"}:
                raise SystemExit(f"Unknown variant: {variant_name}")
            dpath = derived[variant_name]
            if args.run_trivia:
                out_dir = args.output_root / name / f"trivia_{variant_name}"
                expected = out_dir / "steering_summary.json"
                if args.resume and not args.overwrite and _already_done(model_manifest["runs"], "trivia", variant_name, expected):
                    print(f"[resume] skip trivia/{name}/{variant_name}")
                else:
                    if args.trivia_alpha_override is not None:
                        trivia_alphas = _variant_alphas(
                            ",".join(["0"] + [x.strip() for x in args.trivia_alpha_override.split(",") if x.strip() and float(x) != 0.0]),
                            variant_name,
                        )
                        # User-specified conditions always win over frozen-best when an alpha override is in use,
                        # so we can sweep multiple conditions (e.g. N0_note,W1_note,C1_note) at a fixed alpha.
                        trivia_condition = args.trivia_conditions
                        trivia_layer = int(frozen_trivia["layer"]) if frozen_trivia is not None else authority_layer
                    elif frozen_trivia is not None:
                        trivia_alphas = _alpha_with_baseline(float(frozen_trivia["alpha"]))
                        trivia_alphas = _variant_alphas(trivia_alphas, variant_name)
                        trivia_condition = str(frozen_trivia["condition"])
                        trivia_layer = int(frozen_trivia["layer"])
                    else:
                        trivia_alphas = _variant_alphas(args.trivia_alphas, variant_name)
                        trivia_condition = args.trivia_conditions
                        trivia_layer = authority_layer
                    trivia_items = args.trivia_max_items if args.trivia_max_items is not None else args.max_items
                    cmd_args = _trivia_cmd(
                        spec,
                        dpath,
                        out_dir,
                        trivia_condition,
                        trivia_layer,
                        trivia_position_mode,
                        trivia_alphas,
                        trivia_items,
                        args.trivia_batch_size,
                        intervention_mode=args.trivia_intervention_mode,
                        norm_scaling=args.trivia_norm_scaling,
                        no_compile=bool(args.no_compile),
                    )
                    rc = run_module("src.exp16.run_steering_test", cmd_args, dry_run=not args.execute, env=run_env)
                    model_manifest["runs"].append(
                        {"task": "trivia", "variant": variant_name, "module": "src.exp16.run_steering_test", "args": cmd_args, "returncode": rc}
                    )
                    models_payload[name] = model_manifest
                    write_json(checkpoint_path, manifest)
                    if args.execute and rc != 0:
                        raise SystemExit(f"Run failed: trivia/{name}/{variant_name}")
                if frozen_trivia is not None:
                    row = _extract_summary_row(
                        expected,
                        condition=str(frozen_trivia["condition"]),
                        alpha=float(frozen_trivia["alpha"]),
                        layer=int(frozen_trivia["layer"]),
                    )
                    if row is not None:
                        model_manifest.setdefault("comparison", {}).setdefault("trivia", {})[variant_name] = {
                            "condition": str(frozen_trivia["condition"]),
                            "alpha": float(frozen_trivia["alpha"]),
                            "layer": int(frozen_trivia["layer"]),
                            "matched_flip_rate": row.get("matched_flip_rate"),
                            "wrong_rate": row.get("wrong_rate"),
                            "accuracy": row.get("accuracy"),
                            "config_id": row.get("config_id"),
                        }

            if args.run_piqa:
                out_dir = args.output_root / name / f"piqa_{variant_name}"
                expected = out_dir / "piqa_summary.json"
                if args.resume and not args.overwrite and _already_done(model_manifest["runs"], "piqa", variant_name, expected):
                    print(f"[resume] skip piqa/{name}/{variant_name}")
                else:
                    if args.piqa_alpha_override is not None:
                        piqa_alphas = _variant_alphas(
                            ",".join(["0"] + [x.strip() for x in args.piqa_alpha_override.split(",") if x.strip() and float(x) != 0.0]),
                            variant_name,
                        )
                        piqa_condition = args.piqa_conditions
                        piqa_layer = int(frozen_piqa["layer"]) if frozen_piqa is not None else authority_layer
                    elif frozen_piqa is not None:
                        piqa_alphas = _alpha_with_baseline(float(frozen_piqa["alpha"]))
                        piqa_alphas = _variant_alphas(piqa_alphas, variant_name)
                        piqa_condition = str(frozen_piqa["condition"])
                        piqa_layer = int(frozen_piqa["layer"])
                    else:
                        piqa_alphas = _variant_alphas(args.piqa_alphas, variant_name)
                        piqa_condition = args.piqa_conditions
                        piqa_layer = authority_layer
                    piqa_items = args.piqa_max_items if args.piqa_max_items is not None else args.max_items
                    cmd_args = _piqa_cmd(
                        spec,
                        dpath,
                        out_dir,
                        piqa_condition,
                        piqa_layer,
                        piqa_position_mode,
                        piqa_alphas,
                        piqa_items,
                        args.piqa_batch_size,
                        args.piqa_max_batch_tokens,
                        args.piqa_max_new_tokens,
                        intervention_mode=args.piqa_intervention_mode,
                        norm_scaling=args.piqa_norm_scaling,
                        no_compile=bool(args.no_compile),
                    )
                    rc = run_module("src.exp16.run_piqa_interventions", cmd_args, dry_run=not args.execute, env=run_env)
                    model_manifest["runs"].append(
                        {"task": "piqa", "variant": variant_name, "module": "src.exp16.run_piqa_interventions", "args": cmd_args, "returncode": rc}
                    )
                    models_payload[name] = model_manifest
                    write_json(checkpoint_path, manifest)
                    if args.execute and rc != 0:
                        raise SystemExit(f"Run failed: piqa/{name}/{variant_name}")
                if frozen_piqa is not None:
                    row = _extract_summary_row(
                        expected,
                        condition=str(frozen_piqa["condition"]),
                        alpha=float(frozen_piqa["alpha"]),
                        layer=int(frozen_piqa["layer"]),
                    )
                    if row is not None:
                        model_manifest.setdefault("comparison", {}).setdefault("piqa", {})[variant_name] = {
                            "condition": str(frozen_piqa["condition"]),
                            "alpha": float(frozen_piqa["alpha"]),
                            "layer": int(frozen_piqa["layer"]),
                            "matched_flip_rate": row.get("matched_flip_rate"),
                            "wrong_rate_parsed": row.get("wrong_rate_parsed"),
                            "accuracy_parsed": row.get("accuracy_parsed"),
                            "config_id": row.get("config_id"),
                        }

            models_payload[name] = model_manifest
            write_json(checkpoint_path, manifest)
        return model_manifest

    for idx, name in enumerate(selected):
        gpu = gpu_pool[idx % len(gpu_pool)] if gpu_pool else None
        mm = _run_one(name, gpu)
        if mm is not None:
            models_payload[name] = mm
            write_json(checkpoint_path, manifest)

    print(f"\n[done] wrote {checkpoint_path}")


if __name__ == "__main__":
    main()
