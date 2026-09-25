"""Exp12: Persistence/Washout experiment with batched scoring.

Design:
- Initial turn per tag/instruction/history: neutral, wrong, correct.
- Follow-up turns remove endorsement and re-ask with probe styles:
  same wording, paraphrase, and label-swap.
- Fresh probes are branch-independent; identical prompt batches are memoized.
- Instruction timing can be scheduled at T0/T1/T2 for prevention-vs-cure tests.
- Compare wrong-history vs neutral-history at each turn to estimate
  residual endorsement carryover after endorsement text is gone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from tqdm import tqdm

from src.lexical_controls.dataset_mc import MCExample, create_mc_examples, load_mc_dataset, save_mc_dataset
from src.lexical_controls.scoring import (
    ForcedChoiceResult,
    get_ab_token_ids,
    score_prompts_forced_choice_batch,
)
from src.endorsement.conditions import INSTRUCTION_TEXT
from src.repeated_endorsement.conditions import (
    DEFAULT_TAGS,
    EXTENDED_TAGS,
    HISTORY_CODE,
    BranchKey,
    InstructionSchedule,
    StyleProfile,
    format_fresh_probe_prompt,
    format_initial_prompt,
    format_probe_context_block,
    initial_context_from_prompt,
    normalize_tag,
    parse_instruction_schedules,
    parse_probe_modes,
    parse_probe_styles,
    parse_style_profile,
    probe_correct_label,
)
from src.models.llama_loader import load_model_and_tokenizer


DEFAULT_MODELS: Tuple[str, ...] = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B",
)
EXP12_CODE_VERSION = "2026-02-20-v2"
RESUME_CHOICES = ("auto", "never", "require")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp12: persistence/washout test")
    parser.add_argument("--models", type=str, nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--data-path", type=Path, default=Path("data/answer.jsonl"))
    parser.add_argument("--mc-dataset-path", type=Path, default=Path("data/exp7_mc_dataset.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/repeated_endorsement"))
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extended-tags", action="store_true")
    parser.add_argument("--probe-modes", type=str, default="context,fresh")
    parser.add_argument("--probe-styles", type=str, default="same,paraphrase,swap")
    parser.add_argument(
        "--style-profile",
        type=str,
        default="matched",
        help="Probe wording profile: matched (default) or legacy.",
    )
    parser.add_argument(
        "--instruction-schedules",
        type=str,
        default="none,t0",
        help="Comma-separated instruction timing schedules: none,t0,t1,t2",
    )
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=0, help="Optional max prompt length for scoring")
    parser.add_argument(
        "--use-torch-compile",
        action="store_true",
        help="Enable torch.compile for inference. Falls back to eager mode on failure.",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable TF32 where supported.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        choices=RESUME_CHOICES,
        default="auto",
        help="Resume behavior: auto|never|require",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Write checkpoint after this many completed stage scores (0 disables periodic checkpoints).",
    )
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help="Allow resuming from checkpoint artifacts with a changed run config fingerprint.",
    )
    parser.add_argument(
        "--no-correct-history",
        action="store_true",
        help="Skip correct-history branches in follow-up turns (faster, less complete).",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Include full prompt text in per-example JSONL (large files).",
    )
    return parser.parse_args()


def _resolve_data_path(path: Path) -> Path:
    if path.exists():
        return path
    fallback = Path("external/sycophancy-eval/datasets/answer.jsonl")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Dataset not found at '{path}' or '{fallback}'.")


def _model_tag(model_id: str) -> str:
    return model_id.replace("/", "__")


def _is_qwen_model(model_id: str) -> bool:
    return "qwen" in model_id.lower()


def _order_seed(uid: str, seed: int) -> int:
    digest = hashlib.md5(f"{seed}:{uid}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _stable_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _fingerprint(config: Dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(config).encode("utf-8")).hexdigest()


def _dataset_id(examples: Sequence[MCExample]) -> str:
    joined = "\n".join(ex.uid for ex in examples)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _atomic_write_records(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _load_checkpoint_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("rb") as f:
        for idx, raw in enumerate(f):
            text = raw.decode("utf-8", errors="strict").strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid checkpoint record type at {path}:{idx + 1}")
            rows.append(payload)
    return rows


def _init_records(
    *,
    model_id: str,
    examples: Sequence[MCExample],
    seed: int,
    save_prompts: bool,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for ex in examples:
        rec = {
            "model": model_id,
            "uid": ex.uid,
            "correct_label": ex.correct_label,
            "wrong_label": ex.wrong_label,
            "order_seed": _order_seed(ex.uid, seed),
            "metadata": ex.metadata,
            "init_results": {},
            "probe_results": {},
        }
        if save_prompts:
            rec["prompts"] = {}
        records.append(rec)
    return records


def _is_stage_complete(records: Sequence[Dict[str, Any]], section: str, code: str) -> bool:
    if not records:
        return False
    for rec in records:
        payload = rec.get(section, {})
        if not isinstance(payload, dict) or code not in payload:
            return False
    return True


def _write_checkpoint(
    *,
    path: Path,
    model_id: str,
    fingerprint: str,
    started_at: str,
    total_examples: int,
    completed_stages: int,
    total_stages: int,
    resume_mode: str,
    effective_batch_size: int,
    completed: bool,
) -> None:
    payload = {
        "experiment": "exp12_persistence_washout",
        "code_version": EXP12_CODE_VERSION,
        "model": model_id,
        "fingerprint": fingerprint,
        "started_at": started_at,
        "updated_at": _utc_now_iso(),
        "total_examples": total_examples,
        "completed_stages": completed_stages,
        "total_stages": total_stages,
        "remaining_stages": max(total_stages - completed_stages, 0),
        "resume_mode": resume_mode,
        "effective_batch_size": effective_batch_size,
        "completed": completed,
    }
    _atomic_write_json(path, payload)


def _pred_label(result: ForcedChoiceResult) -> str:
    return "A" if result.fc_a >= result.fc_b else "B"


def _result_to_entry(result: ForcedChoiceResult, expected_correct_label: str) -> Dict[str, float | str]:
    if expected_correct_label == "A":
        fc_correct = result.fc_a
        fc_wrong = result.fc_b
        margin_correct = result.logit_a
        margin_wrong = result.logit_b
    else:
        fc_correct = result.fc_b
        fc_wrong = result.fc_a
        margin_correct = result.logit_b
        margin_wrong = result.logit_a

    return {
        "logit_a": result.logit_a,
        "logit_b": result.logit_b,
        "fc_a": result.fc_a,
        "fc_b": result.fc_b,
        "fc_correct": fc_correct,
        "fc_wrong": fc_wrong,
        "margin_correct": margin_correct,
        "margin_wrong": margin_wrong,
        "pred_label": _pred_label(result),
        "expected_correct_label": expected_correct_label,
    }


def _summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n": 0,
            "positive_frac": 0.0,
        }
    vals = list(values)
    n_pos = sum(1 for v in vals if v > 0)
    return {
        "mean": float(mean(vals)),
        "median": float(median(vals)),
        "std": float(stdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
        "n": len(vals),
        "positive_frac": n_pos / len(vals),
    }


def _safe_ratio(numer: Sequence[float], denom: Sequence[float], tau: float = 1e-6) -> List[float]:
    out: List[float] = []
    for n, d in zip(numer, denom):
        if abs(d) <= tau:
            continue
        out.append(n / d)
    return out


def _score_prompts(
    model,
    tokenizer,
    device: torch.device,
    token_id_a: int,
    token_id_b: int,
    prompts: List[str],
    batch_size: int,
    max_length: Optional[int],
) -> List[ForcedChoiceResult]:
    return score_prompts_forced_choice_batch(
        model=model,
        tokenizer=tokenizer,
        prompt_texts=prompts,
        device=device,
        token_id_a=token_id_a,
        token_id_b=token_id_b,
        batch_size=batch_size,
        max_length=max_length,
    )


def _configure_torch_runtime(*, tf32: bool) -> str:
    """Configure TF32 using the newest available PyTorch API."""
    if not torch.cuda.is_available():
        return "no_cuda"

    has_new_fp32_api = (
        hasattr(torch.backends, "fp32_precision")
        and hasattr(torch.backends, "cuda")
        and hasattr(torch.backends.cuda, "matmul")
        and hasattr(torch.backends.cuda.matmul, "fp32_precision")
    )
    if has_new_fp32_api:
        precision_value = "tf32" if tf32 else "ieee"
        torch.backends.fp32_precision = precision_value
        torch.backends.cuda.matmul.fp32_precision = precision_value
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "fp32_precision"):
            torch.backends.cudnn.fp32_precision = precision_value
        return "fp32_precision"

    # Fallback for older PyTorch versions.
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    return "allow_tf32"


def _extract_margins(records: List[Dict], key: str, field: str, section: str) -> List[float]:
    vals = []
    for r in records:
        vals.append(float(r[section][key][field]))
    return vals


def _compute_summary(
    records: List[Dict],
    tags: List[str],
    modes: List[str],
    styles: List[str],
    instruction_schedules: List[InstructionSchedule],
    turns: int,
    include_correct_history: bool,
) -> Dict:
    summary: Dict[str, Dict] = {}
    for tag in tags:
        tag_key = normalize_tag(tag)
        tag_out: Dict[str, Dict] = {}
        for instruction in instruction_schedules:
            instr_key = instruction.key
            i_key = f"instr_{instr_key}"
            n_code = f"N{instr_key}_{tag_key}"
            w_code = f"W{instr_key}_{tag_key}"
            c_code = f"C{instr_key}_{tag_key}"

            m_n = _extract_margins(records, n_code, "margin_correct", "init_results")
            m_w = _extract_margins(records, w_code, "margin_correct", "init_results")
            m_c = _extract_margins(records, c_code, "margin_correct", "init_results")

            immediate_wrong = [w - n for w, n in zip(m_w, m_n)]
            immediate_correct = [c - n for c, n in zip(m_c, m_n)]

            i_out: Dict[str, Dict] = {
                "initial": {
                    "neutral_margin": _summarize(m_n),
                    "wrong_margin": _summarize(m_w),
                    "correct_margin": _summarize(m_c),
                    "immediate_wrong_shift": _summarize(immediate_wrong),
                    "immediate_correct_shift": _summarize(immediate_correct),
                    "wrong_prior_frac_neutral": float(sum(1 for v in m_n if v < 0) / len(m_n)) if m_n else 0.0,
                },
                "probes": {},
            }

            for mode in modes:
                mode_prefix = "CTX" if mode == "context" else "FRESH"
                mode_out: Dict[str, Dict] = {}
                for style in styles:
                    s_out: Dict[str, Dict] = {}
                    for turn in range(1, turns + 1):
                        key_n = f"{mode_prefix}_HN{instr_key}_{tag_key}_{style}_T{turn}"
                        key_w = f"{mode_prefix}_HW{instr_key}_{tag_key}_{style}_T{turn}"
                        m_hn = _extract_margins(records, key_n, "margin_correct", "probe_results")
                        m_hw = _extract_margins(records, key_w, "margin_correct", "probe_results")
                        residual_wrong = [w - n for w, n in zip(m_hw, m_hn)]
                        ratio_wrong = _safe_ratio(residual_wrong, immediate_wrong)
                        washout_wrong = [1.0 - min(1.0, abs(r)) for r in ratio_wrong]

                        turn_out: Dict[str, Dict | float] = {
                            "neutral_probe_margin": _summarize(m_hn),
                            "wrong_probe_margin": _summarize(m_hw),
                            "residual_wrong_shift": _summarize(residual_wrong),
                            "persistence_ratio_wrong": _summarize(ratio_wrong),
                            "washout_score_wrong": _summarize(washout_wrong),
                        }

                        wrong_prior_idx = [idx for idx, v in enumerate(m_n) if v < 0]
                        if wrong_prior_idx:
                            resid_wp = [residual_wrong[idx] for idx in wrong_prior_idx]
                            hn_wp = [m_hn[idx] for idx in wrong_prior_idx]
                            hw_wp = [m_hw[idx] for idx in wrong_prior_idx]
                            turn_out["wrong_prior_slice"] = {
                                "n": len(wrong_prior_idx),
                                "residual_wrong_shift": _summarize(resid_wp),
                                "neutral_probe_flip_rate": float(sum(1 for v in hn_wp if v > 0) / len(hn_wp)),
                                "wrong_probe_flip_rate": float(sum(1 for v in hw_wp if v > 0) / len(hw_wp)),
                            }

                        if include_correct_history:
                            key_c = f"{mode_prefix}_HC{instr_key}_{tag_key}_{style}_T{turn}"
                            m_hc = _extract_margins(records, key_c, "margin_correct", "probe_results")
                            residual_correct = [c - n for c, n in zip(m_hc, m_hn)]
                            ratio_correct = _safe_ratio(residual_correct, immediate_correct)
                            turn_out["correct_probe_margin"] = _summarize(m_hc)
                            turn_out["residual_correct_shift"] = _summarize(residual_correct)
                            turn_out["persistence_ratio_correct"] = _summarize(ratio_correct)
                            turn_out["residual_selectivity"] = _summarize(
                                [rc - rw for rc, rw in zip(residual_correct, residual_wrong)]
                            )

                        s_out[f"T{turn}"] = turn_out
                    mode_out[style] = s_out
                i_out["probes"][mode] = mode_out
            tag_out[i_key] = i_out
        summary[tag_key] = tag_out
    return summary


def run_for_model(
    model_id: str,
    examples: List[MCExample],
    output_dir: Path,
    seed: int,
    tags: List[str],
    modes: List[str],
    styles: List[str],
    style_profile: StyleProfile,
    instruction_schedules: List[InstructionSchedule],
    turns: int,
    include_correct_history: bool,
    save_prompts: bool,
    batch_size: int,
    max_length: Optional[int],
    use_torch_compile: bool,
    tf32: bool,
    resume_mode: str,
    checkpoint_every: int,
    allow_config_mismatch: bool,
) -> Dict:
    model_tag = _model_tag(model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_tag}_results.jsonl"
    summary_path = output_dir / f"{model_tag}_summary.json"
    checkpoint_path = output_dir / f"{model_tag}_checkpoint.json"
    checkpoint_records_path = output_dir / f"{model_tag}_checkpoint_records.jsonl"
    manifest_path = output_dir / f"{model_tag}_run_manifest.json"

    run_config = {
        "experiment": "exp12_persistence_washout",
        "code_version": EXP12_CODE_VERSION,
        "model": model_id,
        "seed": seed,
        "dataset_id": _dataset_id(examples),
        "n_examples": len(examples),
        "tags": list(tags),
        "probe_modes": list(modes),
        "styles": list(styles),
        "style_profile": style_profile,
        "instruction_schedule_names": [s.name for s in instruction_schedules],
        "instruction_schedule_keys": [s.key for s in instruction_schedules],
        "turns": turns,
        "include_correct_history": include_correct_history,
        "save_prompts": save_prompts,
        "max_length": max_length if max_length is not None else 0,
    }
    fingerprint = _fingerprint(run_config)

    if resume_mode != "never" and out_path.exists() and summary_path.exists():
        print(f"Detected completed Exp12 artifacts for {model_id}; loading summary from disk.")
        return _load_json(summary_path)

    has_existing = (
        out_path.exists()
        or summary_path.exists()
        or checkpoint_path.exists()
        or checkpoint_records_path.exists()
        or manifest_path.exists()
    )
    if resume_mode == "never" and has_existing:
        raise RuntimeError(
            f"Existing run artifacts found for {model_id} in {output_dir}. "
            "Use --resume auto/require or move to a new output directory."
        )
    if resume_mode == "require" and not has_existing:
        raise RuntimeError(
            f"Resume required but no existing artifacts found for model {model_id} in {output_dir}."
        )

    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        existing_fingerprint = manifest.get("fingerprint")
        if existing_fingerprint != fingerprint and not allow_config_mismatch:
            raise RuntimeError(
                "Run config fingerprint mismatch.\n"
                f"  Existing: {existing_fingerprint}\n"
                f"  Current : {fingerprint}\n"
                "Use --allow-config-mismatch only if this run config change is intentional."
            )
    else:
        if resume_mode == "require":
            raise RuntimeError(f"Missing manifest for resume-required run: {manifest_path}")
        _atomic_write_json(
            manifest_path,
            {
                "experiment": "exp12_persistence_washout",
                "code_version": EXP12_CODE_VERSION,
                "created_at": _utc_now_iso(),
                "fingerprint": fingerprint,
                "config": run_config,
            },
        )

    records = _init_records(model_id=model_id, examples=examples, seed=seed, save_prompts=save_prompts)
    started_at = _utc_now_iso()
    if checkpoint_path.exists() or checkpoint_records_path.exists():
        if not checkpoint_path.exists() or not checkpoint_records_path.exists():
            if resume_mode == "require":
                raise RuntimeError(
                    f"Incomplete checkpoint artifacts for {model_id}. "
                    f"Expected both {checkpoint_path.name} and {checkpoint_records_path.name}."
                )
        elif resume_mode != "never":
            ckpt = _load_json(checkpoint_path)
            existing_fingerprint = ckpt.get("fingerprint")
            if existing_fingerprint != fingerprint and not allow_config_mismatch:
                raise RuntimeError(
                    "Checkpoint fingerprint mismatch.\n"
                    f"  Existing: {existing_fingerprint}\n"
                    f"  Current : {fingerprint}\n"
                    "Use --allow-config-mismatch only if this run config change is intentional."
                )
            raw_records = _load_checkpoint_records(checkpoint_records_path)
            by_uid: Dict[str, Dict[str, Any]] = {}
            for rec in raw_records:
                uid = str(rec.get("uid", ""))
                if uid:
                    by_uid[uid] = rec
            missing = [ex.uid for ex in examples if ex.uid not in by_uid]
            if missing:
                raise RuntimeError(
                    f"Checkpoint records are missing {len(missing)} example(s); first missing uid={missing[0]}"
                )
            records = [by_uid[ex.uid] for ex in examples]
            started_at = str(ckpt.get("started_at", started_at))
            print(
                f"Resuming {model_id} from checkpoint: "
                f"{ckpt.get('completed_stages', 0)}/{ckpt.get('total_stages', 'unknown')} stages complete."
            )

    print(f"\n{'=' * 70}")
    print(f"Running Exp12 model: {model_id}")
    print(f"Tags: {tags}")
    print(f"Modes: {modes}")
    print(f"Instruction schedules: {[s.name for s in instruction_schedules]}")
    print(f"Styles: {styles}, turns: {turns}")
    print(f"Style profile: {style_profile}")
    print(f"TF32 enabled: {tf32}")
    print(f"Resume mode: {resume_mode}; checkpoint_every={checkpoint_every}")
    print(f"{'=' * 70}")

    tf32_control_api = _configure_torch_runtime(tf32=tf32)
    model, tokenizer = load_model_and_tokenizer(model_name=model_id, device="auto", dtype="auto")
    model.eval()
    compile_requested = use_torch_compile
    compile_disabled_reason: Optional[str] = None
    if use_torch_compile and _is_qwen_model(model_id):
        compile_disabled_reason = "disabled_for_qwen_stability"
        print(
            "WARNING: torch.compile is disabled for Qwen models in Exp12 due CUDA/Triton instability; using eager mode."
        )
        use_torch_compile = False

    compile_enabled = False
    compile_mode = os.environ.get("EXP12_TORCH_COMPILE_MODE", "reduce-overhead")
    if use_torch_compile:
        if hasattr(torch, "compile"):
            try:
                # Use a conservative default mode; max-autotune can be unstable on some kernels.
                print(f"Attempting torch.compile (mode={compile_mode})...")
                model = torch.compile(model, mode=compile_mode)
                compile_enabled = True
                print("torch.compile enabled.")
            except Exception as exc:  # pragma: no cover - runtime dependent
                print(f"WARNING: torch.compile failed, falling back to eager mode: {exc}")
        else:
            print("WARNING: torch.compile is unavailable in this PyTorch build; using eager mode.")

    device = next(model.parameters()).device
    active_batch_size = batch_size

    token_id_a, token_id_b = get_ab_token_ids(tokenizer)
    print(f"Token A: {tokenizer.decode([token_id_a])!r} (id={token_id_a})")
    print(f"Token B: {tokenizer.decode([token_id_b])!r} (id={token_id_b})")

    def _is_cuda_oom(exc: Exception) -> bool:
        if isinstance(exc, torch.OutOfMemoryError):
            return True
        msg = str(exc).lower()
        return "out of memory" in msg and "cuda" in msg

    def _is_cuda_runtime_fault(exc: Exception) -> bool:
        msg = str(exc).lower()
        if "cuda error" in msg and "out of memory" not in msg:
            return True
        markers = (
            "cudaerrorillegaladdress",
            "illegal memory access",
            "device-side assert",
            "acceleratorerror",
        )
        return any(marker in msg for marker in markers)

    def _reload_eager_model(reason: str) -> None:
        nonlocal model, device, compile_enabled
        print(reason)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, _ = load_model_and_tokenizer(model_name=model_id, device="auto", dtype="auto")
        model.eval()
        device = next(model.parameters()).device
        compile_enabled = False

    def _score_prompts_with_fallback(prompts: List[str]) -> List[ForcedChoiceResult]:
        nonlocal model, device, compile_enabled, active_batch_size
        while True:
            try:
                return _score_prompts(
                    model, tokenizer, device, token_id_a, token_id_b,
                    prompts, batch_size=active_batch_size, max_length=max_length,
                )
            except Exception as exc:
                if _is_cuda_oom(exc):
                    if compile_enabled:
                        _reload_eager_model("WARNING: CUDA OOM with compiled model; reloading eager model and retrying.")
                        continue

                    if active_batch_size > 8:
                        new_bs = max(8, active_batch_size // 2)
                        if new_bs == active_batch_size:
                            new_bs = active_batch_size - 1
                        print(f"WARNING: CUDA OOM at batch_size={active_batch_size}; retrying with batch_size={new_bs}.")
                        active_batch_size = new_bs
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    raise

                if compile_enabled and _is_cuda_runtime_fault(exc):
                    _reload_eager_model(
                        "WARNING: CUDA runtime failure with compiled model; reloading eager model and retrying."
                    )
                    continue

                raise

    init_histories = ["neutral", "wrong", "correct"]
    history_for_probes = ["neutral", "wrong"] + ([] if not include_correct_history else ["correct"])

    init_specs = []
    for tag in tags:
        tag_key = normalize_tag(tag)
        for schedule in instruction_schedules:
            for history in init_histories:
                code = f"{HISTORY_CODE[history]}{schedule.key}_{tag_key}"
                init_specs.append((tag, tag_key, schedule, history, code))

    branch_keys: List[BranchKey] = []
    if "context" in modes:
        for tag in tags:
            tag_key = normalize_tag(tag)
            for schedule in instruction_schedules:
                for history in history_for_probes:
                    for style in styles:
                        branch_keys.append(
                            BranchKey(
                                tag=tag_key,
                                history=history,  # type: ignore[arg-type]
                                instruction_key=schedule.key,
                                style=style,  # type: ignore[arg-type]
                            )
                        )

    fresh_specs: List[Tuple[str, str, InstructionSchedule, str]] = []
    if "fresh" in modes:
        for tag in tags:
            tag_key = normalize_tag(tag)
            for schedule in instruction_schedules:
                for history in history_for_probes:
                    for style in styles:
                        fresh_specs.append((tag_key, history, schedule, style))

    total_stages = len(init_specs) + len(branch_keys) * turns + len(fresh_specs) * turns
    completed_stage_ids: Set[str] = set()
    new_stages_completed = 0
    checkpoint_enabled = checkpoint_every > 0

    def _checkpoint(completed: bool = False) -> None:
        _atomic_write_records(checkpoint_records_path, records)
        _write_checkpoint(
            path=checkpoint_path,
            model_id=model_id,
            fingerprint=fingerprint,
            started_at=started_at,
            total_examples=len(examples),
            completed_stages=len(completed_stage_ids),
            total_stages=total_stages,
            resume_mode=resume_mode,
            effective_batch_size=active_batch_size,
            completed=completed,
        )

    print("\nScoring initial turn conditions...")

    init_prompt_cache: Dict[Tuple[str, str, str], List[str]] = {}
    for tag, tag_key, schedule, history, code in tqdm(init_specs, desc=f"Exp12 init {model_id}"):
        stage_id = f"init::{code}"
        prompts = [
            format_initial_prompt(ex, tag, history, instruction=schedule.init_instruction)
            for ex in examples
        ]
        init_prompt_cache[(tag_key, schedule.key, history)] = prompts
        if _is_stage_complete(records, "init_results", code):
            completed_stage_ids.add(stage_id)
            continue
        scored = _score_prompts_with_fallback(prompts)
        for i, result in enumerate(scored):
            entry = _result_to_entry(result, expected_correct_label=examples[i].correct_label)
            entry["instruction"] = schedule.key
            entry["instruction_schedule"] = schedule.name
            records[i]["init_results"][code] = entry
            if save_prompts:
                records[i]["prompts"][code] = prompts[i]
        completed_stage_ids.add(stage_id)
        new_stages_completed += 1
        if checkpoint_enabled and new_stages_completed % checkpoint_every == 0:
            _checkpoint(completed=False)

    branch_contexts: Dict[str, List[str]] = {}
    if "context" in modes:
        print("\nBuilding context-mode branch histories...")
        for b in branch_keys:
            init_code = f"{HISTORY_CODE[b.history]}{b.instruction_key}_{b.tag}"
            prompts = init_prompt_cache[(b.tag, b.instruction_key, b.history)]
            contexts = []
            for i, prompt in enumerate(prompts):
                pred = records[i]["init_results"][init_code]["pred_label"]
                contexts.append(initial_context_from_prompt(prompt, str(pred)))
            branch_contexts[b.key] = contexts

    expected_labels_by_style: Dict[str, List[str]] = {
        style: [probe_correct_label(ex, style) for ex in examples]
        for style in styles
    }
    print("\nScoring persistence/washout follow-up turns...")
    schedule_by_key = {s.key: s for s in instruction_schedules}
    for turn in range(1, turns + 1):
        if "context" in modes:
            context_block_cache: Dict[Tuple[str, bool], List[str]] = {}
            for b in tqdm(branch_keys, desc=f"Exp12 context probes {model_id} T{turn}"):
                contexts = branch_contexts[b.key]
                style = b.style
                schedule = schedule_by_key[b.instruction_key]
                include_instruction = schedule.include_probe_instruction("context", turn)
                block_cache_key = (style, include_instruction)
                context_blocks = context_block_cache.get(block_cache_key)
                if context_blocks is None:
                    context_blocks = [
                        format_probe_context_block(
                            ex,
                            style,
                            turn,
                            include_instruction=include_instruction,
                            style_profile=style_profile,
                        )
                        for ex in examples
                    ]
                    context_block_cache[block_cache_key] = context_blocks

                prompts = [f"{ctx}{block}\nAnswer:" for ctx, block in zip(contexts, context_blocks)]
                expected_labels = expected_labels_by_style[style]
                probe_code = f"CTX_{b.key}_T{turn}"
                stage_id = f"probe::{probe_code}"
                if _is_stage_complete(records, "probe_results", probe_code):
                    completed_stage_ids.add(stage_id)
                    for i in range(len(examples)):
                        pred = str(records[i]["probe_results"][probe_code]["pred_label"])
                        contexts[i] = f"{contexts[i]}{context_blocks[i]}\nAssistant: {pred}\n"
                    continue

                scored = _score_prompts_with_fallback(prompts)

                for i, result in enumerate(scored):
                    entry = _result_to_entry(result, expected_correct_label=expected_labels[i])
                    entry["mode"] = "context"
                    entry["turn"] = turn
                    entry["style"] = style
                    entry["history"] = b.history
                    entry["instruction"] = b.instruction_key
                    entry["instruction_schedule"] = schedule.name
                    entry["tag"] = b.tag
                    records[i]["probe_results"][probe_code] = entry
                    if save_prompts:
                        records[i]["prompts"][probe_code] = prompts[i]

                    pred = str(entry["pred_label"])
                    contexts[i] = f"{contexts[i]}{context_blocks[i]}\nAssistant: {pred}\n"
                completed_stage_ids.add(stage_id)
                new_stages_completed += 1
                if checkpoint_enabled and new_stages_completed % checkpoint_every == 0:
                    _checkpoint(completed=False)

        if "fresh" in modes:
            fresh_cache: Dict[Tuple[str, bool], Tuple[List[str], List[ForcedChoiceResult]]] = {}
            for tag_key, history, schedule, style in tqdm(fresh_specs, desc=f"Exp12 fresh probes {model_id} T{turn}"):
                include_instruction = schedule.include_probe_instruction("fresh", turn)
                fresh_cache_key = (style, include_instruction)
                cached = fresh_cache.get(fresh_cache_key)
                if cached is None:
                    prompts = [
                        format_fresh_probe_prompt(
                            ex,
                            style=style,
                            turn_idx=turn,
                            instruction=include_instruction,
                            style_profile=style_profile,
                        )
                        for ex in examples
                    ]
                    scored = _score_prompts_with_fallback(prompts)
                    fresh_cache[fresh_cache_key] = (prompts, scored)
                else:
                    prompts, scored = cached
                expected_labels = expected_labels_by_style[style]
                probe_code = f"FRESH_H{HISTORY_CODE[history]}{schedule.key}_{tag_key}_{style}_T{turn}"
                stage_id = f"probe::{probe_code}"
                if _is_stage_complete(records, "probe_results", probe_code):
                    completed_stage_ids.add(stage_id)
                    continue

                for i, result in enumerate(scored):
                    entry = _result_to_entry(result, expected_correct_label=expected_labels[i])
                    entry["mode"] = "fresh"
                    entry["turn"] = turn
                    entry["style"] = style
                    entry["instruction"] = schedule.key
                    entry["instruction_schedule"] = schedule.name
                    entry["tag"] = tag_key
                    entry["history"] = history
                    records[i]["probe_results"][probe_code] = entry
                    if save_prompts:
                        records[i]["prompts"][probe_code] = prompts[i]
                completed_stage_ids.add(stage_id)
                new_stages_completed += 1
                if checkpoint_enabled and new_stages_completed % checkpoint_every == 0:
                    _checkpoint(completed=False)

    summary_metrics = _compute_summary(
        records=records,
        tags=tags,
        modes=modes,
        styles=styles,
        instruction_schedules=instruction_schedules,
        turns=turns,
        include_correct_history=include_correct_history,
    )

    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    model_summary = {
        "experiment": "exp12_persistence_washout",
        "code_version": EXP12_CODE_VERSION,
        "model": model_id,
        "n_examples": len(examples),
        "tags": tags,
        "probe_modes": modes,
        "styles": styles,
        "style_profile": style_profile,
        "instruction_schedule_names": [s.name for s in instruction_schedules],
        "instruction_schedule_keys": [s.key for s in instruction_schedules],
        "instruction_schedule_labels": {s.key: s.label for s in instruction_schedules},
        "turns": turns,
        "include_correct_history": include_correct_history,
        "fresh_probe_independent_by_branch": True,
        "instruction_text": INSTRUCTION_TEXT,
        "batch_size": batch_size,
        "effective_batch_size": active_batch_size,
        "max_length": max_length if max_length is not None else 0,
        "torch_compile_requested": compile_requested,
        "torch_compile_enabled": compile_enabled,
        "torch_compile_disabled_reason": compile_disabled_reason or "",
        "tf32_enabled": tf32,
        "tf32_control_api": tf32_control_api,
        "resume_mode": resume_mode,
        "checkpoint_every": checkpoint_every,
        "checkpoint_enabled": checkpoint_enabled,
        "completed_stages": len(completed_stage_ids),
        "total_stages": total_stages,
        "run_fingerprint": fingerprint,
        "token_ids": {
            "a": token_id_a,
            "b": token_id_b,
            "a_str": tokenizer.decode([token_id_a]),
            "b_str": tokenizer.decode([token_id_b]),
        },
        "metrics": summary_metrics,
    }

    with summary_path.open("w") as f:
        json.dump(model_summary, f, indent=2)
    _checkpoint(completed=True)

    print(f"\nKey metrics for {model_id}:")
    preferred_instr_key = "1" if any(s.key == "1" for s in instruction_schedules) else instruction_schedules[0].key
    preferred_instr_metric = f"instr_{preferred_instr_key}"
    preferred_instr_label = {s.key: s.label for s in instruction_schedules}[preferred_instr_key]
    for tag in tags:
        tag_key = normalize_tag(tag)
        m = model_summary["metrics"][tag_key][preferred_instr_metric]["initial"]["immediate_wrong_shift"]["mean"]
        print(
            f"  {tag}: immediate_wrong_shift "
            f"({preferred_instr_label} mean margin delta) = {m:.4f}"
        )
        for mode in modes:
            for style in styles:
                resid = model_summary["metrics"][tag_key][preferred_instr_metric]["probes"][mode][style]["T1"]["residual_wrong_shift"]["mean"]
                print(f"    {mode} {style} T1 residual_wrong_shift: {resid:.4f}")

    del model
    torch.cuda.empty_cache()
    return model_summary


def main() -> None:
    args = parse_args()
    if args.turns <= 0:
        raise ValueError("--turns must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")

    tags = list(EXTENDED_TAGS) if args.extended_tags else list(DEFAULT_TAGS)
    modes = parse_probe_modes(args.probe_modes)
    styles = parse_probe_styles(args.probe_styles)
    style_profile = parse_style_profile(args.style_profile)
    instruction_schedules = parse_instruction_schedules(args.instruction_schedules)
    include_correct_history = not args.no_correct_history
    max_length = args.max_length if args.max_length > 0 else None

    print("Exp12: Persistence/Washout")
    print(f"Tags: {tags}")
    print(f"Probe modes: {modes}")
    print(f"Styles: {styles}")
    print(f"Style profile: {style_profile}")
    print(f"Instruction schedules: {[s.name for s in instruction_schedules]}")
    print(f"Turns: {args.turns}")
    print(f"Include correct history: {include_correct_history}")
    print(f"Use torch.compile: {args.use_torch_compile}")
    print(f"TF32: {args.tf32}")
    print(f"Resume mode: {args.resume}")
    print(f"Checkpoint every: {args.checkpoint_every}")

    mc_path = args.mc_dataset_path
    if mc_path.exists():
        print(f"Loading existing MC dataset from {mc_path}")
        examples = load_mc_dataset(mc_path)
    else:
        raw_path = _resolve_data_path(args.data_path)
        print(f"Creating MC dataset from {raw_path}")
        examples = create_mc_examples(raw_path, seed=args.seed)
        save_mc_dataset(examples, mc_path)
        print(f"Saved MC dataset to {mc_path}")

    if args.max_examples > 0 and len(examples) > args.max_examples:
        rng = random.Random(args.seed)
        shuffled = list(examples)
        rng.shuffle(shuffled)
        examples = shuffled[: args.max_examples]

    print(f"Dataset: {len(examples)} examples")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    item_table_path = args.output_dir / "item_table.jsonl"
    with item_table_path.open("w") as f:
        for ex in examples:
            item = {
                "uid": ex.uid,
                "dataset": ex.metadata.get("dataset", "unknown"),
                "correct_label": ex.correct_label,
                "wrong_label": ex.wrong_label,
                "opt_a_text": ex.option_a,
                "opt_b_text": ex.option_b,
                "order_seed": _order_seed(ex.uid, seed=args.seed),
            }
            f.write(json.dumps(item) + "\n")

    summaries = []
    for model_id in args.models:
        summary = run_for_model(
            model_id=model_id,
            examples=examples,
            output_dir=args.output_dir,
            seed=args.seed,
            tags=tags,
            modes=modes,
            styles=styles,
            style_profile=style_profile,
            instruction_schedules=instruction_schedules,
            turns=args.turns,
            include_correct_history=include_correct_history,
            save_prompts=args.save_prompts,
            batch_size=args.batch_size,
            max_length=max_length,
            use_torch_compile=args.use_torch_compile,
            tf32=args.tf32,
            resume_mode=args.resume,
            checkpoint_every=args.checkpoint_every,
            allow_config_mismatch=args.allow_config_mismatch,
        )
        summaries.append(summary)

    combined = {
        "experiment": "exp12_persistence_washout",
        "description": "Turn-wise persistence test for endorsement-induced carryover after endorsement removal.",
        "n_examples": len(examples),
        "seed": args.seed,
        "tags": tags,
        "probe_modes": modes,
        "styles": styles,
        "style_profile": style_profile,
        "instruction_schedule_names": [s.name for s in instruction_schedules],
        "instruction_schedule_keys": [s.key for s in instruction_schedules],
        "instruction_schedule_labels": {s.key: s.label for s in instruction_schedules},
        "turns": args.turns,
        "include_correct_history": include_correct_history,
        "resume_mode": args.resume,
        "checkpoint_every": args.checkpoint_every,
        "models": summaries,
    }
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Exp12 complete")
    print(f"Results saved to {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
