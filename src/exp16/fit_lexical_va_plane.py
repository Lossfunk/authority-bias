from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from src.exp16.run_steering_test import _direction_from_store, _load_extraction_store, _load_masks
from src.exp16.va_subspace import (
    cosine_np,
    fraction_of_norm_in_basis,
    normalize_np,
    orthonormalize_np,
    projection_coords_np,
)
from src.mechanism.hooks_v2 import SelectiveLayerCapture, setup_h100_optimizations
from src.models.llama_loader import load_model_and_tokenizer


DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_EXTRACTION_DIR = Path("neurips-results/mechanism/gpt_oss_authority_activations")
DEFAULT_ANALYSIS_DIR = Path("neurips-results/mechanism/gpt_oss_compliance_analysis")
DEFAULT_OUTPUT_DIR = Path("neurips-results/mechanism/lexical_va_plane")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit a cheap lexical valence-arousal plane and measure overlap with the compliance direction.")
    p.add_argument("--lexicon-path", type=Path, required=True)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--target-layers", type=str, default="16,18")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-words", type=int, default=4000)
    p.add_argument("--pca-components", type=int, default=16)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--prompt-template", type=str, default="{word}")
    p.add_argument("--position-name", type=str, default="last_token")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extraction-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    p.add_argument("--masks-path", type=Path, default=None)
    p.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument(
        "--skip-compliance-overlap",
        action="store_true",
        help="Fit and save the lexical VA plane without loading authority activations to measure overlap.",
    )
    p.add_argument("--compliance-position", type=str, default="auto")
    p.add_argument(
        "--compliance-kind",
        type=str,
        default="shared_within_label",
        choices=(
            "pooled_authority",
            "content",
            "endorsement_presence",
            "authority_given_correct",
            "authority_given_wrong",
            "shared_within_label",
        ),
    )
    p.add_argument("--compliance-uids-subset", choices=("all", "w1", "c1"), default="all")
    return p.parse_args()


def _parse_layers(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _normalize_header(text: str) -> str:
    return text.strip().lower().replace(".", "_").replace("-", "_").replace(" ", "_")


def _pick_field(fieldnames: Sequence[str], aliases: Sequence[str]) -> str:
    normalized = {_normalize_header(name): name for name in fieldnames}
    for alias in aliases:
        key = _normalize_header(alias)
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Could not find any of {aliases} in header {list(fieldnames)}")


def _load_lexicon(path: Path, max_words: int, seed: int) -> List[Tuple[str, float, float]]:
    text = path.read_text()
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel_tab
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if reader.fieldnames is None:
        raise ValueError(f"Could not parse header in {path}")

    word_key = _pick_field(reader.fieldnames, ("word", "term", "token"))
    valence_key = _pick_field(reader.fieldnames, ("valence", "v_mean_sum", "v_mean"))
    arousal_key = _pick_field(reader.fieldnames, ("arousal", "a_mean_sum", "a_mean"))

    rows: List[Tuple[str, float, float]] = []
    for row in reader:
        word = str(row[word_key]).strip()
        if not word:
            continue
        try:
            valence = float(row[valence_key])
            arousal = float(row[arousal_key])
        except (TypeError, ValueError):
            continue
        rows.append((word, valence, arousal))

    if not rows:
        raise ValueError(f"No lexical rows parsed from {path}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    rows = [rows[idx] for idx in order]
    if max_words > 0:
        rows = rows[:max_words]
    return rows


def _extract_layer_activations(
    *,
    model,
    tokenizer,
    prompts: Sequence[str],
    target_layers: Sequence[int],
    batch_size: int,
    position_name: str,
) -> Dict[int, np.ndarray]:
    captures: Dict[int, List[torch.Tensor]] = {int(layer): [] for layer in target_layers}
    for start in range(0, len(prompts), batch_size):
        batch_prompts = list(prompts[start : start + batch_size])
        model_inputs = tokenizer(batch_prompts, padding=True, truncation=False, return_tensors="pt").to(model.device)
        attention_mask = model_inputs["attention_mask"]
        last_idx = attention_mask.sum(dim=1) - 1
        positions = [{position_name: int(idx)} for idx in last_idx.cpu().tolist()]

        capture = SelectiveLayerCapture(
            model=model,
            target_layers=list(target_layers),
            save_dtype=torch.float32,
            pin_memory=False,
        )
        with capture:
            with torch.inference_mode():
                model(**model_inputs, use_cache=False)

        extracted = capture.extract_positions(
            positions_per_item=positions,
            batch_size=len(batch_prompts),
            position_names=(position_name,),
        )[position_name]
        capture.clear()

        for layer_offset, layer in enumerate(target_layers):
            captures[int(layer)].append(extracted[:, layer_offset, :].float().cpu())

    return {
        layer: torch.cat(chunks, dim=0).numpy().astype(np.float32, copy=False)
        for layer, chunks in captures.items()
    }


def _fit_plane_for_layer(
    x: np.ndarray,
    valence: np.ndarray,
    arousal: np.ndarray,
    n_components: int,
    ridge_alpha: float,
    seed: int,
) -> Dict[str, object]:
    x_centered = x.astype(np.float64, copy=False) - x.mean(axis=0, keepdims=True)
    y_v = valence.astype(np.float64, copy=False) - float(valence.mean())
    y_a = arousal.astype(np.float64, copy=False) - float(arousal.mean())

    k = max(1, min(n_components, x_centered.shape[0] - 1, x_centered.shape[1]))
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    z = pca.fit_transform(x_centered)

    ridge_v = Ridge(alpha=ridge_alpha, fit_intercept=False)
    ridge_a = Ridge(alpha=ridge_alpha, fit_intercept=False)
    ridge_v.fit(z, y_v)
    ridge_a.fit(z, y_a)

    raw_valence = pca.components_.T @ ridge_v.coef_
    raw_arousal = pca.components_.T @ ridge_a.coef_
    basis = orthonormalize_np([raw_valence, raw_arousal])
    if basis.shape[0] == 0:
        raise ValueError("VA basis fit collapsed to zero vectors.")

    valence_vec = basis[0]
    arousal_vec = basis[1] if basis.shape[0] > 1 else np.zeros_like(valence_vec)
    valence_scores = x_centered @ valence_vec
    arousal_scores = x_centered @ arousal_vec

    return {
        "basis": basis,
        "valence_vector": valence_vec,
        "arousal_vector": arousal_vec,
        "valence_corr": cosine_np(valence_scores, y_v),
        "arousal_corr": cosine_np(arousal_scores, y_a),
        "valence_r2_pc": float(ridge_v.score(z, y_v)),
        "arousal_r2_pc": float(ridge_a.score(z, y_a)),
        "pre_orthogonal_cosine": cosine_np(normalize_np(raw_valence), normalize_np(raw_arousal)),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def _resolve_compliance_position(args: argparse.Namespace) -> str:
    if args.compliance_position != "auto":
        return args.compliance_position
    summary_path = args.analysis_dir / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text())
        primary_pair = payload.get("primary_pair", {})
        position = primary_pair.get("position")
        if isinstance(position, str) and position:
            return position
    primary_direction_path = args.analysis_dir / "primary_direction.pt"
    if primary_direction_path.exists():
        payload = torch.load(primary_direction_path, map_location="cpu")
        if isinstance(payload, dict):
            position = payload.get("position")
            if isinstance(position, str) and position:
                return position
    return "endorsement_end"


def _load_subset_uids(masks: Dict[str, object], subset: str) -> List[str]:
    if subset == "w1":
        return list(masks["mask_primary_w1"])
    if subset == "c1":
        return list(masks["mask_primary_c1"])
    return list(masks["mask_primary"])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lexicon_rows = _load_lexicon(args.lexicon_path, max_words=args.max_words, seed=args.seed)
    prompts = [args.prompt_template.format(word=word) for word, _, _ in lexicon_rows]
    valence = np.asarray([row[1] for row in lexicon_rows], dtype=np.float32)
    arousal = np.asarray([row[2] for row in lexicon_rows], dtype=np.float32)
    target_layers = _parse_layers(args.target_layers)

    if torch.cuda.is_available():
        setup_h100_optimizations()
    model, tokenizer = load_model_and_tokenizer(model_name=args.model, device="auto", dtype="auto")

    activations_by_layer = _extract_layer_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        target_layers=target_layers,
        batch_size=args.batch_size,
        position_name=args.position_name,
    )

    extraction_dir = args.extraction_dir
    masks_path = args.masks_path or (extraction_dir / "label_masks.json")
    store = None
    masks = None
    compliance_uids: List[str] = []
    compliance_position = _resolve_compliance_position(args)
    if not args.skip_compliance_overlap:
        store = _load_extraction_store(extraction_dir)
        masks = _load_masks(masks_path)
        compliance_uids = _load_subset_uids(masks, args.compliance_uids_subset)

    summary_layers: Dict[str, object] = {}
    basis_by_layer: Dict[int, torch.Tensor] = {}
    valence_by_layer: Dict[int, torch.Tensor] = {}
    arousal_by_layer: Dict[int, torch.Tensor] = {}

    for layer in target_layers:
        fit = _fit_plane_for_layer(
            x=activations_by_layer[layer],
            valence=valence,
            arousal=arousal,
            n_components=args.pca_components,
            ridge_alpha=args.ridge_alpha,
            seed=args.seed + layer,
        )
        basis = fit["basis"]
        basis_by_layer[layer] = torch.tensor(basis, dtype=torch.float32)
        valence_by_layer[layer] = torch.tensor(fit["valence_vector"], dtype=torch.float32)
        arousal_by_layer[layer] = torch.tensor(fit["arousal_vector"], dtype=torch.float32)

        compliance_overlap = None
        if store is not None:
            compliance_vec = _direction_from_store(
                store,
                uids=compliance_uids,
                position=compliance_position,
                layer_value=layer,
            )[args.compliance_kind].float().numpy()
            coords = projection_coords_np(compliance_vec, basis)
            compliance_overlap = {
                "position": compliance_position,
                "direction_kind": args.compliance_kind,
                "uids_subset": args.compliance_uids_subset,
                "cosine_valence": cosine_np(compliance_vec, fit["valence_vector"]),
                "cosine_arousal": cosine_np(compliance_vec, fit["arousal_vector"]),
                "plane_norm_fraction": fraction_of_norm_in_basis(compliance_vec, basis),
                "plane_coords": coords.tolist(),
            }
        summary_layers[str(layer)] = {
            "valence_corr": fit["valence_corr"],
            "arousal_corr": fit["arousal_corr"],
            "valence_r2_pc": fit["valence_r2_pc"],
            "arousal_r2_pc": fit["arousal_r2_pc"],
            "pre_orthogonal_cosine": fit["pre_orthogonal_cosine"],
            "explained_variance_ratio": fit["explained_variance_ratio"],
            "compliance_overlap": compliance_overlap,
        }

    payload = {
        "model": args.model,
        "lexicon_path": str(args.lexicon_path),
        "position_name": args.position_name,
        "prompt_template": args.prompt_template,
        "target_layers": target_layers,
        "n_words": len(lexicon_rows),
        "basis_by_layer": basis_by_layer,
        "valence_by_layer": valence_by_layer,
        "arousal_by_layer": arousal_by_layer,
        "summary_by_layer": summary_layers,
    }
    torch.save(payload, args.output_dir / "va_plane.pt")

    summary = {
        "model": args.model,
        "lexicon_path": str(args.lexicon_path),
        "output_dir": str(args.output_dir),
        "position_name": args.position_name,
        "prompt_template": args.prompt_template,
        "n_words": len(lexicon_rows),
        "target_layers": target_layers,
        "compliance_position": compliance_position,
        "compliance_kind": args.compliance_kind,
        "summary_by_layer": summary_layers,
        "files": {
            "va_plane": str(args.output_dir / "va_plane.pt"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
