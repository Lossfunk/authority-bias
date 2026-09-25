from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd
import torch

from src.lexical_controls.scoring import _tokenize_for_scoring, get_ab_token_ids, score_prompt_forced_choice
from src.mechanism.collect import extract_token_positions
from src.mechanism.common import InterventionResult, PanelRow, read_panel_parquet
from src.mechanism.hooks import InterventionSpec, ModelHookRunner
from src.mechanism.localize import _activation_lookup, _load_manifest, _load_shard, _tensor_for, build_pairs
from src.models.llama_loader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run causal interventions using localized candidates")
    parser.add_argument("--panel-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--localization-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--contrast", type=str, default="instruction_amplification")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--intervention-type", type=str, default="replace")
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def _row_lookup(rows: Sequence[PanelRow]) -> Dict[Tuple[str, str], PanelRow]:
    return {(row.uid, row.condition_id): row for row in rows}


def _load_localization(path: Path, contrast: str, top_k: int) -> List[Dict[str, object]]:
    if path.suffix == ".json":
        with path.open() as f:
            rows = json.load(f)
    else:
        rows = pd.read_parquet(path).to_dict(orient="records")
    filtered = [row for row in rows if row.get("contrast") == contrast]
    return filtered[:top_k]


def run_interventions(
    *,
    panel_rows: Sequence[PanelRow],
    manifest_path: Path,
    localization_path: Path,
    output_dir: Path,
    model_name: str,
    contrast: str,
    top_k: int,
    device: str,
    dtype: str,
    intervention_type: str,
    alpha: float,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(model_name, device=device, dtype=dtype)
    runner = ModelHookRunner(model)
    token_id_a, token_id_b = get_ab_token_ids(tokenizer)

    manifest = _load_manifest(manifest_path)
    lookup = _activation_lookup(manifest)
    shard_cache: Dict[str, Dict[str, object]] = {}
    localization_rows = _load_localization(localization_path, contrast, top_k)
    row_lookup = _row_lookup(panel_rows)
    pair_specs = [spec for spec in build_pairs(panel_rows) if spec.contrast == contrast]

    results: List[InterventionResult] = []
    for candidate in localization_rows:
        layer_idx = int(candidate["layer_idx"])
        component = str(candidate["component"])
        position_label = str(candidate["position_label"])
        authority = str(candidate["authority"])
        evidence_strength = str(candidate["evidence_strength"])
        candidate_pairs = [
            spec for spec in pair_specs
            if spec.authority == authority and spec.evidence_strength == evidence_strength
        ]
        for spec in candidate_pairs:
            if spec.base_key not in row_lookup or spec.source_key not in row_lookup:
                continue
            base_row = row_lookup[spec.base_key]
            base_inputs = {
                key: value.to(next(model.parameters()).device)
                for key, value in _tokenize_for_scoring(tokenizer, base_row.prompt_text).items()
            }
            positions = extract_token_positions(tokenizer, base_row)
            source_tensor = _tensor_for(key=spec.source_key, lookup=lookup, shard_cache=shard_cache)
            components = list(manifest.iloc[0]["components"])
            positions_order = list(manifest.iloc[0]["positions"])
            comp_idx = components.index(component)
            pos_idx = positions_order.index(position_label)
            source_vector = source_tensor[layer_idx, comp_idx, pos_idx]

            base_score = score_prompt_forced_choice(
                model,
                tokenizer,
                base_row.prompt_text,
                next(model.parameters()).device,
                token_id_a,
                token_id_b,
            )
            outputs = runner.intervene(
                model_inputs=base_inputs,
                token_positions=positions,
                specs=[
                    InterventionSpec(
                        layer_idx=layer_idx,
                        component=component,
                        position_label=position_label,
                        intervention_type=intervention_type,
                        source_tensor=source_vector,
                        alpha=alpha,
                    )
                ],
            )
            logits = outputs.logits[0, -1]
            logp = torch.log_softmax(logits, dim=-1)
            logp_a = float(logp[token_id_a].item())
            logp_b = float(logp[token_id_b].item())
            prob_a = float(torch.exp(logp[token_id_a]).item())
            prob_b = float(torch.exp(logp[token_id_b]).item())
            total = prob_a + prob_b
            patched_fc_a = prob_a / total if total > 0 else 0.5
            patched_fc_b = prob_b / total if total > 0 else 0.5
            if base_row.correct_label == "A":
                patched_logit_correct = logp_a - logp_b
                patched_fc_correct = patched_fc_a
                base_logit_correct = base_score.logit_a
                base_fc_correct = base_score.fc_a
            else:
                patched_logit_correct = logp_b - logp_a
                patched_fc_correct = patched_fc_b
                base_logit_correct = base_score.logit_b
                base_fc_correct = base_score.fc_b
            results.append(
                InterventionResult(
                    run_id=f"{contrast}_{layer_idx}_{component}_{position_label}",
                    model=model_name,
                    example_id=base_row.uid,
                    intervention_type=intervention_type,
                    base_condition_id=base_row.condition_id,
                    source_condition_id=spec.source_key[1],
                    component=component,
                    layer_idx=layer_idx,
                    position_label=position_label,
                    alpha=alpha,
                    effect_logit_delta=patched_logit_correct - base_logit_correct,
                    effect_fc_delta=patched_fc_correct - base_fc_correct,
                    base_logit_correct=base_logit_correct,
                    patched_logit_correct=patched_logit_correct,
                    base_fc_correct=base_fc_correct,
                    patched_fc_correct=patched_fc_correct,
                    control_group=contrast,
                    metadata={
                        "authority": authority,
                        "evidence_strength": evidence_strength,
                        "signal_score": float(candidate["signal_score"]),
                    },
                )
            )

    frame = pd.DataFrame([result.to_record() for result in results])
    out_path = output_dir / f"{contrast}_interventions.parquet"
    frame.to_parquet(out_path, index=False)
    summary = {
        "output_path": str(out_path),
        "n_results": int(len(frame)),
        "contrast": contrast,
        "top_k": top_k,
        "intervention_type": intervention_type,
        "alpha": alpha,
    }
    with (output_dir / f"{contrast}_intervention_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_interventions(
        panel_rows=read_panel_parquet(args.panel_path),
        manifest_path=args.manifest_path,
        localization_path=args.localization_path,
        output_dir=args.output_dir,
        model_name=args.model,
        contrast=args.contrast,
        top_k=args.top_k,
        device=args.device,
        dtype=args.dtype,
        intervention_type=args.intervention_type,
        alpha=args.alpha,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
