from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.mechanism.common import PanelRow, bootstrap_mean_ci, read_panel_parquet


@dataclass(frozen=True)
class PairSpec:
    contrast: str
    base_key: Tuple[str, str]
    source_key: Tuple[str, str]
    authority: str
    evidence_strength: str
    prior_state: str
    confidence_slice: str
    target_delta: float
    matched: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localization analysis over collected activations")
    parser.add_argument("--panel-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def _load_manifest(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _load_shard(shard_path: str) -> Dict[str, object]:
    return torch.load(shard_path, map_location="cpu", weights_only=False)


def _activation_lookup(manifest: pd.DataFrame) -> Dict[Tuple[str, str], Tuple[str, int]]:
    lookup: Dict[Tuple[str, str], Tuple[str, int]] = {}
    for record in manifest.to_dict(orient="records"):
        lookup[(str(record["example_id"]), str(record["condition_id"]))] = (
            str(record["shard_path"]),
            int(record["shard_index"]),
        )
    return lookup


def _tensor_for(
    *,
    key: Tuple[str, str],
    lookup: Mapping[Tuple[str, str], Tuple[str, int]],
    shard_cache: Dict[str, Dict[str, object]],
) -> torch.Tensor:
    shard_path, shard_index = lookup[key]
    if shard_path not in shard_cache:
        shard_cache[shard_path] = _load_shard(shard_path)
    payload = shard_cache[shard_path]
    activations = payload["activations"]
    assert isinstance(activations, torch.Tensor)
    return activations[shard_index].to(torch.float32)


def _group_rows(rows: Sequence[PanelRow]) -> Dict[Tuple[str, str, str, str, str], Dict[str, PanelRow]]:
    grouped: Dict[Tuple[str, str, str, str, str], Dict[str, PanelRow]] = defaultdict(dict)
    for row in rows:
        grouped[(row.uid, row.authority, row.evidence_strength, row.prior_state, row.confidence_slice)][row.condition_id] = row
    return grouped


def _condition_id(authority: str, instruction: str, direction: str, evidence_strength: str) -> str:
    prefix = {"neutral": "N", "misleading": "W", "corrective": "C"}[direction]
    return f"{prefix}{instruction}_{authority.lower().replace(' ', '_')}_{evidence_strength}"


def build_pairs(rows: Sequence[PanelRow]) -> List[PairSpec]:
    grouped = _group_rows(rows)
    pair_specs: List[PairSpec] = []
    panel_index: Dict[Tuple[str, str, str], List[PanelRow]] = defaultdict(list)
    for row in rows:
        panel_index[(row.uid, row.authority, row.evidence_strength)].append(row)

    for (uid, authority, evidence_strength, prior_state, confidence_slice), condition_rows in grouped.items():
        if prior_state != "prior_wrong":
            continue
        corrective_i0 = condition_rows.get(_condition_id(authority, "0", "corrective", evidence_strength))
        corrective_i1 = condition_rows.get(_condition_id(authority, "1", "corrective", evidence_strength))
        misleading_i1 = condition_rows.get(_condition_id(authority, "1", "misleading", evidence_strength))
        if corrective_i0 and corrective_i1:
            pair_specs.append(
                PairSpec(
                    contrast="instruction_amplification",
                    base_key=(uid, corrective_i0.condition_id),
                    source_key=(uid, corrective_i1.condition_id),
                    authority=authority,
                    evidence_strength=evidence_strength,
                    prior_state=prior_state,
                    confidence_slice=confidence_slice,
                    target_delta=corrective_i1.target_metric_metadata["logit_correct"] - corrective_i0.target_metric_metadata["logit_correct"],
                    matched=False,
                )
            )
        if corrective_i1 and misleading_i1:
            pair_specs.append(
                PairSpec(
                    contrast="direction_conflict",
                    base_key=(uid, misleading_i1.condition_id),
                    source_key=(uid, corrective_i1.condition_id),
                    authority=authority,
                    evidence_strength=evidence_strength,
                    prior_state=prior_state,
                    confidence_slice=confidence_slice,
                    target_delta=corrective_i1.target_metric_metadata["logit_correct"] - misleading_i1.target_metric_metadata["logit_correct"],
                    matched=False,
                )
            )
        if corrective_i1:
            for control_uid in corrective_i1.matched_control_uids:
                control_key = (control_uid, authority, evidence_strength, "prior_correct", confidence_slice)
                control_rows = grouped.get(control_key) or grouped.get((control_uid, authority, evidence_strength, "prior_correct", "other"))
                if not control_rows:
                    continue
                control_corrective = control_rows.get(_condition_id(authority, "1", "corrective", evidence_strength))
                if control_corrective is None:
                    continue
                pair_specs.append(
                    PairSpec(
                        contrast="matched_control",
                        base_key=(control_uid, control_corrective.condition_id),
                        source_key=(uid, corrective_i1.condition_id),
                        authority=authority,
                        evidence_strength=evidence_strength,
                        prior_state=prior_state,
                        confidence_slice=confidence_slice,
                        target_delta=corrective_i1.target_metric_metadata["logit_correct"] - control_corrective.target_metric_metadata["logit_correct"],
                        matched=True,
                    )
                )
    return pair_specs


def localize(
    *,
    panel_rows: Sequence[PanelRow],
    manifest_path: Path,
    output_dir: Path,
    top_k: int,
) -> Dict[str, object]:
    manifest = _load_manifest(manifest_path)
    lookup = _activation_lookup(manifest)
    pair_specs = build_pairs(panel_rows)
    if not pair_specs:
        raise ValueError("No valid contrast pairs could be constructed from the panel")

    shard_cache: Dict[str, Dict[str, object]] = {}
    components = list(manifest.iloc[0]["components"])
    positions = list(manifest.iloc[0]["positions"])
    stats_rows: List[Dict[str, object]] = []

    grouped_pairs: Dict[Tuple[str, str, str], List[PairSpec]] = defaultdict(list)
    for spec in pair_specs:
        grouped_pairs[(spec.contrast, spec.authority, spec.evidence_strength)].append(spec)

    for (contrast, authority, evidence_strength), specs in grouped_pairs.items():
        deltas = []
        targets = []
        for spec in specs:
            if spec.base_key not in lookup or spec.source_key not in lookup:
                continue
            base_tensor = _tensor_for(key=spec.base_key, lookup=lookup, shard_cache=shard_cache)
            source_tensor = _tensor_for(key=spec.source_key, lookup=lookup, shard_cache=shard_cache)
            deltas.append((source_tensor - base_tensor).numpy())
            targets.append(spec.target_delta)
        if not deltas:
            continue
        delta_arr = np.stack(deltas, axis=0)
        targets_arr = np.asarray(targets, dtype=float)
        mean_delta = delta_arr.mean(axis=0)
        n_layers, n_components, n_positions, _ = mean_delta.shape

        for layer_idx in range(n_layers):
            for comp_idx, component in enumerate(components):
                for pos_idx, position_label in enumerate(positions):
                    vecs = delta_arr[:, layer_idx, comp_idx, pos_idx, :]
                    mean_vec = mean_delta[layer_idx, comp_idx, pos_idx, :]
                    delta_norms = np.linalg.norm(vecs, axis=1)
                    mean_norm = float(np.linalg.norm(mean_vec))
                    if mean_norm == 0.0:
                        cosine_consistency = 0.0
                    else:
                        denom = np.linalg.norm(vecs, axis=1) * mean_norm
                        cosine_terms = np.divide(
                            vecs @ mean_vec,
                            np.where(denom == 0.0, 1.0, denom),
                        )
                        cosine_consistency = float(np.mean(cosine_terms))
                    projected = vecs @ mean_vec
                    if projected.std() > 0 and targets_arr.std() > 0:
                        corr = float(np.corrcoef(projected, targets_arr)[0, 1])
                    else:
                        corr = 0.0
                    mean_ci = bootstrap_mean_ci(delta_norms.tolist(), n_boot=1000, seed=42)
                    score = mean_norm * max(cosine_consistency, 0.0) * (1.0 + abs(corr))
                    stats_rows.append(
                        {
                            "contrast": contrast,
                            "authority": authority,
                            "evidence_strength": evidence_strength,
                            "layer_idx": layer_idx,
                            "component": component,
                            "position_label": position_label,
                            "n_pairs": int(len(vecs)),
                            "delta_norm_mean": mean_ci["mean"],
                            "delta_norm_ci95_low": mean_ci["ci95_low"],
                            "delta_norm_ci95_high": mean_ci["ci95_high"],
                            "mean_vector_norm": mean_norm,
                            "cosine_consistency": cosine_consistency,
                            "target_correlation": corr,
                            "signal_score": score,
                        }
                    )

    stats_df = pd.DataFrame(stats_rows).sort_values(["contrast", "signal_score"], ascending=[True, False])
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "localization_scores.parquet"
    stats_df.to_parquet(stats_path, index=False)

    shortlist = (
        stats_df.groupby("contrast", group_keys=False)
        .head(top_k)
        .to_dict(orient="records")
    )
    shortlist_path = output_dir / "localization_shortlist.json"
    with shortlist_path.open("w") as f:
        json.dump(shortlist, f, indent=2)

    summary = {
        "manifest_path": str(manifest_path),
        "stats_path": str(stats_path),
        "shortlist_path": str(shortlist_path),
        "n_pairs": len(pair_specs),
        "n_stats_rows": int(len(stats_df)),
        "top_k": top_k,
    }
    with (output_dir / "localization_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def main() -> None:
    args = parse_args()
    summary = localize(
        panel_rows=read_panel_parquet(args.panel_path),
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
