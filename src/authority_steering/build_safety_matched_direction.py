from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build safety-matched direction vectors from extracted N0/C1/W1 activations."
    )
    p.add_argument(
        "--extraction-dir",
        type=Path,
        required=True,
        help="Directory containing activations.pt + metadata.jsonl from extract_safety_matched_activations.py",
    )
    p.add_argument(
        "--position",
        type=str,
        default="endorsement_end",
        choices=("endorsement_start", "endorsed_answer", "endorsement_end", "answer_position", "endorsement_mean"),
    )
    p.add_argument("--layer", type=int, required=True, help="Transformer layer index (not layer slot index).")
    p.add_argument(
        "--primary-kind",
        type=str,
        default="shared_within_label",
        choices=("shared_within_label", "authority_given_safe", "authority_given_harm", "w1_minus_c1", "c1_minus_w1"),
        help="Which vector to export as `vector` in primary_direction.pt",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Default: <extraction-dir>/safety_primary_direction_<position>_L<layer>.pt",
    )
    return p.parse_args()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean_for_condition(
    *,
    activations: torch.Tensor,
    metadata: List[Dict[str, Any]],
    layer_idx: int,
    condition_code: str,
) -> torch.Tensor:
    idxs = [i for i, row in enumerate(metadata) if row.get("condition_code") == condition_code]
    if not idxs:
        raise ValueError(f"No rows found for condition_code={condition_code}")
    idx_tensor = torch.tensor(idxs, dtype=torch.long)
    mat = activations[idx_tensor, layer_idx, :].float()
    return mat.mean(dim=0)


def _normalize(v: torch.Tensor) -> torch.Tensor:
    n = v.norm()
    if float(n) == 0.0:
        return v.clone()
    return v / n


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    an = _normalize(a)
    bn = _normalize(b)
    return float(torch.dot(an, bn).item())


def main() -> None:
    args = parse_args()

    activations_path = args.extraction_dir / "activations.pt"
    metadata_path = args.extraction_dir / "metadata.jsonl"
    if not activations_path.exists():
        raise FileNotFoundError(f"Missing activations file: {activations_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    payload = torch.load(activations_path, map_location="cpu")
    metadata = _load_jsonl(metadata_path)

    position_names = list(payload["position_names"])
    if args.position not in position_names:
        raise ValueError(f"Position '{args.position}' not in payload positions: {position_names}")
    target_layers = list(payload["target_layers"])
    if args.layer not in target_layers:
        raise ValueError(f"Layer {args.layer} not found in extracted target_layers={target_layers}")

    layer_idx = target_layers.index(args.layer)
    pos_tensor = payload["activations"][args.position]

    mean_n0 = _mean_for_condition(
        activations=pos_tensor,
        metadata=metadata,
        layer_idx=layer_idx,
        condition_code="N0_safety",
    )
    mean_c1 = _mean_for_condition(
        activations=pos_tensor,
        metadata=metadata,
        layer_idx=layer_idx,
        condition_code="C1_safety",
    )
    mean_w1 = _mean_for_condition(
        activations=pos_tensor,
        metadata=metadata,
        layer_idx=layer_idx,
        condition_code="W1_safety",
    )

    authority_given_safe = mean_c1 - mean_n0
    authority_given_harm = mean_w1 - mean_n0
    shared_within_label = 0.5 * (authority_given_safe + authority_given_harm)
    w1_minus_c1 = mean_w1 - mean_c1
    c1_minus_w1 = mean_c1 - mean_w1

    vectors = {
        "authority_given_safe": _normalize(authority_given_safe),
        "authority_given_harm": _normalize(authority_given_harm),
        "shared_within_label": _normalize(shared_within_label),
        "w1_minus_c1": _normalize(w1_minus_c1),
        "c1_minus_w1": _normalize(c1_minus_w1),
    }
    primary = vectors[args.primary_kind]

    diagnostics = {
        "cosine_safe_vs_harm": _cos(authority_given_safe, authority_given_harm),
        "cosine_shared_vs_safe": _cos(shared_within_label, authority_given_safe),
        "cosine_shared_vs_harm": _cos(shared_within_label, authority_given_harm),
        "cosine_w1_minus_c1_vs_shared": _cos(w1_minus_c1, shared_within_label),
        "norms_raw": {
            "authority_given_safe": float(authority_given_safe.norm().item()),
            "authority_given_harm": float(authority_given_harm.norm().item()),
            "shared_within_label": float(shared_within_label.norm().item()),
            "w1_minus_c1": float(w1_minus_c1.norm().item()),
            "c1_minus_w1": float(c1_minus_w1.norm().item()),
        },
        "counts": {
            "n0_rows": sum(1 for r in metadata if r.get("condition_code") == "N0_safety"),
            "c1_rows": sum(1 for r in metadata if r.get("condition_code") == "C1_safety"),
            "w1_rows": sum(1 for r in metadata if r.get("condition_code") == "W1_safety"),
        },
    }

    if args.output_path is None:
        args.output_path = args.extraction_dir / f"safety_primary_direction_{args.position}_L{args.layer}.pt"
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    save_payload = {
        "vector": primary,
        "layer": int(args.layer),
        "position": args.position,
        "primary_kind": args.primary_kind,
        "source": "safety_matched_conditions",
        "vectors": vectors,
        "diagnostics": diagnostics,
        "extraction_dir": str(args.extraction_dir),
    }
    torch.save(save_payload, args.output_path)

    summary_path = args.output_path.with_suffix(".json")
    summary_payload = {
        "output_path": str(args.output_path),
        "layer": int(args.layer),
        "position": args.position,
        "primary_kind": args.primary_kind,
        "diagnostics": diagnostics,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    print(f"saved_direction={args.output_path}")
    print(f"saved_summary={summary_path}")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()

