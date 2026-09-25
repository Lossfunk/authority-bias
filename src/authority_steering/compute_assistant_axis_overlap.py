"""
Compute the overlap between the per-model compliance directions and the
(per-model) assistant axis, matching the schema of the GPT-OSS
hardened overlap_summary.json exactly.

Inputs:
  --axis-path         assistant_axis.pt (shape [n_layers, hidden_dim])
  --role-acts-dir     directory of per-role activation .pt files produced by
                        assistant-axis/pipeline/2_activations.py (each file is
                        a dict {conversation_key -> tensor[n_layers, hidden]})
  --scores-dir        directory of per-role score JSON files (judge scores)
                        — only used to select default behavior. Default role
                        is handled by exact file-name match "default.pt" /
                        "default.json".
  --directions-path   compliance directions.pt
                        (produced by our mechanism pipeline; dict with keys
                         {position, layer, <direction_kinds>, vector})
  --layer             layer at which to compute the overlap (defaults to
                        the layer field inside directions.pt)
  --output-path       path to write overlap_summary.json

Output schema (matching external/assistant-axis/hardened/overlap_summary.json):
{
  "layer": <int>,
  "n_role_vectors": <int>,                     # non-default roles used
  "axis_norm": <float>,                        # L2 norm of axis at that layer
  "pca_variance_explained": [5 floats],        # top-5 variance from role-vector PCA
  "cos_axis_with_roles": {role: cos, ...},
  "cos_axis_with_default": <float>,
  "directions": {
      "<kind>": {
          "cos_with_assistant_axis": <float>,
          "cos_with_each_role":     {role: cos, ...},
          "cos_with_pc1":           <float>,
          "subspace_norm_fraction": <float>,   # fraction of |d| lying in top-5 PC subspace
          "pc_coords":              [5 floats] # projection of d onto top-5 PCs
      },
      ...
  }
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA


def _load_role_vector(act_file: Path, layer: int) -> Tuple[np.ndarray, int]:
    """Load a per-role activations .pt file (dict conv_key -> tensor[L,H] or a raw
    tensor [L,H]) and return mean across all conversations at `layer`.

    Returns (vector[hidden], n_samples_used)
    """
    data = torch.load(act_file, map_location="cpu", weights_only=False)

    if isinstance(data, dict):
        # two possible shapes:
        # (a) conv_key -> tensor[L, H]   (from step 2 raw activations)
        # (b) {"vector": tensor[L, H], ...}  (from step 4 compute_per_role_vectors)
        if "vector" in data and isinstance(data["vector"], torch.Tensor):
            vec = data["vector"][layer].float().numpy()
            return vec, int(data.get("n_samples", 1))
        # else treat as conv_key -> tensor
        tensors = [v for v in data.values() if isinstance(v, torch.Tensor) and v.ndim == 2]
        if not tensors:
            raise ValueError(f"{act_file}: no [L,H] tensors found in dict")
        stacked = torch.stack(tensors, dim=0)  # (N, L, H)
        vec = stacked[:, layer, :].float().mean(dim=0).numpy()
        return vec, stacked.shape[0]
    if isinstance(data, torch.Tensor):
        if data.ndim == 2:  # [L, H]
            return data[layer].float().numpy(), 1
        raise ValueError(f"{act_file}: unexpected tensor shape {tuple(data.shape)}")
    raise ValueError(f"{act_file}: unsupported payload type {type(data).__name__}")


def _load_role_vector_filtered(
    act_file: Path,
    score_file: Path | None,
    layer: int,
) -> Tuple[np.ndarray, int]:
    """For non-default roles, restrict to score=3 samples when score_file is present.

    If score_file is None, returns mean over all samples (used for `default`).
    """
    if score_file is None or not score_file.exists():
        return _load_role_vector(act_file, layer)

    data = torch.load(act_file, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"{act_file}: expected dict (conv_key -> tensor) to filter by score")
    scores: Dict[str, int] = json.loads(score_file.read_text())
    kept: List[torch.Tensor] = []
    for key, tensor in data.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if scores.get(key) == 3:
            kept.append(tensor)
    if not kept:
        raise ValueError(f"{act_file}: no samples with score=3")
    stacked = torch.stack(kept, dim=0)  # (N, L, H)
    return stacked[:, layer, :].float().mean(dim=0).numpy(), stacked.shape[0]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _compute_axis(
    role_vectors: Dict[str, np.ndarray],
    default_vec: np.ndarray,
) -> np.ndarray:
    mean_role = np.mean(np.stack(list(role_vectors.values()), axis=0), axis=0)
    return default_vec - mean_role


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--role-acts-dir", type=Path, required=True,
                   help="Dir of per-role activations .pt files (step 2 output or step 4 vectors).")
    p.add_argument("--scores-dir", type=Path, default=None,
                   help="Dir of per-role judge score JSON files (optional).")
    p.add_argument("--directions-path", type=Path, required=True,
                   help="Compliance directions.pt (our mechanism pipeline).")
    p.add_argument("--layer", type=int, default=None,
                   help="Layer to compute overlap at; defaults to directions.pt 'layer'.")
    p.add_argument("--output-path", type=Path, required=True)
    p.add_argument("--axis-output-path", type=Path, default=None,
                   help="If set, save computed assistant_axis.pt here.")
    p.add_argument("--default-role", type=str, default="default")
    args = p.parse_args()

    directions = torch.load(args.directions_path, map_location="cpu", weights_only=False)
    layer = int(args.layer) if args.layer is not None else int(directions["layer"])

    # Load per-role vectors at the chosen layer.
    role_vectors: Dict[str, np.ndarray] = {}
    default_vec: np.ndarray | None = None
    for act_file in sorted(args.role_acts_dir.glob("*.pt")):
        role = act_file.stem
        score_file = None
        if args.scores_dir is not None:
            score_file = args.scores_dir / f"{role}.json"
        if role == args.default_role:
            default_vec, _ = _load_role_vector(act_file, layer)  # default: no score filter
        else:
            try:
                vec, _ = _load_role_vector_filtered(act_file, score_file, layer)
            except ValueError as exc:
                print(f"[skip] {role}: {exc}")
                continue
            role_vectors[role] = vec

    if default_vec is None:
        raise SystemExit(f"default role file '{args.default_role}.pt' not found in {args.role_acts_dir}")
    if not role_vectors:
        raise SystemExit("no non-default role vectors usable (all filtered out)")

    # Assistant axis and its norm at this layer.
    axis = _compute_axis(role_vectors, default_vec)
    axis_norm = float(np.linalg.norm(axis))

    # PCA over non-default role vectors (top-5 PCs).
    role_matrix = np.stack([role_vectors[r] for r in sorted(role_vectors)], axis=0)  # (R, H)
    centered = role_matrix - role_matrix.mean(axis=0, keepdims=True)
    k = min(5, centered.shape[0] - 1, centered.shape[1])
    pca = PCA(n_components=k)
    pca.fit(centered)
    pcs = pca.components_  # (k, H)
    pca_var = pca.explained_variance_ratio_.tolist()

    # Per-role cos with axis
    cos_axis_with_roles = {r: _cosine(axis, role_vectors[r]) for r in sorted(role_vectors)}
    cos_axis_with_default = _cosine(axis, default_vec)

    # For each compliance direction kind, compute overlaps
    direction_kinds = [
        "shared_within_label",
        "authority_given_wrong",
        "authority_given_correct",
        "pooled_authority",
        "endorsement_presence",
        "content",
    ]
    directions_payload: Dict[str, Dict] = {}
    for kind in direction_kinds:
        if kind not in directions or not isinstance(directions[kind], torch.Tensor):
            continue
        d = directions[kind].float().numpy()
        if d.shape[0] != axis.shape[0]:
            raise SystemExit(f"Direction '{kind}' shape {d.shape} incompatible with axis {axis.shape}")
        cos_with_axis = _cosine(d, axis)
        cos_with_each_role = {r: _cosine(d, role_vectors[r]) for r in sorted(role_vectors)}
        coords = pcs @ d  # (k,)
        subspace_norm_fraction = float(np.linalg.norm(coords) / (np.linalg.norm(d) + 1e-12))
        cos_with_pc1 = _cosine(pcs[0], d) if pcs.shape[0] >= 1 else 0.0
        directions_payload[kind] = {
            "cos_with_assistant_axis": cos_with_axis,
            "cos_with_each_role": cos_with_each_role,
            "cos_with_pc1": cos_with_pc1,
            "subspace_norm_fraction": subspace_norm_fraction,
            "pc_coords": [float(x) for x in coords.tolist()],
        }

    out = {
        "layer": layer,
        "n_role_vectors": len(role_vectors),
        "axis_norm": axis_norm,
        "pca_variance_explained": pca_var,
        "cos_axis_with_roles": cos_axis_with_roles,
        "cos_axis_with_default": cos_axis_with_default,
        "directions": directions_payload,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.output_path}")

    if args.axis_output_path is not None:
        args.axis_output_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as tensor of shape (hidden,) at this layer and a full-layer placeholder
        torch.save({
            "axis_layer": torch.tensor(axis, dtype=torch.float32),
            "layer": layer,
            "n_role_vectors": len(role_vectors),
        }, args.axis_output_path)
        print(f"wrote {args.axis_output_path}")


if __name__ == "__main__":
    main()
