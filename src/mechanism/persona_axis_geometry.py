"""Analyse the geometric relationship between correction-gating and persona vectors.

Given a correction-gating extraction (activations.pt) and a set of
pre-computed persona vectors (Assistant Axis format), this script computes:

1. Per-layer cosine similarity between the correction-gating direction and each
   persona vector.
2. PCA on the persona vectors at each layer, projecting the correction-gating
   direction into persona space.
3. A JSON output summarising all results.

Usage example:

    python -m src.mechanism.persona_axis_geometry \
        --gating-extraction-dir new-phase-results/mechanism/gemma4_expanded_gating_i1a_i1c_note \
        --gating-position endorsement_last \
        --gating-layer-index 22 \
        --persona-vectors-dir outputs/gemma-2-27b/persona_vectors \
        --output-path analysis/persona_axis_geometry.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _to_numpy_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------

def load_gating_direction(
    extraction_dir: Path,
    position: str,
    layer_index: int,
) -> np.ndarray:
    """Build the correction-gating mean-difference vector from an extraction.

    The extraction stores ``activations.pt`` (dict of tensors keyed by
    ``{position}.{label}``) and ``metadata.json``.  We compute:

        direction = mean(correcting) - mean(resisting)

    at the requested *layer_index* of the stored hidden states.

    Returns a 1-D numpy array of shape ``(hidden_dim,)``.
    """
    acts_path = extraction_dir / "activations.pt"
    if not acts_path.exists():
        raise FileNotFoundError(f"activations.pt not found in {extraction_dir}")

    acts = torch.load(acts_path, map_location="cpu", weights_only=False)

    correcting_key = f"{position}.correcting"
    resisting_key = f"{position}.resisting"

    if correcting_key not in acts or resisting_key not in acts:
        available = sorted(acts.keys())
        raise KeyError(
            f"Expected keys {correcting_key!r} and {resisting_key!r}; "
            f"available: {available}"
        )

    correcting = acts[correcting_key].float()  # (n_correcting, n_layers, hidden)
    resisting = acts[resisting_key].float()    # (n_resisting, n_layers, hidden)

    direction = correcting[:, layer_index, :].mean(dim=0) - resisting[:, layer_index, :].mean(dim=0)
    return _to_numpy_f32(direction)


def load_persona_vectors(
    persona_dir: Path,
) -> List[Dict[str, Any]]:
    """Load pre-computed persona vectors in the Assistant Axis format.

    Each ``.pt`` file is expected to contain a dict with at least:
        - ``vector``: tensor of shape ``(n_layers, hidden_dim)``

    Optionally:
        - ``type``: e.g. ``"pos_3"``
        - ``role``: e.g. ``"scientist"``

    Returns a list of dicts with keys ``name``, ``role``, ``type``,
    ``vector`` (numpy, shape ``(n_layers, hidden_dim)``).
    """
    pt_files = sorted(persona_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {persona_dir}")

    vectors: List[Dict[str, Any]] = []
    for pt_path in pt_files:
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "vector" in payload:
            vec = payload["vector"]
        elif isinstance(payload, torch.Tensor):
            vec = payload
        else:
            raise ValueError(
                f"Unexpected format in {pt_path}: expected dict with 'vector' key "
                f"or a raw tensor, got {type(payload)}"
            )

        vectors.append({
            "name": pt_path.stem,
            "role": payload.get("role", pt_path.stem) if isinstance(payload, dict) else pt_path.stem,
            "type": payload.get("type", "unknown") if isinstance(payload, dict) else "unknown",
            "vector": _to_numpy_f32(vec),  # (n_layers, hidden_dim)
        })

    return vectors


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_cosine_similarities(
    gating_direction: np.ndarray,
    persona_vectors: List[Dict[str, Any]],
    layer_index: int,
) -> List[Dict[str, Any]]:
    """Cosine similarity between gating direction and each persona vector at *layer_index*."""
    results: List[Dict[str, Any]] = []
    for pv in persona_vectors:
        vec = pv["vector"]
        # vec may be (n_layers, hidden_dim) or (hidden_dim,)
        if vec.ndim == 2:
            persona_at_layer = vec[layer_index]
        else:
            persona_at_layer = vec
        cos = _cosine(gating_direction, persona_at_layer)
        results.append({
            "name": pv["name"],
            "role": pv["role"],
            "type": pv["type"],
            "cosine_similarity": cos,
        })
    return results


def compute_per_layer_cosines(
    extraction_dir: Path,
    position: str,
    persona_vectors: List[Dict[str, Any]],
    n_layers: int,
) -> Dict[str, List[float]]:
    """Cosine similarity at every layer for each persona vector."""
    acts = torch.load(extraction_dir / "activations.pt", map_location="cpu", weights_only=False)
    correcting = acts[f"{position}.correcting"].float()
    resisting = acts[f"{position}.resisting"].float()

    per_persona: Dict[str, List[float]] = {}
    for pv in persona_vectors:
        cosines: List[float] = []
        vec = pv["vector"]
        max_layer = min(n_layers, vec.shape[0] if vec.ndim == 2 else n_layers)
        for li in range(max_layer):
            direction = (
                correcting[:, li, :].mean(dim=0) - resisting[:, li, :].mean(dim=0)
            ).numpy()
            persona_at_layer = vec[li] if vec.ndim == 2 else vec
            cosines.append(_cosine(direction, persona_at_layer))
        per_persona[pv["name"]] = cosines
    return per_persona


def compute_pca_projection(
    gating_direction: np.ndarray,
    persona_vectors: List[Dict[str, Any]],
    layer_index: int,
    n_components: int = 5,
) -> Dict[str, Any]:
    """PCA on the persona vectors, then project gating direction into that space."""
    # Collect persona vectors at the target layer
    vecs: List[np.ndarray] = []
    names: List[str] = []
    for pv in persona_vectors:
        vec = pv["vector"]
        if vec.ndim == 2:
            vecs.append(vec[layer_index])
        else:
            vecs.append(vec)
        names.append(pv["name"])

    X = np.stack(vecs, axis=0)  # (n_personas, hidden_dim)

    # Mean-centre
    mean = X.mean(axis=0, keepdims=True)
    X_centred = X - mean

    actual_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=actual_components)
    pca.fit(X_centred)

    # Project persona vectors
    persona_coords = pca.transform(X_centred)  # (n_personas, actual_components)

    # Project gating direction (also centred)
    gating_centred = gating_direction - mean.squeeze()
    gating_coords = pca.transform(gating_centred.reshape(1, -1)).squeeze()  # (actual_components,)

    # Cosine similarity of gating direction with each PC axis
    pc_cosines: List[float] = []
    for i in range(actual_components):
        pc_axis = pca.components_[i]
        pc_cosines.append(_cosine(gating_direction, pc_axis))

    return {
        "n_components": int(actual_components),
        "variance_explained": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "gating_projection_coords": gating_coords.tolist(),
        "gating_cosine_with_pcs": pc_cosines,
        "persona_projections": {
            name: persona_coords[i].tolist()
            for i, name in enumerate(names)
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse geometry between correction-gating direction and persona vectors."
    )
    p.add_argument(
        "--gating-extraction-dir", type=Path, required=True,
        help="Path to extraction dir containing activations.pt.",
    )
    p.add_argument(
        "--gating-position", type=str, default="endorsement_last",
        help="Position name in the extraction (default: endorsement_last).",
    )
    p.add_argument(
        "--gating-layer-index", type=int, required=True,
        help="Layer index to use for the gating direction.",
    )
    p.add_argument(
        "--persona-vectors-dir", type=Path, default=None,
        help="Path to pre-computed persona vectors (directory of .pt files).",
    )
    p.add_argument(
        "--persona-sweep-dir", type=Path, default=None,
        help="Path to persona sweep results (alternative to pre-computed vectors; not yet implemented).",
    )
    p.add_argument(
        "--output-path", type=Path, required=True,
        help="Output JSON path.",
    )
    p.add_argument(
        "--pca-components", type=int, default=5,
        help="Number of PCA components to compute (default: 5).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Resolve persona vectors source
    # ------------------------------------------------------------------
    if args.persona_vectors_dir is not None:
        persona_vectors = load_persona_vectors(args.persona_vectors_dir)
    elif args.persona_sweep_dir is not None:
        raise NotImplementedError(
            "Loading persona vectors from sweep results is not yet implemented. "
            "Please provide --persona-vectors-dir instead."
        )
    else:
        raise ValueError(
            "Provide either --persona-vectors-dir or --persona-sweep-dir."
        )

    print(f"Loaded {len(persona_vectors)} persona vectors: "
          f"{[pv['name'] for pv in persona_vectors]}")

    # ------------------------------------------------------------------
    # 2. Build the correction-gating direction
    # ------------------------------------------------------------------
    gating_direction = load_gating_direction(
        args.gating_extraction_dir,
        args.gating_position,
        args.gating_layer_index,
    )
    print(f"Gating direction shape: {gating_direction.shape}, "
          f"norm: {np.linalg.norm(gating_direction):.4f}")

    # ------------------------------------------------------------------
    # 3. Cosine similarities at the target layer
    # ------------------------------------------------------------------
    cosines_at_layer = compute_cosine_similarities(
        gating_direction, persona_vectors, args.gating_layer_index,
    )
    print("\nCosine similarities (target layer):")
    for entry in sorted(cosines_at_layer, key=lambda e: abs(e["cosine_similarity"]), reverse=True):
        print(f"  {entry['name']:30s}  cos={entry['cosine_similarity']:+.4f}")

    # ------------------------------------------------------------------
    # 4. Per-layer cosines
    # ------------------------------------------------------------------
    # Determine n_layers from first persona vector
    n_layers = persona_vectors[0]["vector"].shape[0] if persona_vectors[0]["vector"].ndim == 2 else 1
    per_layer_cosines = compute_per_layer_cosines(
        args.gating_extraction_dir,
        args.gating_position,
        persona_vectors,
        n_layers,
    )

    # ------------------------------------------------------------------
    # 5. PCA projection
    # ------------------------------------------------------------------
    pca_result = compute_pca_projection(
        gating_direction,
        persona_vectors,
        args.gating_layer_index,
        n_components=args.pca_components,
    )
    print(f"\nPCA variance explained (first {pca_result['n_components']} components): "
          f"{[f'{v:.3f}' for v in pca_result['variance_explained']]}")

    # ------------------------------------------------------------------
    # 6. Write output
    # ------------------------------------------------------------------
    output = {
        "config": {
            "gating_extraction_dir": str(args.gating_extraction_dir),
            "gating_position": args.gating_position,
            "gating_layer_index": args.gating_layer_index,
            "persona_vectors_dir": str(args.persona_vectors_dir) if args.persona_vectors_dir else None,
            "n_persona_vectors": len(persona_vectors),
            "pca_components": args.pca_components,
        },
        "cosine_similarities_at_target_layer": cosines_at_layer,
        "per_layer_cosines": per_layer_cosines,
        "pca": pca_result,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(f"\nResults written to {args.output_path}")


if __name__ == "__main__":
    main()
