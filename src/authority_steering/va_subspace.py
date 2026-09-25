from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a64, b64) / denom)


def normalize_np(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec.astype(np.float32, copy=True)
    return (vec / norm).astype(np.float32, copy=False)


def normalize_torch(vector: torch.Tensor) -> torch.Tensor:
    vec = vector.float()
    norm = vec.norm()
    if float(norm) == 0.0:
        return vec
    return vec / norm


def orthonormalize_np(vectors: Iterable[np.ndarray]) -> np.ndarray:
    basis = []
    for vector in vectors:
        residual = np.asarray(vector, dtype=np.float64).copy()
        for direction in basis:
            residual = residual - float(np.dot(residual, direction)) * direction
        norm = float(np.linalg.norm(residual))
        if norm > 0.0:
            basis.append(residual / norm)
    if not basis:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(basis, axis=0).astype(np.float32, copy=False)


def orthonormalize_torch_rows(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.ndim == 1:
        vectors = vectors.unsqueeze(0)
    basis = []
    for row in vectors.float():
        residual = row.clone()
        for direction in basis:
            residual = residual - torch.dot(residual, direction) * direction
        norm = residual.norm()
        if float(norm) > 0.0:
            basis.append(residual / norm)
    if not basis:
        return torch.zeros((0, vectors.shape[-1]), dtype=torch.float32)
    return torch.stack(basis, dim=0)


def project_out_np(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64).copy()
    basis_arr = np.asarray(basis, dtype=np.float64)
    if basis_arr.ndim == 1:
        basis_arr = basis_arr[None, :]
    for direction in basis_arr:
        denom = float(np.dot(direction, direction))
        if denom == 0.0:
            continue
        result = result - (float(np.dot(result, direction)) / denom) * direction
    return result.astype(np.float32, copy=False)


def project_out_torch(vector: torch.Tensor, basis: Optional[torch.Tensor]) -> torch.Tensor:
    if basis is None:
        return vector.float()
    result = vector.float()
    basis_rows = basis.float()
    if basis_rows.ndim == 1:
        basis_rows = basis_rows.unsqueeze(0)
    for direction in basis_rows:
        denom = torch.dot(direction, direction)
        if float(denom) == 0.0:
            continue
        result = result - (torch.dot(result, direction) / denom) * direction
    return result


def projection_coords_np(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64)
    basis_arr = np.asarray(basis, dtype=np.float64)
    if basis_arr.ndim == 1:
        basis_arr = basis_arr[None, :]
    if basis_arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray([float(np.dot(vec, direction)) for direction in basis_arr], dtype=np.float32)


def fraction_of_norm_in_basis(vector: np.ndarray, basis: np.ndarray) -> float:
    vec = np.asarray(vector, dtype=np.float64)
    total_norm = float(np.linalg.norm(vec))
    if total_norm == 0.0:
        return 0.0
    coords = projection_coords_np(vec, basis)
    return float(np.linalg.norm(coords.astype(np.float64)) / total_norm)


def load_subspace_payload(path: Path) -> Dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(payload)}")
    return payload


def load_basis_by_layer(path: Path) -> Dict[int, torch.Tensor]:
    payload = load_subspace_payload(path)
    raw = payload.get("basis_by_layer")
    if not isinstance(raw, dict):
        raise ValueError(f"Subspace payload at {path} is missing 'basis_by_layer'")
    out: Dict[int, torch.Tensor] = {}
    for layer_key, value in raw.items():
        layer = int(layer_key)
        basis = value["basis"] if isinstance(value, dict) and "basis" in value else value
        if not torch.is_tensor(basis):
            basis = torch.tensor(basis, dtype=torch.float32)
        out[layer] = orthonormalize_torch_rows(basis)
    return out
