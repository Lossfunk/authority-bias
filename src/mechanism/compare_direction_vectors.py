from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a set of direction vectors via cosine similarity."
    )
    parser.add_argument("--vector-paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, object]] = []
    loaded: List[tuple[str, np.ndarray, Dict[str, object]]] = []
    for path in args.vector_paths:
        payload = torch.load(path, map_location="cpu")
        vector = payload["vector"].float().numpy()
        name = path.stem
        loaded.append((name, vector, payload))
        rows.append(
            {
                "name": name,
                "path": str(path),
                "norm": float(np.linalg.norm(vector)),
                "metadata": {
                    k: v
                    for k, v in payload.items()
                    if k != "vector"
                },
            }
        )

    pairwise: List[Dict[str, object]] = []
    for i, (name_i, vec_i, _) in enumerate(loaded):
        for j, (name_j, vec_j, _) in enumerate(loaded):
            if j < i:
                continue
            pairwise.append(
                {
                    "vector_a": name_i,
                    "vector_b": name_j,
                    "cosine": _cosine(vec_i, vec_j),
                }
            )

    payload = {"vectors": rows, "pairwise_cosine": pairwise}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
