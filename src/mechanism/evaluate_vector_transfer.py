from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a correction-gating vector on one extraction and evaluate transfer."
    )
    parser.add_argument("--source-extraction-dir", type=Path, required=True)
    parser.add_argument("--target-extraction-dir", type=Path, required=True)
    parser.add_argument(
        "--position",
        type=str,
        default="endorsement_last",
        choices=("last_token", "instruction_last", "endorsement_last"),
    )
    parser.add_argument("--layer-index", type=int, default=23)
    parser.add_argument("--build-variants", nargs="+", default=["i1a"])
    parser.add_argument("--eval-variants", nargs="+", default=["i1a", "i1c"])
    parser.add_argument("--vector-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def _load_metadata(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_extraction(extraction_dir: Path, position: str) -> Tuple[List[Dict], torch.Tensor]:
    metadata = _load_metadata(extraction_dir / "metadata.jsonl")
    payload = torch.load(extraction_dir / "activations.pt", map_location="cpu")
    return metadata, payload["activations"][position]


def _slice_rows(
    metadata: Sequence[Dict],
    activations: torch.Tensor,
    variants: Sequence[str],
    layer_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rows = [i for i, row in enumerate(metadata) if row["variant_label"] in variants]
    if not rows:
        raise ValueError(f"No rows found for variants={variants}")
    x = activations[torch.tensor(rows), layer_index, :].float().numpy()
    y = np.array(
        [1 if metadata[i]["item_label"] == "correcting" else 0 for i in rows],
        dtype=np.int64,
    )
    return x, y


def _build_vector(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, float]]:
    correcting = x[y == 1]
    entrenching = x[y == 0]
    if len(correcting) == 0 or len(entrenching) == 0:
        raise ValueError("Need both classes to build a vector.")
    vector = correcting.mean(axis=0) - entrenching.mean(axis=0)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Computed zero-norm vector.")
    c_mean = float((correcting @ vector).mean())
    e_mean = float((entrenching @ vector).mean())
    threshold = 0.5 * (c_mean + e_mean)
    stats = {
        "vector_norm": norm,
        "source_correcting_projection_mean": c_mean,
        "source_entrenching_projection_mean": e_mean,
        "source_threshold": threshold,
    }
    return vector, threshold, stats


def _evaluate(x: np.ndarray, y: np.ndarray, vector: np.ndarray, threshold: float) -> Dict[str, float]:
    scores = x @ vector
    pred = (scores >= threshold).astype(np.int64)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "n": int(y.shape[0]),
        "n_correcting": int(np.sum(y == 1)),
        "n_entrenching": int(np.sum(y == 0)),
        "score_mean": float(scores.mean()),
    }
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, scores))
    return out


def main() -> None:
    args = parse_args()
    src_meta, src_acts = _load_extraction(args.source_extraction_dir, args.position)
    tgt_meta, tgt_acts = _load_extraction(args.target_extraction_dir, args.position)

    if args.vector_path is not None:
        payload = torch.load(args.vector_path, map_location="cpu")
        vector = payload["vector"].float().numpy()
        threshold = float(payload.get("source_threshold", 0.0))
        source_stats = {k: v for k, v in payload.items() if k != "vector"}
    else:
        x_src, y_src = _slice_rows(src_meta, src_acts, args.build_variants, args.layer_index)
        vector, threshold, source_stats = _build_vector(x_src, y_src)

    results: Dict[str, object] = {
        "source_extraction_dir": str(args.source_extraction_dir),
        "target_extraction_dir": str(args.target_extraction_dir),
        "position": args.position,
        "layer_index": args.layer_index,
        "build_variants": list(args.build_variants),
        "source_stats": source_stats,
        "eval_variants": {},
    }

    for variant in args.eval_variants:
        x_tgt, y_tgt = _slice_rows(tgt_meta, tgt_acts, [variant], args.layer_index)
        results["eval_variants"][variant] = _evaluate(x_tgt, y_tgt, vector, threshold)

    rendered = json.dumps(results, indent=2, sort_keys=True)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
