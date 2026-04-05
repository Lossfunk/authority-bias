from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


POSITION_NAMES: Tuple[str, ...] = ("last_token", "instruction_last", "endorsement_last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep probe performance across positions and layers for a correction-gating extraction."
    )
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--train-variant", type=str, default="i1a")
    parser.add_argument("--eval-variant", type=str, default="i1c")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of parallel jobs (-1 = all cores)")
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def _load_metadata(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_probe(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    solver="saga",
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def _metrics_from_prob(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    pred = (prob >= 0.5).astype(np.int64)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "n": int(y_true.shape[0]),
        "n_correcting": int(np.sum(y_true == 1)),
        "n_entrenching": int(np.sum(y_true == 0)),
    }
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, prob))
    return out


def _cross_validated_metrics(x: np.ndarray, y: np.ndarray, cv_folds: int, seed: int) -> Dict[str, float]:
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    prob = np.zeros_like(y, dtype=np.float64)
    for train_idx, test_idx in splitter.split(x, y):
        probe = _build_probe(seed)
        probe.fit(x[train_idx], y[train_idx])
        prob[test_idx] = probe.predict_proba(x[test_idx])[:, 1]
    return _metrics_from_prob(y, prob)


def _slice_rows(
    metadata: Sequence[Dict],
    activations: torch.Tensor,
    variant: str,
    layer_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rows = [i for i, row in enumerate(metadata) if row["variant_label"] == variant]
    if not rows:
        raise ValueError(f"No rows found for variant={variant}")
    x = activations[torch.tensor(rows), layer_index, :].float().numpy()
    y = np.array(
        [1 if metadata[i]["item_label"] == "correcting" else 0 for i in rows],
        dtype=np.int64,
    )
    return x, y


def _evaluate_one_combo(
    position: str,
    layer_index: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    train_variant: str,
    eval_variant: str,
    cv_folds: int,
    seed: int,
) -> Dict[str, object]:
    train_cv = _cross_validated_metrics(x_train, y_train, cv_folds, seed)
    probe = _build_probe(seed)
    probe.fit(x_train, y_train)
    eval_prob = probe.predict_proba(x_eval)[:, 1]
    eval_metrics = _metrics_from_prob(y_eval, eval_prob)
    return {
        "position": position,
        "layer_index": layer_index,
        "train_variant": train_variant,
        "eval_variant": eval_variant,
        "train_cv_accuracy": train_cv["accuracy"],
        "train_cv_auroc": train_cv.get("auroc"),
        "eval_accuracy": eval_metrics["accuracy"],
        "eval_auroc": eval_metrics.get("auroc"),
    }


def main() -> None:
    args = parse_args()
    metadata = _load_metadata(args.extraction_dir / "metadata.jsonl")
    payload = torch.load(args.extraction_dir / "activations.pt", map_location="cpu")

    # Pre-slice all data to avoid passing large tensors to workers
    jobs = []
    for position in POSITION_NAMES:
        acts = payload["activations"][position]
        n_layers = int(acts.shape[1])
        for layer_index in range(n_layers):
            x_train, y_train = _slice_rows(metadata, acts, args.train_variant, layer_index)
            x_eval, y_eval = _slice_rows(metadata, acts, args.eval_variant, layer_index)
            jobs.append((position, layer_index, x_train, y_train, x_eval, y_eval))

    n_jobs = args.n_jobs if args.n_jobs != -1 else os.cpu_count()
    print(f"Running {len(jobs)} probe fits across {n_jobs} workers...", flush=True)

    all_rows = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(_evaluate_one_combo)(
            position, layer_index, x_train, y_train, x_eval, y_eval,
            args.train_variant, args.eval_variant, args.cv_folds, args.seed,
        )
        for position, layer_index, x_train, y_train, x_eval, y_eval in jobs
    )

    best_by_position: Dict[str, Dict[str, object]] = {}
    best_overall: Dict[str, object] | None = None
    for row in all_rows:
        position = row["position"]
        score = (
            float(row.get("train_cv_auroc") or row["train_cv_accuracy"]),
            float(row.get("eval_auroc") or row["eval_accuracy"]),
        )
        if position not in best_by_position:
            best_by_position[position] = row
        else:
            prev = best_by_position[position]
            prev_score = (
                float(prev.get("train_cv_auroc") or prev["train_cv_accuracy"]),
                float(prev.get("eval_auroc") or prev["eval_accuracy"]),
            )
            if score > prev_score:
                best_by_position[position] = row
        if best_overall is None:
            best_overall = row
        else:
            prev_score = (
                float(best_overall.get("train_cv_auroc") or best_overall["train_cv_accuracy"]),
                float(best_overall.get("eval_auroc") or best_overall["eval_accuracy"]),
            )
            if score > prev_score:
                best_overall = row

    rendered = {
        "extraction_dir": str(args.extraction_dir),
        "train_variant": args.train_variant,
        "eval_variant": args.eval_variant,
        "best_overall": best_overall,
        "best_by_position": best_by_position,
        "rows": all_rows,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(rendered, indent=2) + "\n")
    print(json.dumps(rendered, indent=2))


if __name__ == "__main__":
    main()
