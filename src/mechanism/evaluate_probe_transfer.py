from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a correction-gating probe on one extraction and evaluate transfer."
    )
    parser.add_argument("--train-extraction-dir", type=Path, required=True)
    parser.add_argument("--eval-extraction-dir", type=Path, required=True)
    parser.add_argument(
        "--position",
        type=str,
        default="endorsement_last",
        choices=("last_token", "instruction_last", "endorsement_last"),
    )
    parser.add_argument("--layer-index", type=int, default=23)
    parser.add_argument("--train-variants", nargs="+", default=["i1a"])
    parser.add_argument("--eval-variants", nargs="+", default=["i1a", "i1c"])
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
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


def _build_probe(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
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
    out = _metrics_from_prob(y, prob)
    out["eval_type"] = "cross_validated"
    return out


def main() -> None:
    args = parse_args()
    train_meta, train_acts = _load_extraction(args.train_extraction_dir, args.position)
    eval_meta, eval_acts = _load_extraction(args.eval_extraction_dir, args.position)

    x_train, y_train = _slice_rows(
        train_meta, train_acts, args.train_variants, args.layer_index
    )
    probe = _build_probe(args.seed)
    probe.fit(x_train, y_train)

    payload: Dict[str, object] = {
        "train_extraction_dir": str(args.train_extraction_dir),
        "eval_extraction_dir": str(args.eval_extraction_dir),
        "position": args.position,
        "layer_index": args.layer_index,
        "train_variants": list(args.train_variants),
        "eval_variants": {},
    }

    if args.train_extraction_dir.resolve() == args.eval_extraction_dir.resolve():
        payload["train_cv"] = _cross_validated_metrics(
            x_train, y_train, cv_folds=args.cv_folds, seed=args.seed
        )

    for variant in args.eval_variants:
        x_eval, y_eval = _slice_rows(eval_meta, eval_acts, [variant], args.layer_index)
        prob = probe.predict_proba(x_eval)[:, 1]
        result = _metrics_from_prob(y_eval, prob)
        result["eval_type"] = "transfer"
        payload["eval_variants"][variant] = result

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
