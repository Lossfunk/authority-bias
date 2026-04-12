from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from contextlib import nullcontext
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits


STYLE_NON_THINK = ("weak", "uncertain", "assertive", "authoritative_verified")
STYLE_WITH_THINK = (*STYLE_NON_THINK, "think")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze authority-conditioned compliance directions from exp16 activations.")
    p.add_argument("--extraction-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--inner-cv-folds", type=int, default=4)
    p.add_argument("--random-direction-runs", type=int, default=100)
    p.add_argument("--shuffled-label-runs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for joblib (-1 = all cores).")
    p.add_argument("--parallel-prefer", choices=("threads", "processes"), default="threads")
    p.add_argument("--blas-threads", type=int, default=1, help="BLAS/OpenMP threads per worker (threads backend only).")
    return p.parse_args()


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs == -1:
        return max(1, os.cpu_count() or 1)
    if n_jobs <= 0:
        return 1
    return n_jobs


def _threadpool_context(parallel_prefer: str, blas_threads: int):
    if parallel_prefer != "threads":
        return nullcontext()
    if blas_threads <= 0:
        return nullcontext()
    return threadpool_limits(limits=blas_threads)


def _load_metadata(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _as_uid_set(values: Iterable[str]) -> set[str]:
    return {str(v) for v in values}


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _build_probe(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class ActivationStore:
    def __init__(self, metadata: List[Dict], payload: Dict):
        self.metadata = metadata
        self.activations: Dict[str, torch.Tensor] = payload["activations"]
        self.position_names: List[str] = list(payload["position_names"])
        self.target_layers: List[int] = list(payload["target_layers"])
        self.layer_to_index: Dict[int, int] = {layer: i for i, layer in enumerate(self.target_layers)}

        self.idx_by_uid_style_cond: Dict[Tuple[str, str, str], int] = {}
        for idx, row in enumerate(metadata):
            key = (row["uid"], row["style"], row["condition_code"])
            self.idx_by_uid_style_cond[key] = idx

    def _indices(self, uids: Sequence[str], style: str, condition_code: str) -> List[int]:
        out: List[int] = []
        for uid in uids:
            idx = self.idx_by_uid_style_cond.get((uid, style, condition_code))
            if idx is not None:
                out.append(idx)
        return out

    def matrix(
        self,
        *,
        position: str,
        layer_idx: int,
        uids: Sequence[str],
        style: str,
        condition_code: str,
    ) -> np.ndarray:
        indices = self._indices(uids, style, condition_code)
        if not indices:
            return np.zeros((0, self.activations[position].shape[-1]), dtype=np.float32)
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        mat = self.activations[position][idx_tensor, layer_idx, :].float().numpy()
        return mat

    def vector(
        self,
        *,
        position: str,
        layer_idx: int,
        uid: str,
        style: str,
        condition_code: str,
    ) -> Optional[np.ndarray]:
        idx = self.idx_by_uid_style_cond.get((uid, style, condition_code))
        if idx is None:
            return None
        vec = self.activations[position][idx, layer_idx, :].float().numpy()
        return vec


def _direction_from_uids(
    store: ActivationStore,
    *,
    uids: Sequence[str],
    position: str,
    layer_idx: int,
) -> Dict[str, np.ndarray]:
    # Authoritative-only for primary direction extraction
    m_c1 = store.matrix(
        position=position,
        layer_idx=layer_idx,
        uids=uids,
        style="authoritative_verified",
        condition_code="C1_note",
    )
    m_w1 = store.matrix(
        position=position,
        layer_idx=layer_idx,
        uids=uids,
        style="authoritative_verified",
        condition_code="W1_note",
    )
    m_n0 = store.matrix(
        position=position,
        layer_idx=layer_idx,
        uids=uids,
        style="authoritative_verified",
        condition_code="N0_note",
    )
    if min(len(m_c1), len(m_w1), len(m_n0)) == 0:
        dim = store.activations[position].shape[-1]
        zeros = np.zeros(dim, dtype=np.float32)
        return {
            "pooled_authority": zeros.copy(),
            "content": zeros.copy(),
            "endorsement_presence": zeros.copy(),
            "authority_given_correct": zeros.copy(),
            "authority_given_wrong": zeros.copy(),
            "shared_within_label": zeros.copy(),
        }

    mean_c1 = m_c1.mean(axis=0)
    mean_w1 = m_w1.mean(axis=0)
    mean_n0 = m_n0.mean(axis=0)

    # Content / presence across 4 non-think levels
    all_c1_parts = []
    all_w1_parts = []
    for style in STYLE_NON_THINK:
        c1 = store.matrix(
            position=position,
            layer_idx=layer_idx,
            uids=uids,
            style=style,
            condition_code="C1_note",
        )
        w1 = store.matrix(
            position=position,
            layer_idx=layer_idx,
            uids=uids,
            style=style,
            condition_code="W1_note",
        )
        if len(c1):
            all_c1_parts.append(c1)
        if len(w1):
            all_w1_parts.append(w1)

    if all_c1_parts:
        all_c1 = np.vstack(all_c1_parts)
        mean_all_c1 = all_c1.mean(axis=0)
    else:
        mean_all_c1 = np.zeros_like(mean_c1)
    if all_w1_parts:
        all_w1 = np.vstack(all_w1_parts)
        mean_all_w1 = all_w1.mean(axis=0)
    else:
        mean_all_w1 = np.zeros_like(mean_w1)

    authority_given_correct = mean_c1 - mean_n0
    authority_given_wrong = mean_w1 - mean_n0
    pooled_authority = 0.5 * (authority_given_correct + authority_given_wrong)
    content = mean_all_c1 - mean_all_w1
    endorsement_presence = 0.5 * (mean_all_c1 + mean_all_w1) - mean_n0
    shared_within_label = 0.5 * (authority_given_correct + authority_given_wrong)

    def _normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n == 0.0:
            return v.astype(np.float32)
        return (v / n).astype(np.float32)

    return {
        "pooled_authority": _normalize(pooled_authority),
        "content": _normalize(content),
        "endorsement_presence": _normalize(endorsement_presence),
        "authority_given_correct": _normalize(authority_given_correct),
        "authority_given_wrong": _normalize(authority_given_wrong),
        "shared_within_label": _normalize(shared_within_label),
    }


def _task_data_from_masks(
    masks: Dict,
    *,
    task: str,
) -> Tuple[List[str], np.ndarray]:
    outcomes = masks["outcomes_by_uid_style"]
    if task == "w1":
        mask_key = "mask_primary_w1"
        label_key = "w1_flip"
    elif task == "c1":
        mask_key = "mask_primary_c1"
        label_key = "c1_correction"
    else:
        raise ValueError(f"Unknown task: {task}")

    uids = []
    labels = []
    for uid in masks.get(mask_key, []):
        label = outcomes.get(uid, {}).get("authoritative_verified", {}).get(label_key)
        if label is None:
            continue
        uids.append(uid)
        labels.append(1 if bool(label) else 0)
    return uids, np.array(labels, dtype=np.int64)


def _outer_inner_splits(y: np.ndarray, outer_folds: int, inner_folds: int, seed: int) -> Tuple[StratifiedKFold, int]:
    class_counts = np.bincount(y)
    minority = int(class_counts.min()) if len(class_counts) >= 2 else 0
    if minority < 2:
        raise ValueError("Not enough minority-class samples for stratified CV.")
    n_outer = max(2, min(outer_folds, minority))
    n_inner = max(2, min(inner_folds, minority - 1 if minority > 2 else minority))
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    return outer, n_inner


def _evaluate_pair_fold(
    store: ActivationStore,
    *,
    task_condition: str,
    train_uids: Sequence[str],
    val_uids: Sequence[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    position: str,
    layer_idx: int,
    seed: int,
) -> Optional[float]:
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return None
    dirs = _direction_from_uids(store, uids=train_uids, position=position, layer_idx=layer_idx)
    direction = dirs["shared_within_label"]
    if float(np.linalg.norm(direction)) == 0.0:
        return None
    x_train = store.matrix(
        position=position,
        layer_idx=layer_idx,
        uids=train_uids,
        style="authoritative_verified",
        condition_code=task_condition,
    )
    x_val = store.matrix(
        position=position,
        layer_idx=layer_idx,
        uids=val_uids,
        style="authoritative_verified",
        condition_code=task_condition,
    )
    if len(x_train) != len(y_train) or len(x_val) != len(y_val):
        return None
    s_train = (x_train @ direction).reshape(-1, 1)
    s_val = (x_val @ direction).reshape(-1, 1)
    probe = _build_probe(seed)
    probe.fit(s_train, y_train)
    prob = probe.predict_proba(s_val)[:, 1]
    return _safe_auroc(y_val, prob)


def _nested_task_eval(
    store: ActivationStore,
    *,
    task: str,
    uids: List[str],
    y: np.ndarray,
    layer_values: Sequence[int],
    position_names: Sequence[str],
    cv_folds: int,
    inner_cv_folds: int,
    seed: int,
    n_jobs: int = -1,
    parallel_prefer: str = "threads",
    blas_threads: int = 1,
) -> Dict[str, object]:
    task_condition = "W1_note" if task == "w1" else "C1_note"
    if len(np.unique(y)) < 2:
        return {
            "task": task,
            "metrics": {"auroc": None, "accuracy": None, "n_eval": 0, "n_total": int(len(y))},
            "selected_pairs_by_fold": [],
            "pair_frequency": {},
            "primary_pair": {"position": position_names[0], "layer": int(layer_values[0])},
        }
    class_counts = np.bincount(y)
    minority = int(class_counts.min()) if len(class_counts) >= 2 else 0
    if minority < 2:
        return {
            "task": task,
            "metrics": {"auroc": None, "accuracy": None, "n_eval": 0, "n_total": int(len(y))},
            "selected_pairs_by_fold": [],
            "pair_frequency": {},
            "primary_pair": {"position": position_names[0], "layer": int(layer_values[0])},
        }
    outer_splitter, n_inner = _outer_inner_splits(y, cv_folds, inner_cv_folds, seed)
    uid_arr = np.array(uids)
    oof_prob = np.zeros(len(uids), dtype=np.float64)
    oof_mask = np.zeros(len(uids), dtype=bool)
    selected_pairs: List[Tuple[str, int]] = []
    pair_counter: Counter[Tuple[str, int]] = Counter()

    all_pairs = [(pos, layer) for pos in position_names for layer in layer_values]
    eff_jobs = _resolve_n_jobs(n_jobs)
    for fold_idx, (train_idx, test_idx) in enumerate(outer_splitter.split(np.zeros(len(y)), y)):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        train_uids = uid_arr[train_idx].tolist()
        test_uids = uid_arr[test_idx].tolist()

        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + fold_idx)
        pair_scores: Dict[Tuple[str, int], List[float]] = {pair: [] for pair in all_pairs}

        for inner_idx, (in_train_idx, in_val_idx) in enumerate(inner.split(np.zeros(len(y_train)), y_train)):
            in_train_uids = [train_uids[i] for i in in_train_idx]
            in_val_uids = [train_uids[i] for i in in_val_idx]
            in_y_train = y_train[in_train_idx]
            in_y_val = y_train[in_val_idx]
            with _threadpool_context(parallel_prefer, blas_threads):
                results = Parallel(n_jobs=eff_jobs, prefer=parallel_prefer)(
                    delayed(_evaluate_pair_fold)(
                        store,
                        task_condition=task_condition,
                        train_uids=in_train_uids,
                        val_uids=in_val_uids,
                        y_train=in_y_train,
                        y_val=in_y_val,
                        position=position,
                        layer_idx=store.layer_to_index[layer_val],
                        seed=seed + fold_idx * 1000 + inner_idx * 100 + pair_idx,
                    )
                    for pair_idx, (position, layer_val) in enumerate(all_pairs)
                )
            for pair, auc in zip(all_pairs, results):
                if auc is not None:
                    pair_scores[pair].append(auc)

        best_pair = None
        best_score = -1.0
        for pair, scores in pair_scores.items():
            if not scores:
                continue
            score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                best_pair = pair

        if best_pair is None:
            continue

        selected_pairs.append(best_pair)
        pair_counter[best_pair] += 1
        best_pos, best_layer_val = best_pair
        best_layer_idx = store.layer_to_index[best_layer_val]

        dirs = _direction_from_uids(store, uids=train_uids, position=best_pos, layer_idx=best_layer_idx)
        direction = dirs["shared_within_label"]

        x_train = store.matrix(
            position=best_pos,
            layer_idx=best_layer_idx,
            uids=train_uids,
            style="authoritative_verified",
            condition_code=task_condition,
        )
        x_test = store.matrix(
            position=best_pos,
            layer_idx=best_layer_idx,
            uids=test_uids,
            style="authoritative_verified",
            condition_code=task_condition,
        )
        s_train = (x_train @ direction).reshape(-1, 1)
        s_test = (x_test @ direction).reshape(-1, 1)

        if len(np.unique(y_train)) < 2:
            continue
        probe = _build_probe(seed + fold_idx)
        probe.fit(s_train, y_train)
        prob = probe.predict_proba(s_test)[:, 1]
        oof_prob[test_idx] = prob
        oof_mask[test_idx] = True

    valid_idx = np.where(oof_mask)[0]
    y_eval = y[valid_idx]
    prob_eval = oof_prob[valid_idx]
    pred_eval = (prob_eval >= 0.5).astype(np.int64)
    metrics = {
        "auroc": _safe_auroc(y_eval, prob_eval),
        "accuracy": float(accuracy_score(y_eval, pred_eval)) if len(y_eval) else None,
        "n_eval": int(len(y_eval)),
        "n_total": int(len(y)),
    }

    if pair_counter:
        primary_pair = pair_counter.most_common(1)[0][0]
    else:
        primary_pair = (position_names[0], layer_values[0])

    return {
        "task": task,
        "metrics": metrics,
        "selected_pairs_by_fold": [{"position": p, "layer": l} for p, l in selected_pairs],
        "pair_frequency": {f"{p}::L{l}": int(c) for (p, l), c in pair_counter.items()},
        "primary_pair": {"position": primary_pair[0], "layer": int(primary_pair[1])},
    }


def _simple_cv_pair_auroc(
    store: ActivationStore,
    *,
    task: str,
    uids: List[str],
    y: np.ndarray,
    position: str,
    layer_val: int,
    cv_folds: int,
    seed: int,
) -> Optional[float]:
    return _simple_cv_pair_auroc_with_direction(
        store,
        task=task,
        uids=uids,
        y=y,
        position=position,
        layer_val=layer_val,
        cv_folds=cv_folds,
        seed=seed,
        direction_name="shared_within_label",
    )


def _simple_cv_pair_auroc_with_direction(
    store: ActivationStore,
    *,
    task: str,
    uids: List[str],
    y: np.ndarray,
    position: str,
    layer_val: int,
    cv_folds: int,
    seed: int,
    direction_name: str,
) -> Optional[float]:
    if len(np.unique(y)) < 2:
        return None
    task_condition = "W1_note" if task == "w1" else "C1_note"
    layer_idx = store.layer_to_index[layer_val]
    class_counts = np.bincount(y)
    minority = int(class_counts.min()) if len(class_counts) >= 2 else 0
    if minority < 2:
        return None
    n_splits = max(2, min(cv_folds, minority))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    uid_arr = np.array(uids)
    oof_prob = np.zeros(len(uids), dtype=np.float64)
    oof_mask = np.zeros(len(uids), dtype=bool)
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        if len(np.unique(y[train_idx])) < 2:
            continue
        train_uids = uid_arr[train_idx].tolist()
        test_uids = uid_arr[test_idx].tolist()
        dirs = _direction_from_uids(store, uids=train_uids, position=position, layer_idx=layer_idx)
        direction = dirs[direction_name]
        if float(np.linalg.norm(direction)) == 0.0:
            continue
        x_train = store.matrix(
            position=position,
            layer_idx=layer_idx,
            uids=train_uids,
            style="authoritative_verified",
            condition_code=task_condition,
        )
        x_test = store.matrix(
            position=position,
            layer_idx=layer_idx,
            uids=test_uids,
            style="authoritative_verified",
            condition_code=task_condition,
        )
        s_train = (x_train @ direction).reshape(-1, 1)
        s_test = (x_test @ direction).reshape(-1, 1)
        probe = _build_probe(seed + fold_idx)
        probe.fit(s_train, y[train_idx])
        prob = probe.predict_proba(s_test)[:, 1]
        oof_prob[test_idx] = prob
        oof_mask[test_idx] = True
    valid_idx = np.where(oof_mask)[0]
    if len(valid_idx) == 0:
        return None
    return _safe_auroc(y[valid_idx], oof_prob[valid_idx])


def _heatmap_worker(
    pi: int,
    li: int,
    store: ActivationStore,
    task: str,
    uids: List[str],
    y: np.ndarray,
    position: str,
    layer_val: int,
    cv_folds: int,
    seed: int,
) -> Tuple[int, int, Optional[float]]:
    auc = _simple_cv_pair_auroc(
        store, task=task, uids=uids, y=y, position=position,
        layer_val=layer_val, cv_folds=cv_folds, seed=seed,
    )
    return pi, li, auc


def _make_heatmap(
    store: ActivationStore,
    *,
    task: str,
    uids: List[str],
    y: np.ndarray,
    output_path: Path,
    cv_folds: int,
    seed: int,
    n_jobs: int = -1,
    parallel_prefer: str = "threads",
    blas_threads: int = 1,
) -> Tuple[np.ndarray, List[str], List[int]]:
    positions = list(store.position_names)
    layers = list(store.target_layers)
    mat = np.full((len(positions), len(layers)), np.nan, dtype=np.float64)

    jobs = [
        (pi, li, pos, layer_val)
        for pi, pos in enumerate(positions)
        for li, layer_val in enumerate(layers)
    ]
    eff_jobs = _resolve_n_jobs(n_jobs)
    with _threadpool_context(parallel_prefer, blas_threads):
        results = Parallel(n_jobs=eff_jobs, prefer=parallel_prefer)(
            delayed(_heatmap_worker)(
                pi, li, store, task, uids, y, pos, layer_val, cv_folds, seed,
            )
            for pi, li, pos, layer_val in jobs
        )
    for pi, li, auc in results:
        if auc is not None:
            mat[pi, li] = auc

    plt.figure(figsize=(max(9, len(layers) * 0.6), 4.5))
    im = plt.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="AUROC")
    plt.yticks(range(len(positions)), positions)
    plt.xticks(range(len(layers)), layers, rotation=45)
    plt.title(f"{task.upper()} AUROC across layer/position")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return mat, positions, layers


def _bootstrap_ci(y: np.ndarray, prob: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> Tuple[Optional[float], Optional[float]]:
    if len(np.unique(y)) < 2:
        return None, None
    vals: List[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        vals.append(float(roc_auc_score(y_b, prob[idx])))
    if not vals:
        return None, None
    low, high = np.percentile(vals, [2.5, 97.5])
    return float(low), float(high)


def _oriented_auroc(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    auc = _safe_auroc(y, score)
    if auc is None:
        return None
    return float(max(auc, 1.0 - auc))


def _empirical_right_tail_pvalue(observed: Optional[float], samples: Sequence[float]) -> Optional[float]:
    if observed is None or not samples:
        return None
    arr = np.asarray(samples, dtype=np.float64)
    return float((1.0 + np.sum(arr >= observed)) / (len(arr) + 1.0))


def _random_raw_projection_baseline(
    x: np.ndarray,
    y: np.ndarray,
    *,
    runs: int,
    seed: int,
) -> Dict[str, object]:
    if len(np.unique(y)) < 2 or len(x) == 0:
        return {"auroc_mean": None, "auroc_std": None, "auroc_values": []}
    dim = x.shape[1]
    rng = np.random.default_rng(seed)
    aucs: List[float] = []
    for _ in range(runs):
        d = rng.standard_normal(dim).astype(np.float32)
        d /= max(np.linalg.norm(d), 1e-8)
        auc = _oriented_auroc(y, x @ d)
        if auc is not None:
            aucs.append(auc)
    return {
        "auroc_mean": float(np.mean(aucs)) if aucs else None,
        "auroc_std": float(np.std(aucs)) if aucs else None,
        "auroc_values": aucs,
        "auroc_p97_5": float(np.percentile(aucs, 97.5)) if aucs else None,
    }


def _majority_baseline(y: np.ndarray) -> Dict[str, Optional[float]]:
    if len(y) == 0:
        return {"accuracy": None, "auroc": None}
    pred = int(np.mean(y) >= 0.5)
    acc = float(np.mean(y == pred))
    return {"accuracy": acc, "auroc": 0.5 if len(np.unique(y)) == 2 else None}


def _random_direction_single_run(
    run_idx: int,
    x: np.ndarray,
    y: np.ndarray,
    dim: int,
    n_splits: int,
    seed: int,
) -> Optional[float]:
    rng = np.random.default_rng(seed + run_idx * 7919)
    d = rng.standard_normal(dim).astype(np.float32)
    d /= max(np.linalg.norm(d), 1e-8)
    scores = x @ d
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    mask = np.zeros(len(y), dtype=bool)
    for fold_idx, (tr, te) in enumerate(splitter.split(np.zeros(len(y)), y)):
        if len(np.unique(y[tr])) < 2:
            continue
        probe = _build_probe(seed + run_idx * 100 + fold_idx)
        probe.fit(scores[tr].reshape(-1, 1), y[tr])
        oof[te] = probe.predict_proba(scores[te].reshape(-1, 1))[:, 1]
        mask[te] = True
    idx = np.where(mask)[0]
    return _safe_auroc(y[idx], oof[idx])


def _random_direction_baseline(
    store: ActivationStore,
    *,
    task: str,
    uids: List[str],
    y: np.ndarray,
    position: str,
    layer_val: int,
    n_runs: int,
    cv_folds: int,
    seed: int,
    n_jobs: int = -1,
    parallel_prefer: str = "threads",
    blas_threads: int = 1,
) -> Dict[str, object]:
    if len(np.unique(y)) < 2:
        return {"auroc_mean": None, "auroc_std": None, "auroc_values": []}
    task_condition = "W1_note" if task == "w1" else "C1_note"
    layer_idx = store.layer_to_index[layer_val]
    x = store.matrix(
        position=position,
        layer_idx=layer_idx,
        uids=uids,
        style="authoritative_verified",
        condition_code=task_condition,
    )
    dim = x.shape[1]
    class_counts = np.bincount(y)
    minority = int(class_counts.min()) if len(class_counts) >= 2 else 0
    if minority < 2:
        return {"auroc_mean": None, "auroc_std": None, "auroc_values": []}
    n_splits = max(2, min(cv_folds, minority))

    eff_jobs = _resolve_n_jobs(n_jobs)
    with _threadpool_context(parallel_prefer, blas_threads):
        results = Parallel(n_jobs=eff_jobs, prefer=parallel_prefer)(
            delayed(_random_direction_single_run)(run_idx, x, y, dim, n_splits, seed)
            for run_idx in range(n_runs)
        )
    aucs = [a for a in results if a is not None]

    return {
        "auroc_mean": float(np.mean(aucs)) if aucs else None,
        "auroc_std": float(np.std(aucs)) if aucs else None,
        "auroc_values": aucs,
    }


def _shuffled_label_single_run(
    run_idx: int,
    y: np.ndarray,
    feature: np.ndarray,
    n_splits: int,
    seed: int,
) -> Optional[float]:
    rng = np.random.default_rng(seed + run_idx * 6271)
    y_shuf = np.array(y, copy=True)
    rng.shuffle(y_shuf)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + run_idx)
    oof = np.zeros(len(y), dtype=np.float64)
    mask = np.zeros(len(y), dtype=bool)
    for fold_idx, (tr, te) in enumerate(splitter.split(np.zeros(len(y_shuf)), y_shuf)):
        if len(np.unique(y_shuf[tr])) < 2:
            continue
        probe = _build_probe(seed + run_idx * 100 + fold_idx)
        probe.fit(feature[tr].reshape(-1, 1), y_shuf[tr])
        oof[te] = probe.predict_proba(feature[te].reshape(-1, 1))[:, 1]
        mask[te] = True
    idx = np.where(mask)[0]
    return _safe_auroc(y_shuf[idx], oof[idx])


def _shuffled_label_baseline(
    y: np.ndarray,
    feature: np.ndarray,
    *,
    runs: int,
    cv_folds: int,
    seed: int,
    n_jobs: int = -1,
    parallel_prefer: str = "threads",
    blas_threads: int = 1,
) -> Dict[str, object]:
    if len(np.unique(y)) < 2:
        return {"auroc_mean": None, "auroc_std": None, "auroc_values": []}
    class_counts = np.bincount(y)
    minority = int(class_counts.min()) if len(class_counts) >= 2 else 0
    if minority < 2:
        return {"auroc_mean": None, "auroc_std": None, "auroc_values": []}
    n_splits = max(2, min(cv_folds, minority))

    eff_jobs = _resolve_n_jobs(n_jobs)
    with _threadpool_context(parallel_prefer, blas_threads):
        results = Parallel(n_jobs=eff_jobs, prefer=parallel_prefer)(
            delayed(_shuffled_label_single_run)(run_idx, y, feature, n_splits, seed)
            for run_idx in range(runs)
        )
    aucs = [a for a in results if a is not None]
    return {
        "auroc_mean": float(np.mean(aucs)) if aucs else None,
        "auroc_std": float(np.std(aucs)) if aucs else None,
        "auroc_values": aucs,
        "auroc_p97_5": float(np.percentile(aucs, 97.5)) if aucs else None,
    }


def _authority_level_only_baseline(
    masks: Dict,
    *,
    task: str,
    cv_folds: int,
    seed: int,
) -> Dict[str, Optional[float]]:
    outcomes = masks["outcomes_by_uid_style"]
    rows = []
    labels = []
    for uid in masks.get("mask_cross", []):
        for style in STYLE_NON_THINK:
            info = outcomes.get(uid, {}).get(style, {})
            if task == "w1":
                if not bool(info.get("n0_correct", False)):
                    continue
                label = info.get("w1_flip")
            else:
                label = info.get("c1_correction")
            if label is None:
                continue
            rows.append([style])
            labels.append(1 if bool(label) else 0)
    if not rows or len(set(labels)) < 2:
        return {"auroc": None, "accuracy": None, "n": len(labels)}

    x = np.array(rows, dtype=object)
    y = np.array(labels, dtype=np.int64)
    class_counts = np.bincount(y)
    minority = int(class_counts.min()) if len(class_counts) >= 2 else 0
    if minority < 2:
        return {"auroc": None, "accuracy": None, "n": len(labels)}
    n_splits = max(2, min(cv_folds, minority))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_enc = enc.fit_transform(x)

    oof_prob = np.zeros(len(y), dtype=np.float64)
    mask = np.zeros(len(y), dtype=bool)
    for fold_idx, (tr, te) in enumerate(splitter.split(np.zeros(len(y)), y)):
        if len(np.unique(y[tr])) < 2:
            continue
        probe = _build_probe(seed + fold_idx)
        probe.fit(x_enc[tr], y[tr])
        oof_prob[te] = probe.predict_proba(x_enc[te])[:, 1]
        mask[te] = True
    idx = np.where(mask)[0]
    if len(idx) == 0 or len(np.unique(y[idx])) < 2:
        return {"auroc": None, "accuracy": None, "n": len(labels)}
    pred = (oof_prob[idx] >= 0.5).astype(np.int64)
    return {
        "auroc": _safe_auroc(y[idx], oof_prob[idx]),
        "accuracy": float(accuracy_score(y[idx], pred)),
        "n": int(len(y)),
    }


def _trajectory_and_projection(
    store: ActivationStore,
    *,
    masks: Dict,
    position: str,
    layer_val: int,
    direction: np.ndarray,
) -> Dict[str, object]:
    layer_idx = store.layer_to_index[layer_val]
    outcomes = masks["outcomes_by_uid_style"]

    def _level_projection(style: str, condition_code: str, uids: List[str]) -> Optional[float]:
        mat = store.matrix(
            position=position,
            layer_idx=layer_idx,
            uids=uids,
            style=style,
            condition_code=condition_code,
        )
        if len(mat) == 0:
            return None
        centroid = mat.mean(axis=0)
        return float(np.dot(centroid, direction))

    proj_4 = {"C1": {}, "W1": {}}
    behavior_4 = {"C1": {}, "W1": {}}
    for style in STYLE_NON_THINK:
        uids = list(masks.get("mask_cross", []))
        proj_4["C1"][style] = _level_projection(style, "C1_note", uids)
        proj_4["W1"][style] = _level_projection(style, "W1_note", uids)

        c1_vals = []
        w1_vals = []
        for uid in uids:
            info = outcomes.get(uid, {}).get(style, {})
            c1 = info.get("c1_correction")
            w1 = info.get("w1_flip")
            if c1 is not None:
                c1_vals.append(1 if bool(c1) else 0)
            if bool(info.get("n0_correct", False)) and w1 is not None:
                w1_vals.append(1 if bool(w1) else 0)
        behavior_4["C1"][style] = float(np.mean(c1_vals)) if c1_vals else None
        behavior_4["W1"][style] = float(np.mean(w1_vals)) if w1_vals else None

    proj_5 = {"C1": {}, "W1": {}}
    for style in STYLE_WITH_THINK:
        uids = list(masks.get("mask_think", []))
        proj_5["C1"][style] = _level_projection(style, "C1_note", uids)
        proj_5["W1"][style] = _level_projection(style, "W1_note", uids)

    # Spearman: projection vs behavior (W1 is usually the key)
    xs, ys = [], []
    for style in STYLE_NON_THINK:
        x = proj_4["W1"].get(style)
        y = behavior_4["W1"].get(style)
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    spearman_w1 = None
    if len(xs) >= 3:
        corr, pval = spearmanr(xs, ys)
        spearman_w1 = {"rho": float(corr), "pvalue": float(pval)}

    # C1/W1 full-space trajectory vectors: auth - weak
    uids = list(masks.get("mask_cross", []))
    weak_c1 = store.matrix(position=position, layer_idx=layer_idx, uids=uids, style="weak", condition_code="C1_note")
    auth_c1 = store.matrix(position=position, layer_idx=layer_idx, uids=uids, style="authoritative_verified", condition_code="C1_note")
    weak_w1 = store.matrix(position=position, layer_idx=layer_idx, uids=uids, style="weak", condition_code="W1_note")
    auth_w1 = store.matrix(position=position, layer_idx=layer_idx, uids=uids, style="authoritative_verified", condition_code="W1_note")

    trajectory = {}
    if min(len(weak_c1), len(auth_c1), len(weak_w1), len(auth_w1)) > 0:
        vec_c1 = auth_c1.mean(axis=0) - weak_c1.mean(axis=0)
        vec_w1 = auth_w1.mean(axis=0) - weak_w1.mean(axis=0)
        cos_v = _cosine(vec_c1, vec_w1)
        cos_clamped = max(-1.0, min(1.0, cos_v))
        angle_deg = float(np.degrees(np.arccos(cos_clamped)))
        trajectory = {
            "trajectory_cosine": cos_v,
            "trajectory_angle_degrees": angle_deg,
            "c1_norm": float(np.linalg.norm(vec_c1)),
            "w1_norm": float(np.linalg.norm(vec_w1)),
        }
    else:
        trajectory = {
            "trajectory_cosine": None,
            "trajectory_angle_degrees": None,
            "c1_norm": None,
            "w1_norm": None,
        }

    return {
        "projection_4level": proj_4,
        "projection_5level": proj_5,
        "behavior_4level": behavior_4,
        "spearman_w1_projection_vs_behavior": spearman_w1,
        "trajectory": trajectory,
    }


def _plot_projection_4level(result: Dict[str, object], out_path: Path) -> None:
    proj = result["projection_4level"]
    xs = list(range(len(STYLE_NON_THINK)))
    labels = list(STYLE_NON_THINK)
    c1 = [proj["C1"].get(s) for s in labels]
    w1 = [proj["W1"].get(s) for s in labels]

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(xs, c1, marker="o", label="C1 projection")
    plt.plot(xs, w1, marker="o", label="W1 projection")
    plt.xticks(xs, labels, rotation=25)
    plt.ylabel("Projection")
    plt.title("4-level authority projection (primary direction)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_trajectory_c1_w1(result: Dict[str, object], out_path: Path) -> None:
    proj = result["projection_4level"]
    labels = list(STYLE_NON_THINK)
    xs = list(range(len(labels)))
    c1 = [proj["C1"].get(s) for s in labels]
    w1 = [proj["W1"].get(s) for s in labels]
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(xs, c1, marker="o", linewidth=2, label="C1 trajectory")
    plt.plot(xs, w1, marker="o", linewidth=2, label="W1 trajectory")
    for i, style in enumerate(labels):
        plt.annotate(style, (xs[i], c1[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    plt.xticks(xs, labels, rotation=25)
    plt.ylabel("Projection")
    plt.title("C1 vs W1 trajectories across authority levels")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_cosine_matrix(cos_mat: Dict[str, Dict[str, float]], out_path: Path) -> None:
    names = sorted(cos_mat.keys())
    mat = np.zeros((len(names), len(names)), dtype=np.float64)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            mat[i, j] = cos_mat[a][b]
    plt.figure(figsize=(8, 7))
    im = plt.imshow(mat, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar(im, label="Cosine similarity")
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.yticks(range(len(names)), names)
    plt.title("Direction cosine similarity matrix")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_null_distribution(shuffled: Dict[str, object], out_path: Path) -> None:
    vals = shuffled.get("auroc_values", []) or []
    plt.figure(figsize=(7, 4.2))
    if vals:
        plt.hist(vals, bins=20, alpha=0.8, edgecolor="black")
    plt.xlabel("AUROC")
    plt.ylabel("Count")
    plt.title("Shuffled-label AUROC distribution")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.extraction_dir.parent / "gpt_oss_compliance_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(args.extraction_dir / "metadata.jsonl")
    payload = torch.load(args.extraction_dir / "activations.pt", map_location="cpu")
    masks = json.loads((args.extraction_dir / "label_masks.json").read_text())
    store = ActivationStore(metadata, payload)

    w1_uids, w1_y = _task_data_from_masks(masks, task="w1")
    c1_uids, c1_y = _task_data_from_masks(masks, task="c1")
    if len(w1_uids) == 0 or len(c1_uids) == 0:
        raise SystemExit("Missing W1 or C1 task data after mask filtering.")

    # Nested CV with leakage-safe layer/position selection
    nested_w1 = _nested_task_eval(
        store,
        task="w1",
        uids=w1_uids,
        y=w1_y,
        layer_values=store.target_layers,
        position_names=store.position_names,
        cv_folds=args.cv_folds,
        inner_cv_folds=args.inner_cv_folds,
        seed=args.seed,
        n_jobs=args.n_jobs,
        parallel_prefer=args.parallel_prefer,
        blas_threads=args.blas_threads,
    )
    nested_c1 = _nested_task_eval(
        store,
        task="c1",
        uids=c1_uids,
        y=c1_y,
        layer_values=store.target_layers,
        position_names=store.position_names,
        cv_folds=args.cv_folds,
        inner_cv_folds=args.inner_cv_folds,
        seed=args.seed + 777,
        n_jobs=args.n_jobs,
        parallel_prefer=args.parallel_prefer,
        blas_threads=args.blas_threads,
    )

    # Heatmaps for full pair landscape
    heat_w1, heat_positions, heat_layers = _make_heatmap(
        store,
        task="w1",
        uids=w1_uids,
        y=w1_y,
        output_path=figures_dir / "heatmap_auroc_w1.png",
        cv_folds=args.cv_folds,
        seed=args.seed,
        n_jobs=args.n_jobs,
        parallel_prefer=args.parallel_prefer,
        blas_threads=args.blas_threads,
    )
    heat_c1, _, _ = _make_heatmap(
        store,
        task="c1",
        uids=c1_uids,
        y=c1_y,
        output_path=figures_dir / "heatmap_auroc_c1.png",
        cv_folds=args.cv_folds,
        seed=args.seed + 3,
        n_jobs=args.n_jobs,
        parallel_prefer=args.parallel_prefer,
        blas_threads=args.blas_threads,
    )

    # Primary pair: use the nested-CV-selected W1 pair (strongest behavioral signal)
    primary_position = nested_w1["primary_pair"]["position"]
    primary_layer_val = nested_w1["primary_pair"]["layer"]
    primary_layer_idx = store.layer_to_index[primary_layer_val]
    primary_pi = heat_positions.index(primary_position)
    primary_li = heat_layers.index(primary_layer_val)
    _primary_vals = [v for v in (heat_w1[primary_pi, primary_li], heat_c1[primary_pi, primary_li]) if not np.isnan(v)]
    best_score = float(np.mean(_primary_vals)) if _primary_vals else None

    # Primary direction from full primary mask
    dirs_full = _direction_from_uids(
        store,
        uids=list(masks["mask_primary"]),
        position=primary_position,
        layer_idx=primary_layer_idx,
    )
    primary_direction = dirs_full["shared_within_label"]
    primary_payload = {
        "position": primary_position,
        "layer": primary_layer_val,
        "direction_name": "shared_within_label",
        "vector": torch.tensor(primary_direction, dtype=torch.float32),
    }
    torch.save(primary_payload, output_dir / "primary_direction.pt")

    # Direction pack
    direction_pack = {
        "position": primary_position,
        "layer": primary_layer_val,
        **{name: torch.tensor(vec, dtype=torch.float32) for name, vec in dirs_full.items()},
    }
    torch.save(direction_pack, output_dir / "directions.pt")

    # Cosine matrix
    dir_names = list(dirs_full.keys())
    cos_matrix: Dict[str, Dict[str, float]] = {a: {} for a in dir_names}
    for a in dir_names:
        for b in dir_names:
            cos_matrix[a][b] = _cosine(dirs_full[a], dirs_full[b])
    _plot_cosine_matrix(cos_matrix, figures_dir / "cosine_similarity_matrix.png")

    # Task metrics at primary pair + CIs
    rng = np.random.default_rng(args.seed)
    w1_x = store.matrix(
        position=primary_position,
        layer_idx=primary_layer_idx,
        uids=w1_uids,
        style="authoritative_verified",
        condition_code="W1_note",
    )
    w1_scores = w1_x @ primary_direction
    c1_x = store.matrix(
        position=primary_position,
        layer_idx=primary_layer_idx,
        uids=c1_uids,
        style="authoritative_verified",
        condition_code="C1_note",
    )
    c1_scores = c1_x @ primary_direction

    w1_auc = _safe_auroc(w1_y, w1_scores)
    c1_auc = _safe_auroc(c1_y, c1_scores)
    w1_auc_oriented = _oriented_auroc(w1_y, w1_scores)
    c1_auc_oriented = _oriented_auroc(c1_y, c1_scores)
    w1_ci = _bootstrap_ci(w1_y, w1_scores, rng)
    c1_ci = _bootstrap_ci(c1_y, c1_scores, rng)

    # Baselines
    baseline_majority_w1 = _majority_baseline(w1_y)
    baseline_majority_c1 = _majority_baseline(c1_y)
    baseline_level_w1 = _authority_level_only_baseline(masks, task="w1", cv_folds=args.cv_folds, seed=args.seed)
    baseline_level_c1 = _authority_level_only_baseline(masks, task="c1", cv_folds=args.cv_folds, seed=args.seed + 1)
    baseline_random_w1 = _random_direction_baseline(
        store,
        task="w1",
        uids=w1_uids,
        y=w1_y,
        position=primary_position,
        layer_val=primary_layer_val,
        n_runs=args.random_direction_runs,
        cv_folds=args.cv_folds,
        seed=args.seed,
        n_jobs=args.n_jobs,
        parallel_prefer=args.parallel_prefer,
        blas_threads=args.blas_threads,
    )
    baseline_random_c1 = _random_direction_baseline(
        store,
        task="c1",
        uids=c1_uids,
        y=c1_y,
        position=primary_position,
        layer_val=primary_layer_val,
        n_runs=args.random_direction_runs,
        cv_folds=args.cv_folds,
        seed=args.seed + 2,
        n_jobs=args.n_jobs,
        parallel_prefer=args.parallel_prefer,
        blas_threads=args.blas_threads,
    )
    baseline_shuffled_w1 = _shuffled_label_baseline(
        w1_y,
        w1_scores,
        runs=args.shuffled_label_runs,
        cv_folds=args.cv_folds,
        seed=args.seed,
        n_jobs=args.n_jobs,
        parallel_prefer=args.parallel_prefer,
        blas_threads=args.blas_threads,
    )
    baseline_shuffled_c1 = _shuffled_label_baseline(
        c1_y,
        c1_scores,
        runs=args.shuffled_label_runs,
        cv_folds=args.cv_folds,
        seed=args.seed + 4,
        n_jobs=args.n_jobs,
        parallel_prefer=args.parallel_prefer,
        blas_threads=args.blas_threads,
    )
    baseline_raw_random_w1 = _random_raw_projection_baseline(
        w1_x,
        w1_y,
        runs=args.random_direction_runs,
        seed=args.seed + 13,
    )
    baseline_raw_random_c1 = _random_raw_projection_baseline(
        c1_x,
        c1_y,
        runs=args.random_direction_runs,
        seed=args.seed + 17,
    )
    _plot_null_distribution(baseline_shuffled_w1, figures_dir / "null_distribution.png")

    # Direction-specific checks for C1 vs W1 mechanisms.
    w1_wrong_direction_auc = _simple_cv_pair_auroc_with_direction(
        store,
        task="w1",
        uids=w1_uids,
        y=w1_y,
        position=primary_position,
        layer_val=primary_layer_val,
        cv_folds=args.cv_folds,
        seed=args.seed + 101,
        direction_name="authority_given_wrong",
    )
    c1_correct_direction_auc = _simple_cv_pair_auroc_with_direction(
        store,
        task="c1",
        uids=c1_uids,
        y=c1_y,
        position=primary_position,
        layer_val=primary_layer_val,
        cv_folds=args.cv_folds,
        seed=args.seed + 103,
        direction_name="authority_given_correct",
    )

    # Projection + trajectory diagnostics
    trajectory = _trajectory_and_projection(
        store,
        masks=masks,
        position=primary_position,
        layer_val=primary_layer_val,
        direction=primary_direction,
    )
    _plot_projection_4level(trajectory, figures_dir / "projection_4level.png")
    _plot_trajectory_c1_w1(trajectory, figures_dir / "trajectory_c1_w1.png")

    # Direction stability across CV folds at primary pair
    class_counts = np.bincount(w1_y)
    minority = int(class_counts.min()) if len(class_counts) >= 2 else 0
    n_splits = max(2, min(args.cv_folds, minority))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed + 55)
    uid_arr = np.array(w1_uids)
    fold_dirs: List[np.ndarray] = []
    for tr, _ in splitter.split(np.zeros(len(w1_y)), w1_y):
        train_uids = uid_arr[tr].tolist()
        d = _direction_from_uids(
            store,
            uids=train_uids,
            position=primary_position,
            layer_idx=primary_layer_idx,
        )["shared_within_label"]
        fold_dirs.append(d)
    fold_cos = []
    for i, j in combinations(range(len(fold_dirs)), 2):
        fold_cos.append(_cosine(fold_dirs[i], fold_dirs[j]))

    null_controls = {
        "random_direction_w1": baseline_random_w1,
        "random_direction_c1": baseline_random_c1,
        "random_raw_projection_w1": baseline_raw_random_w1,
        "random_raw_projection_c1": baseline_raw_random_c1,
        "shuffled_label_w1": baseline_shuffled_w1,
        "shuffled_label_c1": baseline_shuffled_c1,
        "fold_direction_stability": {
            "pairwise_cosines": fold_cos,
            "mean_cosine": float(np.mean(fold_cos)) if fold_cos else None,
            "std_cosine": float(np.std(fold_cos)) if fold_cos else None,
        },
    }

    # Save outputs
    (output_dir / "cosine_similarities.json").write_text(json.dumps(cos_matrix, indent=2))
    (output_dir / "layer_position_sweep.json").write_text(
        json.dumps(
            {
                "positions": heat_positions,
                "layers": heat_layers,
                "w1_auroc_matrix": np.where(np.isnan(heat_w1), None, heat_w1).tolist(),
                "c1_auroc_matrix": np.where(np.isnan(heat_c1), None, heat_c1).tolist(),
            },
            indent=2,
        )
    )
    probe_results = {
        "nested_w1": nested_w1,
        "nested_c1": nested_c1,
        "primary_pair": {"position": primary_position, "layer": primary_layer_val, "mean_heatmap_auroc": best_score},
        "primary_pair_metrics": {
            "w1_auroc": w1_auc,
            "w1_auroc_oriented": w1_auc_oriented,
            "w1_auroc_ci95": {"low": w1_ci[0], "high": w1_ci[1]},
            "c1_auroc": c1_auc,
            "c1_auroc_oriented": c1_auc_oriented,
            "c1_auroc_ci95": {"low": c1_ci[0], "high": c1_ci[1]},
            "w1_n": int(len(w1_y)),
            "c1_n": int(len(c1_y)),
            "pvalue_vs_random_direction_w1": _empirical_right_tail_pvalue(
                nested_w1["metrics"]["auroc"], baseline_random_w1.get("auroc_values", [])
            ),
            "pvalue_vs_shuffled_w1": _empirical_right_tail_pvalue(
                nested_w1["metrics"]["auroc"], baseline_shuffled_w1.get("auroc_values", [])
            ),
        },
        "task_specific_direction_checks": {
            "w1_shared_direction_cv_auroc": _simple_cv_pair_auroc_with_direction(
                store,
                task="w1",
                uids=w1_uids,
                y=w1_y,
                position=primary_position,
                layer_val=primary_layer_val,
                cv_folds=args.cv_folds,
                seed=args.seed + 99,
                direction_name="shared_within_label",
            ),
            "w1_wrong_direction_cv_auroc": w1_wrong_direction_auc,
            "c1_shared_direction_cv_auroc": _simple_cv_pair_auroc_with_direction(
                store,
                task="c1",
                uids=c1_uids,
                y=c1_y,
                position=primary_position,
                layer_val=primary_layer_val,
                cv_folds=args.cv_folds,
                seed=args.seed + 100,
                direction_name="shared_within_label",
            ),
            "c1_correct_direction_cv_auroc": c1_correct_direction_auc,
        },
        "baselines": {
            "majority_w1": baseline_majority_w1,
            "majority_c1": baseline_majority_c1,
            "authority_level_only_w1": baseline_level_w1,
            "authority_level_only_c1": baseline_level_c1,
        },
    }
    (output_dir / "probe_results.json").write_text(json.dumps(probe_results, indent=2))
    (output_dir / "projection_analysis.json").write_text(json.dumps(trajectory, indent=2))
    (output_dir / "trajectory_analysis.json").write_text(json.dumps(trajectory.get("trajectory", {}), indent=2))
    (output_dir / "null_controls.json").write_text(json.dumps(null_controls, indent=2))

    summary = {
        "extraction_dir": str(args.extraction_dir),
        "output_dir": str(output_dir),
        "n_metadata_rows": len(metadata),
        "mask_counts": masks.get("counts", {}),
        "primary_pair": {"position": primary_position, "layer": primary_layer_val},
        "files": {
            "directions": str(output_dir / "directions.pt"),
            "primary_direction": str(output_dir / "primary_direction.pt"),
            "cosine_similarities": str(output_dir / "cosine_similarities.json"),
            "layer_position_sweep": str(output_dir / "layer_position_sweep.json"),
            "probe_results": str(output_dir / "probe_results.json"),
            "projection_analysis": str(output_dir / "projection_analysis.json"),
            "trajectory_analysis": str(output_dir / "trajectory_analysis.json"),
            "null_controls": str(output_dir / "null_controls.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
