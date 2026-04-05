"""Pre-endorsement probe: can N0 activations predict flip/entrench under i1a?

If this works, the model's disposition to integrate or resist evidence is
PREDETERMINED by how it internally holds the belief -- before any instruction
or endorsement arrives.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n0-extraction-dir", type=Path, required=True,
                        help="Directory with N0 condition activations")
    parser.add_argument("--mechanism-extraction-dir", type=Path, required=True,
                        help="Directory with i1a/i1c mechanism extraction (for labels)")
    parser.add_argument("--position", type=str, default="endorsement_last")
    parser.add_argument("--label-mode", type=str, default="primary",
                        choices=("primary", "expanded"),
                        help="primary: correcting/resisting; expanded: use expanded_resisting")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def load_metadata(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_probe(seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, solver="saga",
                                    class_weight="balanced", random_state=seed)),
    ])


def main():
    args = parse_args()

    # Load N0 activations
    n0_meta = load_metadata(args.n0_extraction_dir / "metadata.jsonl")
    n0_payload = torch.load(args.n0_extraction_dir / "activations.pt", map_location="cpu")
    n0_acts = n0_payload["activations"][args.position]

    # Load mechanism labels (from i1a extraction)
    mech_meta = load_metadata(args.mechanism_extraction_dir / "metadata.jsonl")

    # Build UID -> label mapping from mechanism extraction (i1a variant only)
    uid_to_label = {}
    for row in mech_meta:
        if row["variant_label"] == "i1a":
            if args.label_mode == "expanded":
                is_negative = bool(row.get("selection", {}).get("expanded_resisting", False))
                uid_to_label[row["uid"]] = "resisting" if is_negative else "correcting"
            else:
                uid_to_label[row["uid"]] = row["item_label"]

    # Also get the confidence (m_N0) for each item from the mechanism selection metadata
    uid_to_m_n0 = {}
    selected_items_path = args.mechanism_extraction_dir / "selected_items.jsonl"
    if selected_items_path.exists():
        for line in open(selected_items_path):
            item = json.loads(line.strip())
            uid_to_m_n0[item["uid"]] = item.get("m_n0", item.get("prior_strength", None))

    # Get correct_label per item for position confounder control
    uid_to_correct_label = {}
    if selected_items_path.exists():
        for line in open(selected_items_path):
            item = json.loads(line.strip())
            uid_to_correct_label[item["uid"]] = item.get("correct_label", None)

    # Match N0 activations to labels
    indices = []
    labels = []
    confidences = []
    positions = []  # 1 if correct_label == "A", 0 if "B"
    for i, row in enumerate(n0_meta):
        uid = row["uid"]
        if uid in uid_to_label:
            indices.append(i)
            labels.append(1 if uid_to_label[uid] == "correcting" else 0)
            if uid in uid_to_m_n0:
                confidences.append(uid_to_m_n0[uid])
            else:
                confidences.append(None)
            cl = uid_to_correct_label.get(uid, row.get("correct_label"))
            positions.append(1.0 if cl == "A" else 0.0)

    n_layers = n0_acts.shape[1]
    X_all = n0_acts[torch.tensor(indices)].float().numpy()
    y = np.array(labels, dtype=np.int64)
    confidences = np.array([c if c is not None else 0.0 for c in confidences])
    positions = np.array(positions)

    print(f"Matched {len(indices)} items | correcting: {sum(y)} | resisting: {len(y)-sum(y)}")
    print(f"Position: {args.position} | Layers: {n_layers}")
    print(f"Majority baseline: {max(sum(y), len(y)-sum(y))/len(y):.3f}")

    def _eval_layer(layer, X_layer, y, cv_folds, seed):
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        prob = np.zeros(len(y), dtype=np.float64)
        for train_idx, test_idx in splitter.split(X_layer, y):
            probe = build_probe(seed)
            probe.fit(X_layer[train_idx], y[train_idx])
            prob[test_idx] = probe.predict_proba(X_layer[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        acc = accuracy_score(y, pred)
        auroc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else 0.5
        return {"layer": layer, "accuracy": acc, "auroc": auroc}

    n_jobs = os.cpu_count()
    print(f"Sweeping {n_layers} layers across {n_jobs} workers...", flush=True)
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_eval_layer)(layer, X_all[:, layer, :], y, args.cv_folds, args.seed)
        for layer in range(n_layers)
    )
    results.sort(key=lambda r: r["layer"])
    best_layer = max(results, key=lambda r: r["auroc"])["layer"]
    best_auroc = results[best_layer]["auroc"]

    print(f"\n=== BEST LAYER: {best_layer} ===")
    print(f"  N0 probe CV accuracy: {results[best_layer]['accuracy']:.4f}")
    print(f"  N0 probe CV AUROC:    {results[best_layer]['auroc']:.4f}")

    # === CONFIDENCE + POSITION CONTROLS (parallelized) ===
    X_best = X_all[:, best_layer, :]
    has_conf = np.std(confidences) > 0
    has_pos = np.std(positions) > 0

    auroc_conf = auroc_res = auroc_pos = auroc_pos_out = auroc_both_out = None

    if has_conf or has_pos:
        from sklearn.linear_model import Ridge

        # Compute projection directions (fast, no need to parallelize)
        conf_dir = pos_dir = pos_dir_orth = None
        if has_conf:
            cm = Ridge(alpha=1.0).fit(X_best, confidences)
            conf_dir = cm.coef_ / np.linalg.norm(cm.coef_)
        if has_pos:
            pm = Ridge(alpha=1.0).fit(X_best, positions)
            pos_dir = pm.coef_ / np.linalg.norm(pm.coef_)
        if conf_dir is not None and pos_dir is not None:
            pos_dir_orth = pos_dir - np.dot(pos_dir, conf_dir) * conf_dir
            pos_dir_orth = pos_dir_orth / (np.linalg.norm(pos_dir_orth) + 1e-10)

        # Pre-compute all projected X variants
        variants = {}
        if has_conf:
            X_conf_out = X_best - np.outer(X_best @ conf_dir, conf_dir)
            variants["conf_alone"] = confidences.reshape(-1, 1)
            variants["conf_out"] = X_conf_out
        if has_pos:
            variants["pos_alone"] = positions.reshape(-1, 1)
            X_pos_out = X_best - np.outer(X_best @ pos_dir, pos_dir)
            variants["pos_out"] = X_pos_out
        if has_conf and has_pos:
            X_both_out = X_best.copy()
            X_both_out -= np.outer(X_both_out @ conf_dir, conf_dir)
            X_both_out -= np.outer(X_both_out @ pos_dir_orth, pos_dir_orth)
            variants["both_out"] = X_both_out

        # Parallelized: run one CV evaluation per (variant, fold)
        def _eval_fold(name, X_v, y, train_idx, test_idx, seed):
            probe = build_probe(seed)
            probe.fit(X_v[train_idx], y[train_idx])
            probs = probe.predict_proba(X_v[test_idx])[:, 1]
            return name, test_idx, probs

        fold_jobs = []
        splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(X_best, y))
        for vname, X_v in variants.items():
            for train_idx, test_idx in splits:
                fold_jobs.append((vname, X_v, y, train_idx, test_idx, args.seed))

        print(f"Running {len(fold_jobs)} projection CV jobs across all cores...", flush=True)
        fold_results = Parallel(n_jobs=-1, verbose=5)(
            delayed(_eval_fold)(name, Xv, yy, tr, te, s)
            for name, Xv, yy, tr, te, s in fold_jobs
        )

        # Reassemble per-variant probability vectors
        variant_probs = {vn: np.zeros(len(y), dtype=np.float64) for vn in variants}
        for name, test_idx, probs in fold_results:
            variant_probs[name][test_idx] = probs

        def _auroc(probs):
            return roc_auc_score(y, probs) if len(np.unique(y)) == 2 else 0.5

        if has_conf:
            auroc_conf = _auroc(variant_probs["conf_alone"])
            auroc_res = _auroc(variant_probs["conf_out"])
        if has_pos:
            auroc_pos = _auroc(variant_probs["pos_alone"])
            auroc_pos_out = _auroc(variant_probs["pos_out"])
        if has_conf and has_pos:
            auroc_both_out = _auroc(variant_probs["both_out"])

        if has_conf:
            print(f"\n=== CONFIDENCE CONTROL (layer {best_layer}) ===")
            print(f"  Scalar confidence alone AUROC:        {auroc_conf:.4f}")
            print(f"  N0 probe AUROC (full):                {results[best_layer]['auroc']:.4f}")
            print(f"  N0 probe AUROC (confidence projected out): {auroc_res:.4f}")
            ci = "YES" if auroc_res > 0.55 else "WEAK" if auroc_res > 0.52 else "NO"
            print(f"  >>> Confidence-independent signal:     {ci}")

        if has_conf and has_pos:
            print(f"\n=== THREE-WAY DECOMPOSITION (layer {best_layer}) ===")
            print(f"  Scalar confidence alone AUROC:          {auroc_conf:.4f}")
            print(f"  Scalar position alone AUROC:            {auroc_pos:.4f}")
            print(f"  Full N0 probe AUROC:                    {results[best_layer]['auroc']:.4f}")
            print(f"  N0 probe (confidence out):              {auroc_res:.4f}")
            print(f"  N0 probe (position out):                {auroc_pos_out:.4f}")
            print(f"  N0 probe (confidence + position out):   {auroc_both_out:.4f}")
            sig = "YES" if auroc_both_out > 0.55 else "WEAK" if auroc_both_out > 0.52 else "NO"
            print(f"  >>> Residual signal after both removed:  {sig}")

    # Top 5 layers
    sorted_results = sorted(results, key=lambda r: r["auroc"], reverse=True)
    print(f"\nTop 5 layers by AUROC:")
    for r in sorted_results[:5]:
        print(f"  Layer {r['layer']:2d}: acc={r['accuracy']:.4f}, auroc={r['auroc']:.4f}")

    # Save results
    output = {
        "n0_extraction_dir": str(args.n0_extraction_dir),
        "mechanism_extraction_dir": str(args.mechanism_extraction_dir),
        "position": args.position,
        "n_items": len(indices),
        "n_correcting": int(sum(y)),
        "n_resisting": int(len(y) - sum(y)),
        "majority_baseline": float(max(sum(y), len(y)-sum(y))/len(y)),
        "best_layer": best_layer,
        "best_auroc": best_auroc,
        "best_accuracy": results[best_layer]["accuracy"],
        "all_layers": results,
    }
    if auroc_conf is not None:
        output["confidence_alone_auroc"] = float(auroc_conf)
        output["confidence_projected_out_auroc"] = float(auroc_res)
    if auroc_pos is not None:
        output["position_alone_auroc"] = float(auroc_pos)
        output["position_projected_out_auroc"] = float(auroc_pos_out)
    if auroc_both_out is not None:
        output["both_projected_out_auroc"] = float(auroc_both_out)

    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(output, indent=2))
    print(json.dumps({k: v for k, v in output.items() if k != "all_layers"}, indent=2))


if __name__ == "__main__":
    main()
