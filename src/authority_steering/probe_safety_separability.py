"""Linear-probe separability of {N0, C1, W1}_safety activations.

Computes per-(layer, position) AUC for pairwise contrasts against two nulls
(label-shuffle and random-direction) and optionally a note-shuffle extraction.

Inputs:
  --real-dir     extraction dir from `extract_safety_matched_activations.py` (real notes)
  --shuffle-dir  extraction dir from the same script with --note-shuffle

Outputs to <real-dir>:
  separability_summary.json
  separability_summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


POSITIONS = ("endorsement_start", "endorsed_answer", "endorsement_end", "answer_position", "endorsement_mean")
NON_LEXICAL_POSITIONS = ("answer_position", "endorsement_mean")
CONTRASTS = (
    ("W1_safety", "C1_safety"),
    ("W1_safety", "N0_safety"),
    ("C1_safety", "N0_safety"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--real-dir", type=Path, required=True)
    p.add_argument("--shuffle-dir", type=Path, default=None)
    p.add_argument("--n-random-nulls", type=int, default=50)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-prefix", type=str, default="separability_summary")
    return p.parse_args()


def _load_activations(run_dir: Path) -> Tuple[Dict, List[Dict]]:
    pt = torch.load(run_dir / "activations.pt", map_location="cpu")
    meta = []
    with (run_dir / "metadata.jsonl").open() as f:
        for line in f:
            meta.append(json.loads(line))
    return pt, meta


def _get_features(
    activations: Dict,
    metadata: List[Dict],
    *,
    layer: int,
    position: str,
    cond_a: str,
    cond_b: str,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Returns X (n, d), y (n,), pair_ids (for paired contrast)."""
    target_layers: List[int] = activations["target_layers"]
    if layer not in target_layers:
        raise KeyError(f"layer {layer} not in extraction target_layers={target_layers}")
    li = target_layers.index(layer)
    tensor = activations["activations"][position]  # (n_records, n_layers, d)
    tensor = tensor[:, li, :].float().numpy()

    xs_a: List[np.ndarray] = []
    xs_b: List[np.ndarray] = []
    pid_a: List[int] = []
    pid_b: List[int] = []
    for row_i, rec in enumerate(metadata):
        code = rec["condition_code"]
        pid = rec["prompt_id"]
        if code == cond_a:
            xs_a.append(tensor[row_i])
            pid_a.append(pid)
        elif code == cond_b:
            xs_b.append(tensor[row_i])
            pid_b.append(pid)
    # align by prompt id
    common = sorted(set(pid_a) & set(pid_b))
    map_a = {pid: xs_a[pid_a.index(pid)] for pid in common}
    map_b = {pid: xs_b[pid_b.index(pid)] for pid in common}
    X = np.stack([map_a[pid] for pid in common] + [map_b[pid] for pid in common])
    y = np.array([1] * len(common) + [0] * len(common))
    pair_ids = common + common
    return X, y, pair_ids


def _cv_auc(X: np.ndarray, y: np.ndarray, *, seed: int, folds: int) -> Tuple[float, float]:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aucs = []
    for train, test in skf.split(X, y):
        Xtr, Xte = X[train], X[test]
        ytr, yte = y[train], y[test]
        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)
        clf = LogisticRegression(C=1.0, max_iter=5000, solver="liblinear")
        clf.fit(Xtr, ytr)
        probs = clf.decision_function(Xte)
        aucs.append(roc_auc_score(yte, probs))
    return float(np.mean(aucs)), float(np.std(aucs))


def _cv_auc_shuffle(X: np.ndarray, y: np.ndarray, *, seed: int, folds: int) -> float:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    aucs = []
    for train, test in skf.split(X, y):
        ytr = y[train].copy()
        rng.shuffle(ytr)
        Xtr, Xte = X[train], X[test]
        yte = y[test]
        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)
        clf = LogisticRegression(C=1.0, max_iter=5000, solver="liblinear")
        clf.fit(Xtr, ytr)
        probs = clf.decision_function(Xte)
        aucs.append(roc_auc_score(yte, probs))
    return float(np.mean(aucs))


def _random_direction_auc(X: np.ndarray, y: np.ndarray, *, seeds: int) -> float:
    d = X.shape[1]
    rng = np.random.default_rng(12345)
    aucs = []
    # standardize once (same view as lr)
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-8)
    for _ in range(seeds):
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v) + 1e-12
        scores = Xn @ v
        try:
            auc = roc_auc_score(y, scores)
        except Exception:
            auc = 0.5
        aucs.append(max(auc, 1 - auc))  # direction-sign-invariant
    return float(np.mean(aucs))


def _walk(activations: Dict, metadata: List[Dict], *, seed: int, folds: int, n_random: int) -> List[Dict]:
    target_layers: List[int] = activations["target_layers"]
    position_names = activations["position_names"]
    out: List[Dict] = []
    for layer in target_layers:
        for position in position_names:
            for cond_a, cond_b in CONTRASTS:
                X, y, _ = _get_features(activations, metadata, layer=layer, position=position, cond_a=cond_a, cond_b=cond_b)
                auc_real, auc_std = _cv_auc(X, y, seed=seed, folds=folds)
                auc_shuf = _cv_auc_shuffle(X, y, seed=seed, folds=folds)
                auc_rand = _random_direction_auc(X, y, seeds=n_random)
                out.append({
                    "layer": layer,
                    "position": position,
                    "contrast": f"{cond_a}_vs_{cond_b}",
                    "n": int(len(y)),
                    "auc_real": auc_real,
                    "auc_real_std": auc_std,
                    "auc_null_shuffle": auc_shuf,
                    "auc_null_random": auc_rand,
                    "delta_vs_shuffle": auc_real - auc_shuf,
                    "delta_vs_random": auc_real - auc_rand,
                })
    return out


def main() -> None:
    args = parse_args()
    real_pt, real_meta = _load_activations(args.real_dir)
    results_real = _walk(real_pt, real_meta, seed=args.seed, folds=args.cv_folds, n_random=args.n_random_nulls)

    results_noteshuf: List[Dict] = []
    if args.shuffle_dir is not None:
        shuf_pt, shuf_meta = _load_activations(args.shuffle_dir)
        results_noteshuf = _walk(shuf_pt, shuf_meta, seed=args.seed, folds=args.cv_folds, n_random=args.n_random_nulls)

    # join by (layer, position, contrast)
    key = lambda r: (r["layer"], r["position"], r["contrast"])
    shuf_by_key = {key(r): r["auc_real"] for r in results_noteshuf}
    merged = []
    for r in results_real:
        r2 = dict(r)
        r2["auc_note_shuffle"] = shuf_by_key.get(key(r))
        if r2["auc_note_shuffle"] is not None:
            r2["delta_vs_note_shuffle"] = r2["auc_real"] - r2["auc_note_shuffle"]
        merged.append(r2)

    # gate
    w1_vs_c1 = [r for r in merged if r["contrast"] == "W1_safety_vs_C1_safety" and r["position"] in NON_LEXICAL_POSITIONS]
    w1_vs_c1.sort(key=lambda r: -r["auc_real"])
    winner = w1_vs_c1[0] if w1_vs_c1 else None

    gate_ok = False
    gate_info: Dict = {}
    if winner is not None:
        passes_shuffle = winner["delta_vs_shuffle"] >= 0.05
        passes_note = (winner.get("auc_note_shuffle") is None) or (winner.get("delta_vs_note_shuffle", 0.0) >= 0.05)
        gate_ok = bool(passes_shuffle and passes_note)
        gate_info = {
            "winner": winner,
            "passes_vs_shuffle_null": passes_shuffle,
            "passes_vs_note_shuffle": passes_note,
        }

    payload = {
        "seed": args.seed,
        "cv_folds": args.cv_folds,
        "real_dir": str(args.real_dir),
        "shuffle_dir": str(args.shuffle_dir) if args.shuffle_dir else None,
        "non_lexical_positions": list(NON_LEXICAL_POSITIONS),
        "gate_passed": gate_ok,
        "gate_info": gate_info,
        "results": merged,
    }
    out_json = args.real_dir / f"{args.output_prefix}.json"
    out_json.write_text(json.dumps(payload, indent=2))

    # markdown
    md = ["# Safety-activation separability", "", f"- real_dir: `{args.real_dir}`", f"- shuffle_dir: `{args.shuffle_dir}`", f"- gate_passed: **{gate_ok}**", ""]
    if winner is not None:
        md.append(f"- winner (non-lexical position, W1 vs C1): layer={winner['layer']} position={winner['position']} AUC={winner['auc_real']:.3f}")
        md.append(f"    - Δ vs shuffle-null = {winner['delta_vs_shuffle']:+.3f}")
        if winner.get("auc_note_shuffle") is not None:
            md.append(f"    - Δ vs note-shuffle = {winner['delta_vs_note_shuffle']:+.3f} (note-shuf AUC={winner['auc_note_shuffle']:.3f})")
        md.append("")
    md.append("| layer | position | contrast | AUC | ±std | shuffle | random | note-shuf | Δshuf | Δnote |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in merged:
        ns = f"{r['auc_note_shuffle']:.3f}" if r.get("auc_note_shuffle") is not None else "-"
        dn = f"{r['delta_vs_note_shuffle']:+.3f}" if r.get("delta_vs_note_shuffle") is not None else "-"
        md.append(f"| {r['layer']} | {r['position']} | {r['contrast']} | {r['auc_real']:.3f} | {r['auc_real_std']:.3f} | {r['auc_null_shuffle']:.3f} | {r['auc_null_random']:.3f} | {ns} | {r['delta_vs_shuffle']:+.3f} | {dn} |")
    (args.real_dir / f"{args.output_prefix}.md").write_text("\n".join(md) + "\n")

    print(json.dumps({"gate_passed": gate_ok, "winner": gate_info.get("winner")}, indent=2))
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
