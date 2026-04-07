"""N0 residual vs actual update correlation test.

Precision-weighting prediction: the N0 probe's confidence-independent
residual score should correlate with actual update magnitude |m_C1 - m_N0|.
If the residual IS a pre-computed updatability estimate, items with higher
residual scores should show larger actual updates.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def signed_margin(row, cond):
    cr = row["condition_results"][cond]
    if row["correct_label"] == "A":
        return cr["logit_a"] - cr["logit_b"]
    return cr["logit_b"] - cr["logit_a"]


def run_model(name, n0_act_dir, mech_dir, behav_path, tag, best_layer):
    print(f"\n{'='*60}")
    print(f"  {name} (layer {best_layer}, tag={tag})")
    print(f"{'='*60}")

    # Load N0 activations
    payload = torch.load(n0_act_dir / "activations.pt", map_location="cpu")
    n0_meta = load_jsonl(n0_act_dir / "metadata.jsonl")
    acts = payload["activations"]["endorsement_last"]  # [N, n_layers, d]

    # Load mechanism labels
    mech_meta = load_jsonl(mech_dir / "metadata.jsonl")
    uid_to_label = {}
    for row in mech_meta:
        if row["variant_label"] == "i1a":
            uid_to_label[row["uid"]] = row["item_label"]

    # Load confidence from selected_items
    uid_to_m_n0 = {}
    sel_path = mech_dir / "selected_items.jsonl"
    if sel_path.exists():
        for line in open(sel_path):
            item = json.loads(line.strip())
            uid_to_m_n0[item["uid"]] = item.get("m_n0", 0.0)

    # Load behavioral results for actual update magnitude
    behav = {r["uid"]: r for r in load_jsonl(behav_path)}

    # Match items
    indices, labels, confidences, actual_updates, uids = [], [], [], [], []
    for i, row in enumerate(n0_meta):
        uid = row["uid"]
        if uid in uid_to_label and uid in behav:
            indices.append(i)
            labels.append(1 if uid_to_label[uid] == "correcting" else 0)
            confidences.append(uid_to_m_n0.get(uid, 0.0))
            # Actual update: |m_C1 - m_N0| (how much the model changed under correct endorsement + instruction)
            br = behav[uid]
            m_n0 = signed_margin(br, f"N0_{tag}")
            m_c1 = signed_margin(br, f"C1_{tag}")
            actual_updates.append(m_c1 - m_n0)
            uids.append(uid)

    X = acts[torch.tensor(indices), best_layer, :].float().numpy()
    y = np.array(labels)
    conf = np.array(confidences)
    updates = np.array(actual_updates)

    print(f"  Items: {len(y)} (correcting={sum(y)}, resisting={len(y)-sum(y)})")

    # Step 1: Project out confidence direction
    conf_model = Ridge(alpha=100.0).fit(X, conf)
    conf_dir = conf_model.coef_ / np.linalg.norm(conf_model.coef_)
    X_residual = X - np.outer(X @ conf_dir, conf_dir)

    # Step 2: Get cross-validated residual probe scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    residual_scores = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X_residual, y):
        probe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, solver="saga",
                                       class_weight="balanced", random_state=0))
        ])
        probe.fit(X_residual[train_idx], y[train_idx])
        residual_scores[test_idx] = probe.predict_proba(X_residual[test_idx])[:, 1]

    # Step 3: Correlate residual scores with actual update magnitude
    r_pearson, p_pearson = pearsonr(residual_scores, updates)
    r_spearman, p_spearman = spearmanr(residual_scores, updates)

    print(f"\n  PRECISION-WEIGHTING TEST:")
    print(f"  Residual probe score vs actual update (m_C1 - m_N0):")
    print(f"    Pearson r  = {r_pearson:.4f} (p = {p_pearson:.4e})")
    print(f"    Spearman r = {r_spearman:.4f} (p = {p_spearman:.4e})")

    # Also: does confidence alone predict updates? (sanity check)
    r_conf, p_conf = pearsonr(conf, updates)
    print(f"\n  CONTROL: Confidence vs actual update:")
    print(f"    Pearson r  = {r_conf:.4f} (p = {p_conf:.4e})")

    # Partial correlation: residual vs updates, controlling for confidence
    # Use residual of updates after regressing out confidence
    from sklearn.linear_model import LinearRegression
    update_resid = updates - LinearRegression().fit(conf.reshape(-1, 1), updates).predict(conf.reshape(-1, 1))
    r_partial, p_partial = pearsonr(residual_scores, update_resid)
    print(f"\n  PARTIAL CORRELATION (controlling for confidence):")
    print(f"    Pearson r  = {r_partial:.4f} (p = {p_partial:.4e})")

    # Bin analysis: do high-residual items update more?
    terciles = np.percentile(residual_scores, [33, 67])
    print(f"\n  BIN ANALYSIS (residual score terciles):")
    for name, lo, hi in [("Low residual", 0, terciles[0]),
                          ("Mid residual", terciles[0], terciles[1]),
                          ("High residual", terciles[1], 1.01)]:
        mask = (residual_scores >= lo) & (residual_scores < hi)
        if mask.sum() == 0:
            continue
        mean_update = updates[mask].mean()
        mean_conf = conf[mask].mean()
        flip_rate = (updates[mask] > -conf[mask]).mean()  # m_C1 > 0 means corrected
        print(f"    {name} (N={mask.sum()}): mean_update={mean_update:+.3f}, mean_conf={mean_conf:.3f}")

    return {
        "model": name,
        "n_items": len(y),
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "spearman_r": r_spearman,
        "spearman_p": p_spearman,
        "confidence_r": r_conf,
        "partial_r": r_partial,
        "partial_p": p_partial,
    }


def main():
    base = Path("new-phase-results")

    results = []

    # Qwen3-4B
    results.append(run_model(
        "Qwen3-4B",
        n0_act_dir=base / "mechanism/qwen_n0_note_activations",
        mech_dir=base / "mechanism/qwen_correction_gating_i1a_i1c_note",
        behav_path=base / "qwen3-4b-results/exp10_extended/Qwen__Qwen3-4B-Instruct-2507_results.jsonl",
        tag="expert",
        best_layer=28,
    ))

    # GPT-oss-20B
    results.append(run_model(
        "GPT-oss-20B",
        n0_act_dir=base / "mechanism/gpt_oss_n0_note_activations",
        mech_dir=base / "mechanism/gpt_oss_20b_correction_gating_i1a_i1c_note",
        behav_path=base / "gpt_oss_20b-results/exp10_i1a_note/openai__gpt-oss-20b_results.jsonl",
        tag="note",
        best_layer=21,
    ))

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['model']}: residual-update r={r['pearson_r']:.4f} (p={r['pearson_p']:.2e}), "
              f"partial r={r['partial_r']:.4f} (p={r['partial_p']:.2e})")
    
    out = Path("new-phase-results/mechanism/n0_residual_correlation.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
