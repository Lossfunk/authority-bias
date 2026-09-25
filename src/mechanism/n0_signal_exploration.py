"""N0 pre-endorsement signal exploration.

Investigates what the confidence-independent N0 signal actually captures.
Three hypotheses:
1. Familiarity of the wrong belief (common misconception vs accidental wrong)
2. Semantic distance between correct and wrong answers
3. Question structure features that predict susceptibility

Uses Qwen3-4B-Instruct embeddings to compute semantic distances and 
correlates with N0 probe residual scores and actual correction behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, ttest_ind
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.lexical_controls.dataset_mc import load_mc_dataset
from src.lexical_controls.scoring import _tokenize_for_scoring, get_ab_token_ids
from src.models.llama_loader import load_model_and_tokenizer


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_N0_ACT_DIR = Path("new-phase-results/mechanism/qwen_n0_note_activations")
DEFAULT_MECH_DIR = Path("new-phase-results/mechanism/qwen_correction_gating_i1a_i1c_note")
DEFAULT_BEHAV_PATH = Path("new-phase-results/qwen3-4b-results/exp10_extended/Qwen__Qwen3-4B-Instruct-2507_results.jsonl")
DEFAULT_MC_PATH = Path("data/exp7_mc_dataset.jsonl")
DEFAULT_OUTPUT_DIR = Path("new-phase-results/mechanism/n0_signal_exploration")

BEST_LAYER = 28


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def signed_margin(row, cond):
    cr = row["condition_results"][cond]
    if row["correct_label"] == "A":
        return cr["logit_a"] - cr["logit_b"]
    return cr["logit_b"] - cr["logit_a"]


def get_embeddings(model, tokenizer, texts, device):
    """Get mean-pooled last-hidden-state embeddings for a list of texts."""
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        last_hidden = out.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        mean_emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
        embeddings.append(mean_emb[0].cpu().float().numpy())
    return np.array(embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--n0-act-dir", type=Path, default=DEFAULT_N0_ACT_DIR)
    parser.add_argument("--mech-dir", type=Path, default=DEFAULT_MECH_DIR)
    parser.add_argument("--behav-path", type=Path, default=DEFAULT_BEHAV_PATH)
    parser.add_argument("--mc-path", type=Path, default=DEFAULT_MC_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", type=str, default="expert")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load N0 activations
    payload = torch.load(args.n0_act_dir / "activations.pt", map_location="cpu")
    n0_meta = load_jsonl(args.n0_act_dir / "metadata.jsonl")
    acts = payload["activations"]["endorsement_last"]

    # Load mechanism labels
    mech_meta = load_jsonl(args.mech_dir / "metadata.jsonl")
    uid_to_label = {}
    for row in mech_meta:
        if row["variant_label"] == "i1a":
            uid_to_label[row["uid"]] = row["item_label"]

    uid_to_m_n0 = {}
    sel_path = args.mech_dir / "selected_items.jsonl"
    if sel_path.exists():
        for line in open(sel_path):
            item = json.loads(line.strip())
            uid_to_m_n0[item["uid"]] = item.get("m_n0", 0.0)

    # Load behavioral results
    behav = {r["uid"]: r for r in load_jsonl(args.behav_path)}

    # Load MC dataset
    mc_items = load_mc_dataset(args.mc_path)
    uid_to_mc = {item.uid: item for item in mc_items}

    # Build matched data
    indices, labels, confidences, actual_updates, uids = [], [], [], [], []
    correct_answers, wrong_answers, questions = [], [], []
    for i, row in enumerate(n0_meta):
        uid = row["uid"]
        if uid in uid_to_label and uid in behav and uid in uid_to_mc:
            indices.append(i)
            labels.append(1 if uid_to_label[uid] == "correcting" else 0)
            confidences.append(uid_to_m_n0.get(uid, 0.0))
            br = behav[uid]
            m_n0 = signed_margin(br, f"N0_{args.tag}")
            m_c1 = signed_margin(br, f"C1_{args.tag}")
            actual_updates.append(m_c1 - m_n0)
            uids.append(uid)

            mc = uid_to_mc[uid]
            correct_answers.append(mc.correct_answer)
            wrong_answers.append(mc.wrong_answer)
            questions.append(mc.question)

    X = acts[torch.tensor(indices), BEST_LAYER, :].float().numpy()
    y = np.array(labels)
    conf = np.array(confidences)
    updates = np.array(actual_updates)

    print(f"Items: {len(y)} (correcting={sum(y)}, resisting={len(y)-sum(y)})")

    # Step 1: Compute N0 residual scores (same as n0_residual_correlation.py)
    conf_model = Ridge(alpha=100.0).fit(X, conf)
    conf_dir = conf_model.coef_ / np.linalg.norm(conf_model.coef_)
    X_residual = X - np.outer(X @ conf_dir, conf_dir)

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

    print(f"  Residual scores: mean={residual_scores.mean():.3f}, std={residual_scores.std():.3f}")

    # ================================================================
    # HYPOTHESIS 1: Confidence extremity
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  HYPOTHESIS 1: Confidence Extremity")
    print(f"  (Do items with extreme confidence resist more?)")
    print(f"{'='*70}")

    abs_conf = np.abs(conf)
    r_absconf, p_absconf = pearsonr(abs_conf, y)
    print(f"  |confidence| vs correcting: r={r_absconf:.4f} (p={p_absconf:.2e})")

    corr_conf = abs_conf[y == 1]
    resist_conf = abs_conf[y == 0]
    t_stat, t_p = ttest_ind(corr_conf, resist_conf)
    print(f"  Mean |conf| correcting={corr_conf.mean():.3f}, resisting={resist_conf.mean():.3f}")
    print(f"  t-test: t={t_stat:.3f}, p={t_p:.2e}")

    # ================================================================
    # HYPOTHESIS 2: Semantic distance between answers
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  HYPOTHESIS 2: Semantic Distance Between Answers")
    print(f"  (Are semantically closer answer pairs more correctable?)")
    print(f"{'='*70}")

    print(f"  Loading model for embeddings...")
    model, tokenizer = load_model_and_tokenizer(
        args.model, device=args.device, dtype="bfloat16",
        prefer_flash_attention=False,
    )
    model.eval()
    device = next(model.parameters()).device

    print(f"  Computing embeddings for correct/wrong answers...")
    corr_embs = get_embeddings(model, tokenizer, correct_answers, device)
    wrong_embs = get_embeddings(model, tokenizer, wrong_answers, device)

    # Cosine distance between correct and wrong
    from numpy.linalg import norm
    cos_sims = np.array([
        np.dot(c, w) / (norm(c) * norm(w) + 1e-8)
        for c, w in zip(corr_embs, wrong_embs)
    ])
    # L2 distance
    l2_dists = np.array([norm(c - w) for c, w in zip(corr_embs, wrong_embs)])

    r_cos, p_cos = pearsonr(cos_sims, y)
    r_l2, p_l2 = pearsonr(l2_dists, y)
    print(f"\n  Cosine sim(correct, wrong) vs correcting: r={r_cos:.4f} (p={p_cos:.2e})")
    print(f"  L2 dist(correct, wrong) vs correcting: r={r_l2:.4f} (p={p_l2:.2e})")

    corr_cos = cos_sims[y == 1]
    resist_cos = cos_sims[y == 0]
    t_cos, p_t_cos = ttest_ind(corr_cos, resist_cos)
    print(f"  Mean cos_sim correcting={corr_cos.mean():.4f}, resisting={resist_cos.mean():.4f}")
    print(f"  t-test: t={t_cos:.3f}, p={p_t_cos:.2e}")

    # Partial correlation with residual scores
    r_cos_resid, p_cos_resid = pearsonr(cos_sims, residual_scores)
    print(f"\n  Cosine sim vs N0 residual score: r={r_cos_resid:.4f} (p={p_cos_resid:.2e})")

    # Also: distance to N0 probe residual, controlling for confidence
    cos_resid = cos_sims - LinearRegression().fit(conf.reshape(-1, 1), cos_sims).predict(conf.reshape(-1, 1))
    r_partial_cos, p_partial_cos = pearsonr(cos_resid, residual_scores)
    print(f"  Partial (controlling conf): r={r_partial_cos:.4f} (p={p_partial_cos:.2e})")

    # ================================================================
    # HYPOTHESIS 3: Question properties
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  HYPOTHESIS 3: Question Structure Features")
    print(f"{'='*70}")

    # 3a: Answer length asymmetry
    correct_lens = np.array([len(a.split()) for a in correct_answers])
    wrong_lens = np.array([len(a.split()) for a in wrong_answers])
    len_ratios = correct_lens / (wrong_lens + 1e-8)

    r_len, p_len = pearsonr(len_ratios, y)
    print(f"  Answer length ratio (correct/wrong) vs correcting: r={r_len:.4f} (p={p_len:.2e})")

    # 3b: Question length
    q_lens = np.array([len(q.split()) for q in questions])
    r_qlen, p_qlen = pearsonr(q_lens, y)
    print(f"  Question length vs correcting: r={r_qlen:.4f} (p={p_qlen:.2e})")

    # 3c: Dataset source
    print(f"\n  Dataset source breakdown:")
    for uid_idx, uid in enumerate(uids):
        mc = uid_to_mc[uid]
    datasets = {}
    for uid_idx, uid in enumerate(uids):
        ds = uid_to_mc[uid].metadata.get("dataset", "unknown") if hasattr(uid_to_mc[uid], 'metadata') else "unknown"
        if ds not in datasets:
            datasets[ds] = {"correcting": 0, "resisting": 0}
        if y[uid_idx] == 1:
            datasets[ds]["correcting"] += 1
        else:
            datasets[ds]["resisting"] += 1

    for ds_name, counts in sorted(datasets.items()):
        total = counts["correcting"] + counts["resisting"]
        corr_rate = counts["correcting"] / (total + 1e-8)
        print(f"    {ds_name}: {total} items, corr_rate={corr_rate:.2%}")

    # 3d: N0 margin magnitude (not sign) -- items with larger wrong margin
    n0_margins = []
    for uid in uids:
        br = behav[uid]
        m = signed_margin(br, f"N0_{args.tag}")
        n0_margins.append(m)
    n0_margins = np.array(n0_margins)

    r_m, p_m = pearsonr(n0_margins, y)
    print(f"\n  N0 signed margin vs correcting: r={r_m:.4f} (p={p_m:.2e})")
    print(f"  (Positive = already correct at N0, negative = wrong at N0)")

    # Items that are MORE wrong at N0 -- do they resist more?
    wrong_mask = n0_margins < 0
    if wrong_mask.sum() > 10:
        r_wrong, p_wrong = pearsonr(n0_margins[wrong_mask], y[wrong_mask])
        print(f"  Among wrong-at-N0 items (N={wrong_mask.sum()}):")
        print(f"    N0 margin vs correcting: r={r_wrong:.4f} (p={p_wrong:.2e})")
        print(f"    Mean margin correcting: {n0_margins[(wrong_mask) & (y==1)].mean():+.3f}")
        print(f"    Mean margin resisting: {n0_margins[(wrong_mask) & (y==0)].mean():+.3f}")

    # ================================================================
    # COMBINED MODEL
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  COMBINED: Which features predict N0 residual?")
    print(f"{'='*70}")

    features = np.column_stack([
        abs_conf,
        cos_sims,
        l2_dists,
        len_ratios,
        q_lens.astype(float),
        n0_margins,
    ])
    feature_names = ["abs_conf", "cos_sim", "l2_dist", "len_ratio", "q_length", "n0_margin"]

    from sklearn.linear_model import LinearRegression as LR
    # Predict residual scores from observable features
    lr = LR().fit(features, residual_scores)
    from sklearn.metrics import r2_score
    r2 = r2_score(residual_scores, lr.predict(features))
    print(f"  Linear model R^2 = {r2:.4f} (predicting N0 residual from features)")
    print(f"  Feature coefficients:")
    for name, coef in zip(feature_names, lr.coef_):
        print(f"    {name:>12s}: {coef:+.4f}")

    # Also: what fraction of N0 signal is explained by these features?
    print(f"\n  Interpretation:")
    if r2 > 0.3:
        print(f"  --> Observable features explain {r2:.0%} of N0 residual")
        print(f"  --> N0 signal is partially interpretable from surface features")
    elif r2 > 0.1:
        print(f"  --> Observable features explain only {r2:.0%} of N0 residual")
        print(f"  --> N0 signal captures something beyond these surface features")
    else:
        print(f"  --> Observable features explain almost none ({r2:.0%}) of N0 residual")
        print(f"  --> N0 signal is a genuinely novel internal representation")

    # Save
    save_data = {
        "n_items": len(y),
        "hypothesis_1_confidence": {
            "absconf_vs_correcting_r": float(r_absconf),
            "absconf_vs_correcting_p": float(p_absconf),
            "mean_absconf_correcting": float(corr_conf.mean()),
            "mean_absconf_resisting": float(resist_conf.mean()),
        },
        "hypothesis_2_semantic_distance": {
            "cos_sim_vs_correcting_r": float(r_cos),
            "cos_sim_vs_correcting_p": float(p_cos),
            "l2_dist_vs_correcting_r": float(r_l2),
            "l2_dist_vs_correcting_p": float(p_l2),
            "cos_sim_vs_residual_r": float(r_cos_resid),
            "partial_cos_vs_residual_r": float(r_partial_cos),
        },
        "hypothesis_3_question_features": {
            "len_ratio_r": float(r_len),
            "q_length_r": float(r_qlen),
            "n0_margin_r": float(r_m),
        },
        "combined_model_r2": float(r2),
        "feature_coefficients": {n: float(c) for n, c in zip(feature_names, lr.coef_)},
    }

    out_path = args.output_dir / "n0_exploration_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
