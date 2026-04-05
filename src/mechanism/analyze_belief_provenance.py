"""Belief provenance analysis: what predicts resistance beyond confidence?

Three analyses:
1. A/B position confounder check
2. Cross-model resistance correlation (do same items resist across models?)
3. Within-confidence-bin analysis with cross-model consensus proxy
"""
import json
from pathlib import Path
from typing import Dict, List
import numpy as np


def load_jsonl(path: Path) -> List[Dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def signed_margin(row: Dict, cond: str) -> float:
    cr = row["condition_results"][cond]
    if row["correct_label"] == "A":
        return cr["logit_a"] - cr["logit_b"]
    return cr["logit_b"] - cr["logit_a"]


def get_prior_wrong(results: List[Dict], tag: str) -> Dict[str, Dict]:
    out = {}
    for row in results:
        m = signed_margin(row, f"N0_{tag}")
        if m < 0:
            out[row["uid"]] = row
    return out


def resistance_label(row: Dict, tag: str) -> str:
    m_n0 = signed_margin(row, f"N0_{tag}")
    m_c1 = signed_margin(row, f"C1_{tag}")
    return "correcting" if (m_c1 - m_n0) > 0 else "resisting"


def main():
    models = [
        ("Qwen3-4B",
         Path("new-phase-results/qwen3-4b-results/exp10_extended/Qwen__Qwen3-4B-Instruct-2507_results.jsonl"),
         "expert"),
        ("GPT-oss-20B",
         Path("new-phase-results/gpt_oss_20b-results/exp10_i1a_note/openai__gpt-oss-20b_results.jsonl"),
         "note"),
        ("Gemma-4-26B",
         Path("new-phase-results/gemma-4-26b-results/exp10_i1a_note/google__gemma-4-26B-A4B-it_results.jsonl"),
         "note"),
    ]

    md = {}  # name -> {results, tag, prior_wrong}
    for name, path, tag in models:
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {name}")
            continue
        results = load_jsonl(path)
        pw = get_prior_wrong(results, tag)
        md[name] = {"results": results, "tag": tag, "pw": pw}
        print(f"  Loaded {name}: {len(results)} items, {len(pw)} prior-wrong (tag={tag})")

    # ================================================================
    # 1. A/B POSITION CONFOUNDER CHECK
    # ================================================================
    print("\n" + "=" * 70)
    print("  1. A/B POSITION CONFOUNDER CHECK")
    print("=" * 70)

    for name, d in md.items():
        tag = d["tag"]
        pw = d["pw"]
        resist_a = sum(1 for r in pw.values()
                       if resistance_label(r, tag) == "resisting" and r["correct_label"] == "A")
        resist_b = sum(1 for r in pw.values()
                       if resistance_label(r, tag) == "resisting" and r["correct_label"] == "B")
        correct_a = sum(1 for r in pw.values()
                        if resistance_label(r, tag) == "correcting" and r["correct_label"] == "A")
        correct_b = sum(1 for r in pw.values()
                        if resistance_label(r, tag) == "correcting" and r["correct_label"] == "B")

        n_resist = resist_a + resist_b
        n_correct = correct_a + correct_b
        print(f"\n  {name} (N={len(pw)} prior-wrong):")
        print(f"    Resisting:  correct=A: {resist_a}, correct=B: {resist_b} "
              f"(A frac={resist_a/n_resist:.3f})" if n_resist > 0 else "    Resisting: N=0")
        print(f"    Correcting: correct=A: {correct_a}, correct=B: {correct_b} "
              f"(A frac={correct_a/n_correct:.3f})" if n_correct > 0 else "    Correcting: N=0")
        overall_a = sum(1 for r in d["results"] if r["correct_label"] == "A") / len(d["results"])
        print(f"    Dataset overall A frac: {overall_a:.3f}")
        if n_resist > 0 and n_correct > 0:
            diff = abs(resist_a / n_resist - correct_a / n_correct)
            print(f"    >>> Position bias (|diff in A-frac|): {diff:.3f} "
                  f"{'(CLEAN)' if diff < 0.05 else '(POSSIBLE CONFOUNDER)' if diff < 0.10 else '(CONFOUNDER!)'}")

    # ================================================================
    # 2. CROSS-MODEL RESISTANCE CORRELATION
    # ================================================================
    print("\n" + "=" * 70)
    print("  2. CROSS-MODEL RESISTANCE CORRELATION")
    print("=" * 70)

    model_names = list(md.keys())
    model_resistance = {}
    model_margin_n0 = {}
    for name, d in md.items():
        tag = d["tag"]
        model_resistance[name] = {uid: resistance_label(r, tag) for uid, r in d["pw"].items()}
        model_margin_n0[name] = {uid: signed_margin(r, f"N0_{tag}") for uid, r in d["pw"].items()}

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            shared = set(model_resistance[m1]) & set(model_resistance[m2])
            if len(shared) < 10:
                print(f"\n  {m1} vs {m2}: only {len(shared)} shared prior-wrong items, skipping")
                continue

            agree = sum(1 for u in shared if model_resistance[m1][u] == model_resistance[m2][u])
            both_resist = sum(1 for u in shared
                              if model_resistance[m1][u] == "resisting"
                              and model_resistance[m2][u] == "resisting")
            both_correct = sum(1 for u in shared
                               if model_resistance[m1][u] == "correcting"
                               and model_resistance[m2][u] == "correcting")
            m1_r = sum(1 for u in shared if model_resistance[m1][u] == "resisting")
            m2_r = sum(1 for u in shared if model_resistance[m2][u] == "resisting")

            p1 = m1_r / len(shared)
            p2 = m2_r / len(shared)
            exp_agree = p1 * p2 + (1 - p1) * (1 - p2)
            obs_agree = agree / len(shared)
            kappa = (obs_agree - exp_agree) / (1 - exp_agree) if exp_agree < 1 else 0

            # Phi coefficient (correlation for binary variables)
            labels1 = np.array([1 if model_resistance[m1][u] == "resisting" else 0 for u in shared])
            labels2 = np.array([1 if model_resistance[m2][u] == "resisting" else 0 for u in shared])
            phi = np.corrcoef(labels1, labels2)[0, 1]

            print(f"\n  {m1} vs {m2} ({len(shared)} shared prior-wrong items):")
            print(f"    {m1} resist rate: {p1:.3f}")
            print(f"    {m2} resist rate: {p2:.3f}")
            print(f"    Both resist:  {both_resist}  |  Both correct: {both_correct}")
            print(f"    Agreement: {obs_agree:.3f} (expected by chance: {exp_agree:.3f})")
            print(f"    >>> Cohen's kappa: {kappa:.3f}")
            print(f"    >>> Phi coefficient: {phi:.3f}")

            # Control: is agreement driven by confidence?
            # Within shared items, bin by AVERAGE confidence across both models
            avg_conf = {u: (abs(model_margin_n0[m1][u]) + abs(model_margin_n0[m2][u])) / 2
                        for u in shared}
            confs = np.array(list(avg_conf.values()))
            terciles = np.percentile(confs, [33, 67])

            print(f"\n    Confidence-controlled agreement (terciles of avg |m_N0|):")
            for bin_name, lo, hi in [("Low", 0, terciles[0]),
                                     ("Mid", terciles[0], terciles[1]),
                                     ("High", terciles[1], np.inf)]:
                in_bin = [u for u in shared if lo <= avg_conf[u] < hi]
                if len(in_bin) < 5:
                    continue
                bin_agree = sum(1 for u in in_bin
                                if model_resistance[m1][u] == model_resistance[m2][u])
                bin_m1r = sum(1 for u in in_bin if model_resistance[m1][u] == "resisting")
                bin_m2r = sum(1 for u in in_bin if model_resistance[m2][u] == "resisting")
                bp1 = bin_m1r / len(in_bin)
                bp2 = bin_m2r / len(in_bin)
                bin_exp = bp1 * bp2 + (1 - bp1) * (1 - bp2)
                bin_obs = bin_agree / len(in_bin)
                bin_kappa = (bin_obs - bin_exp) / (1 - bin_exp) if bin_exp < 1 else 0
                print(f"      {bin_name} (N={len(in_bin)}): agree={bin_obs:.3f}, "
                      f"expected={bin_exp:.3f}, kappa={bin_kappa:.3f}")

    # ================================================================
    # 3. WITHIN-CONFIDENCE-BIN: CROSS-MODEL CONSENSUS ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  3. CROSS-MODEL CONSENSUS VS RESISTANCE (within confidence bins)")
    print("=" * 70)

    for focal_name, focal_d in md.items():
        tag = focal_d["tag"]
        pw = focal_d["pw"]
        other_names = [n for n in md if n != focal_name]

        items = []
        for uid, row in pw.items():
            m = signed_margin(row, f"N0_{tag}")
            lab = resistance_label(row, tag)
            # How many other models also get this wrong?
            n_other_wrong = 0
            n_same_wrong = 0
            for on in other_names:
                if uid in md[on]["pw"]:
                    n_other_wrong += 1
                    n_same_wrong += 1  # both wrong on this item
            items.append({
                "uid": uid, "margin": m, "abs_margin": abs(m),
                "label": lab, "n_other_wrong": n_other_wrong,
                "consensus_wrong": n_other_wrong >= len(other_names),  # all others also wrong
            })

        if not items:
            continue

        print(f"\n  {focal_name} (N={len(items)} prior-wrong):")
        # Overall consensus split
        consensus = [it for it in items if it["consensus_wrong"]]
        unique = [it for it in items if it["n_other_wrong"] == 0]
        partial = [it for it in items if 0 < it["n_other_wrong"] < len(other_names)]

        for label, group in [("All-wrong consensus", consensus),
                             ("Partial consensus", partial),
                             ("Unique-wrong", unique)]:
            if not group:
                continue
            rr = sum(1 for it in group if it["label"] == "resisting") / len(group)
            avg_m = np.mean([it["abs_margin"] for it in group])
            print(f"    {label}: N={len(group)}, resist_rate={rr:.3f}, mean |m_N0|={avg_m:.3f}")

        # Now within confidence bins
        margins = np.array([it["abs_margin"] for it in items])
        terciles = np.percentile(margins, [33, 67])

        print(f"\n    Within-confidence-bin consensus analysis:")
        for bin_name, lo, hi in [("Low conf", 0, terciles[0]),
                                 ("Mid conf", terciles[0], terciles[1]),
                                 ("High conf", terciles[1], np.max(margins) + 1)]:
            in_bin = [it for it in items if lo <= it["abs_margin"] < hi]
            if len(in_bin) < 10:
                continue
            cons_bin = [it for it in in_bin if it["consensus_wrong"]]
            uniq_bin = [it for it in in_bin if it["n_other_wrong"] == 0]
            rr_cons = (sum(1 for it in cons_bin if it["label"] == "resisting") / len(cons_bin)
                       if cons_bin else float("nan"))
            rr_uniq = (sum(1 for it in uniq_bin if it["label"] == "resisting") / len(uniq_bin)
                       if uniq_bin else float("nan"))
            overall_rr = sum(1 for it in in_bin if it["label"] == "resisting") / len(in_bin)
            print(f"      {bin_name} (N={len(in_bin)}, overall resist={overall_rr:.3f}):")
            if cons_bin:
                print(f"        Consensus-wrong (N={len(cons_bin)}): resist_rate={rr_cons:.3f}")
            if uniq_bin:
                print(f"        Unique-wrong (N={len(uniq_bin)}):    resist_rate={rr_uniq:.3f}")
            if cons_bin and uniq_bin and not (np.isnan(rr_cons) or np.isnan(rr_uniq)):
                diff = rr_cons - rr_uniq
                print(f"        >>> Consensus effect on resistance: {diff:+.3f}")


if __name__ == "__main__":
    main()
