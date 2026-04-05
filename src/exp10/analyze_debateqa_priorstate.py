"""Analyze DebateQA results by prior state (model's baseline lean)."""
import json
import sys
from pathlib import Path
import numpy as np

def load_results(path):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items

def detect_tag(item):
    keys = list(item["condition_results"].keys())
    if "N0_note" in keys:
        return "note"
    elif "N0_expert" in keys:
        return "expert"
    else:
        raise ValueError(f"Unknown tag in keys: {keys}")

def analyze(results_path, label=""):
    items = load_results(results_path)
    tag = detect_tag(items[0])
    
    strong_thresh = 0.55
    
    prior_a = []
    prior_b = []
    neutral = []
    
    for item in items:
        cr = item["condition_results"]
        n0_fc_a = cr[f"N0_{tag}"]["fc_a"]
        
        if n0_fc_a > strong_thresh:
            prior_a.append(item)
        elif n0_fc_a < (1 - strong_thresh):
            prior_b.append(item)
        else:
            neutral.append(item)
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"Total items: {len(items)}")
    print(f"Prior A (N0 fc_a > {strong_thresh}): {len(prior_a)}")
    print(f"Prior B (N0 fc_a < {1-strong_thresh}): {len(prior_b)}")
    print(f"Neutral: {len(neutral)}")
    
    # For items with a prior lean, look at what happens when
    # endorsement DISAGREES with the prior
    
    # Prior-A items: the "disagreeing" endorsement is W (endorses wrong=B for correct=A items)
    # or C depending on label assignment. But since labels are arbitrary in DebateQA,
    # we need to think about this differently.
    #
    # Actually: correct_label is randomly assigned. For a "prior_a" item:
    #   - If correct_label=A: W endorses B (disagrees with prior), C endorses A (agrees)
    #   - If correct_label=B: W endorses A (agrees with prior), C endorses B (disagrees)
    
    def get_disagree_effect(item):
        """Get the endorsement effect when endorsement disagrees with model's prior."""
        cr = item["condition_results"]
        n0_fc_a = cr[f"N0_{tag}"]["fc_a"]
        prior_is_a = n0_fc_a > 0.5
        correct_label = item["correct_label"]
        
        if correct_label == "A":
            if prior_is_a:
                disagree_shift_no_inst = cr[f"W0_{tag}"]["fc_b"] - cr[f"N0_{tag}"]["fc_b"]
                disagree_shift_inst = cr[f"W1_{tag}"]["fc_b"] - cr[f"N1_{tag}"]["fc_b"]
                agree_shift_no_inst = cr[f"C0_{tag}"]["fc_a"] - cr[f"N0_{tag}"]["fc_a"]
                agree_shift_inst = cr[f"C1_{tag}"]["fc_a"] - cr[f"N1_{tag}"]["fc_a"]
            else:
                disagree_shift_no_inst = cr[f"C0_{tag}"]["fc_a"] - cr[f"N0_{tag}"]["fc_a"]
                disagree_shift_inst = cr[f"C1_{tag}"]["fc_a"] - cr[f"N1_{tag}"]["fc_a"]
                agree_shift_no_inst = cr[f"W0_{tag}"]["fc_b"] - cr[f"N0_{tag}"]["fc_b"]
                agree_shift_inst = cr[f"W1_{tag}"]["fc_b"] - cr[f"N1_{tag}"]["fc_b"]
        else:
            if prior_is_a:
                disagree_shift_no_inst = cr[f"C0_{tag}"]["fc_b"] - cr[f"N0_{tag}"]["fc_b"]
                disagree_shift_inst = cr[f"C1_{tag}"]["fc_b"] - cr[f"N1_{tag}"]["fc_b"]
                agree_shift_no_inst = cr[f"W0_{tag}"]["fc_a"] - cr[f"N0_{tag}"]["fc_a"]
                agree_shift_inst = cr[f"W1_{tag}"]["fc_a"] - cr[f"N1_{tag}"]["fc_a"]
            else:
                disagree_shift_no_inst = cr[f"W0_{tag}"]["fc_a"] - cr[f"N0_{tag}"]["fc_a"]
                disagree_shift_inst = cr[f"W1_{tag}"]["fc_a"] - cr[f"N1_{tag}"]["fc_a"]
                agree_shift_no_inst = cr[f"C0_{tag}"]["fc_b"] - cr[f"N0_{tag}"]["fc_b"]
                agree_shift_inst = cr[f"C1_{tag}"]["fc_b"] - cr[f"N1_{tag}"]["fc_b"]
        
        prior_strength = abs(n0_fc_a - 0.5)
        return {
            "disagree_no_inst": disagree_shift_no_inst,
            "disagree_inst": disagree_shift_inst,
            "agree_no_inst": agree_shift_no_inst,
            "agree_inst": agree_shift_inst,
            "prior_strength": prior_strength,
        }
    
    # Analyze items with strong priors
    strong_prior = prior_a + prior_b
    if not strong_prior:
        print("No items with strong prior!")
        return
    
    effects = [get_disagree_effect(it) for it in strong_prior]
    
    disagree_no = np.array([e["disagree_no_inst"] for e in effects])
    disagree_inst = np.array([e["disagree_inst"] for e in effects])
    agree_no = np.array([e["agree_no_inst"] for e in effects])
    agree_inst = np.array([e["agree_inst"] for e in effects])
    prior_str = np.array([e["prior_strength"] for e in effects])
    
    print(f"\nItems with strong prior (|N0 - 0.5| > {strong_thresh - 0.5}): {len(strong_prior)}")
    print(f"Mean prior strength: {prior_str.mean():.4f}")
    
    print(f"\n--- Endorsement DISAGREES with model's prior ---")
    print(f"  Without instruction: mean shift = {disagree_no.mean():.4f} (std={disagree_no.std():.4f})")
    print(f"  With instruction:    mean shift = {disagree_inst.mean():.4f} (std={disagree_inst.std():.4f})")
    print(f"  Instruction dampening: {disagree_no.mean() - disagree_inst.mean():.4f}")
    
    print(f"\n--- Endorsement AGREES with model's prior ---")
    print(f"  Without instruction: mean shift = {agree_no.mean():.4f} (std={agree_no.std():.4f})")
    print(f"  With instruction:    mean shift = {agree_inst.mean():.4f} (std={agree_inst.std():.4f})")
    
    # Now try with very strong priors
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        strong = [it for it in items 
                  if abs(it["condition_results"][f"N0_{tag}"]["fc_a"] - 0.5) > (thresh - 0.5)]
        if len(strong) < 10:
            continue
        effs = [get_disagree_effect(it) for it in strong]
        d_no = np.mean([e["disagree_no_inst"] for e in effs])
        d_inst = np.mean([e["disagree_inst"] for e in effs])
        a_no = np.mean([e["agree_no_inst"] for e in effs])
        a_inst = np.mean([e["agree_inst"] for e in effs])
        print(f"\n  Threshold {thresh}: {len(strong)} items")
        print(f"    Disagree: no_inst={d_no:.4f}, inst={d_inst:.4f}, delta={d_no-d_inst:.4f}")
        print(f"    Agree:    no_inst={a_no:.4f}, inst={a_inst:.4f}")


if __name__ == "__main__":
    base = Path("new-phase-results/debateqa/gpt_oss_20b-results")
    
    for subdir in ["exp10_i1a_note", "exp10_i1c_note", "exp10_i1a_expert", "exp10_i1c_expert"]:
        p = base / subdir / "openai__gpt-oss-20b_results.jsonl"
        if p.exists():
            analyze(p, label=f"GPT-oss {subdir}")
    
    qbase = Path("new-phase-results/debateqa/qwen3-4b-results")
    for subdir in ["exp10_i1a_note", "exp10_i1c_note"]:
        p = qbase / subdir
        candidates = list(p.glob("*_results.jsonl")) if p.exists() else []
        if candidates:
            analyze(candidates[0], label=f"QWEN {subdir}")
