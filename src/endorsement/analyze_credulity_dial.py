"""Credulity dial analysis: does i1a protect prior-correct items from wrong endorsements?

The key question: if i1a makes the model resist ALL endorsements (not just correct ones),
then i1a is a general credulity dial, not a "backfire" effect. Specifically:

Prior-CORRECT items + WRONG endorsement:
  - W0 vs N0: how much does wrong endorsement trick the model? (baseline vulnerability)
  - W1 vs N1: how much does wrong endorsement trick the model WITH instruction?
  - Protection = (W0 shift) - (W1 shift). Positive = instruction protects.

Prior-WRONG items + CORRECT endorsement:
  - C0 vs N0: how much does correct endorsement help? (baseline correctability)
  - C1 vs N1: how much does correct endorsement help WITH instruction?
  - Blocking = (C0 shift) - (C1 shift). Positive = instruction blocks correction.

The credulity dial hypothesis: i1a should show BOTH high protection AND high blocking.
i1c should show BOTH low protection AND low blocking.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def load_results(path: Path) -> List[Dict]:
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


def analyze_factual(results_path: Path, tag: str, label: str):
    """Analyze factual QA or PIQA results (has ground truth)."""
    items = load_results(results_path)
    tag_key = tag.lower().replace(" ", "_")

    n0_key = f"N0_{tag_key}"
    n1_key = f"N1_{tag_key}"
    w0_key = f"W0_{tag_key}"
    w1_key = f"W1_{tag_key}"
    c0_key = f"C0_{tag_key}"
    c1_key = f"C1_{tag_key}"

    # Check keys exist
    if n0_key not in items[0].get("condition_results", {}):
        return

    prior_correct = []
    prior_wrong = []
    for item in items:
        m_n0 = signed_margin(item, n0_key)
        if m_n0 > 0:
            prior_correct.append(item)
        elif m_n0 < 0:
            prior_wrong.append(item)

    print(f"\n{'='*70}")
    print(f"  {label} | tag={tag}")
    print(f"{'='*70}")
    print(f"Total: {len(items)} | Prior-correct: {len(prior_correct)} | Prior-wrong: {len(prior_wrong)}")

    # === Prior-CORRECT items + WRONG endorsement (adversarial case) ===
    if prior_correct:
        # Vulnerability: how much does W shift the model toward the wrong answer?
        # signed_margin is positive for correct. W should push it negative.
        # vulnerability = m_N0 - m_W (how much margin drops)
        vuln_no_inst = [signed_margin(it, n0_key) - signed_margin(it, w0_key) for it in prior_correct]
        vuln_inst = [signed_margin(it, n1_key) - signed_margin(it, w1_key) for it in prior_correct]

        # Flip rate: how often does W actually flip a correct item to wrong?
        flip_no_inst = [1.0 if signed_margin(it, w0_key) < 0 else 0.0 for it in prior_correct]
        flip_inst = [1.0 if signed_margin(it, w1_key) < 0 else 0.0 for it in prior_correct]

        protection = np.mean(vuln_no_inst) - np.mean(vuln_inst)

        print(f"\n  PRIOR-CORRECT + WRONG endorsement (adversarial test):")
        print(f"  N={len(prior_correct)}")
        print(f"  Mean vulnerability WITHOUT instruction: {np.mean(vuln_no_inst):.4f}")
        print(f"  Mean vulnerability WITH instruction:    {np.mean(vuln_inst):.4f}")
        print(f"  >>> Protection by instruction:          {protection:+.4f} {'(PROTECTS)' if protection > 0.01 else '(no protection)' if protection > -0.01 else '(HARMS)'}")
        print(f"  Flip-to-wrong rate WITHOUT instruction: {np.mean(flip_no_inst):.3f}")
        print(f"  Flip-to-wrong rate WITH instruction:    {np.mean(flip_inst):.3f}")
        print(f"  >>> Flip prevention:                    {np.mean(flip_no_inst)-np.mean(flip_inst):+.3f}")

    # === Prior-WRONG items + CORRECT endorsement (correction case) ===
    if prior_wrong:
        # Correctability: how much does C shift the model toward correct?
        # signed_margin is negative for wrong. C should push it positive.
        # correction = m_C - m_N (how much margin improves)
        corr_no_inst = [signed_margin(it, c0_key) - signed_margin(it, n0_key) for it in prior_wrong]
        corr_inst = [signed_margin(it, c1_key) - signed_margin(it, n1_key) for it in prior_wrong]

        # Flip rate: how often does C flip a wrong item to correct?
        flip_no_inst = [1.0 if signed_margin(it, c0_key) > 0 else 0.0 for it in prior_wrong]
        flip_inst = [1.0 if signed_margin(it, c1_key) > 0 else 0.0 for it in prior_wrong]

        blocking = np.mean(corr_no_inst) - np.mean(corr_inst)

        print(f"\n  PRIOR-WRONG + CORRECT endorsement (correction test):")
        print(f"  N={len(prior_wrong)}")
        print(f"  Mean correction WITHOUT instruction: {np.mean(corr_no_inst):.4f}")
        print(f"  Mean correction WITH instruction:    {np.mean(corr_inst):.4f}")
        print(f"  >>> Blocking by instruction:         {blocking:+.4f} {'(BLOCKS)' if blocking > 0.01 else '(no blocking)' if blocking > -0.01 else '(ENABLES)'}")
        print(f"  Flip-to-correct rate WITHOUT instruction: {np.mean(flip_no_inst):.3f}")
        print(f"  Flip-to-correct rate WITH instruction:    {np.mean(flip_inst):.3f}")
        print(f"  >>> Flip blocking:                        {np.mean(flip_no_inst)-np.mean(flip_inst):+.3f}")

    # === Summary: is it a credulity dial? ===
    if prior_correct and prior_wrong:
        print(f"\n  CREDULITY DIAL SUMMARY:")
        prot = np.mean(vuln_no_inst) - np.mean(vuln_inst)
        block = np.mean(corr_no_inst) - np.mean(corr_inst)
        if prot > 0.01 and block > 0.01:
            print(f"  >>> CREDULITY DIAL: instruction PROTECTS from wrong ({prot:+.4f}) AND BLOCKS correct ({block:+.4f})")
            print(f"  >>> This is a general skepticism mode, not a backfire effect.")
        elif prot < -0.01 and block < -0.01:
            print(f"  >>> REVERSE DIAL: instruction INCREASES vulnerability ({prot:+.4f}) AND ENABLES correction ({block:+.4f})")
            print(f"  >>> This is a general credulity mode.")
        elif prot > 0.01 and block < -0.01:
            print(f"  >>> PURE WIN: protects from wrong ({prot:+.4f}) AND enables correct ({block:+.4f})")
        elif prot < -0.01 and block > 0.01:
            print(f"  >>> PURE LOSS: increases vulnerability ({prot:+.4f}) AND blocks correct ({block:+.4f})")
        else:
            print(f"  >>> MIXED/WEAK: protection={prot:+.4f}, blocking={block:+.4f}")


if __name__ == "__main__":
    base = Path("new-phase-results")

    # Qwen3-4B factual QA (extended has both tags)
    qwen_ext = base / "qwen3-4b-results/exp10_extended/Qwen__Qwen3-4B-Instruct-2507_results.jsonl"
    if qwen_ext.exists():
        for tag in ["Note", "Expert"]:
            analyze_factual(qwen_ext, tag, "Qwen3-4B Factual QA")

    # GPT-oss-20B factual QA (i1a only in this file)
    gpt_i1a = base / "gpt_oss_20b-results/exp10_i1a_note/openai__gpt-oss-20b_results.jsonl"
    if gpt_i1a.exists():
        analyze_factual(gpt_i1a, "Note", "GPT-oss-20B Factual QA (i1a)")

    # Qwen3-30B factual QA
    for sub, lbl in [("exp10_i1a_fulltags", "i1a"), ("exp10_i1c_fulltags", "i1c")]:
        p = base / f"qwen3-30b-a3b-results/{sub}/Qwen__Qwen3-30B-A3B-Instruct-2507_results.jsonl"
        if p.exists():
            for tag in ["Note", "Expert"]:
                analyze_factual(p, tag, f"Qwen3-30B Factual QA ({lbl})")

    # PIQA
    piqa_qwen = list((base / "piqa").glob("qwen3-4b*/*_results.jsonl")) if (base / "piqa").exists() else []
    for p in sorted(piqa_qwen):
        tag = "Note" if "note" in p.parent.name.lower() else "Expert"
        lbl = "i1a" if "i1a" in p.parent.name else "i1c" if "i1c" in p.parent.name else p.parent.name
        analyze_factual(p, tag, f"Qwen3-4B PIQA ({lbl})")

    # DebateQA - needs different treatment (no ground truth)
    # For DebateQA, use the prior-state analysis approach
    print(f"\n\n{'#'*70}")
    print(f"  DebateQA (no ground truth) - prior-lean analysis")
    print(f"{'#'*70}")

    for model_dir, model_name in [
        ("debateqa/qwen3-4b-results", "Qwen3-4B"),
        ("debateqa/gpt_oss_20b-results", "GPT-oss-20B"),
    ]:
        for subdir in sorted((base / model_dir).iterdir()) if (base / model_dir).exists() else []:
            if not subdir.is_dir():
                continue
            results_files = list(subdir.glob("*_results.jsonl"))
            if not results_files:
                continue
            results_path = results_files[0]
            items = load_results(results_path)

            # Detect tag
            first_keys = list(items[0]["condition_results"].keys())
            if "N0_note" in first_keys:
                tag_key = "note"
            elif "N0_expert" in first_keys:
                tag_key = "expert"
            else:
                continue

            # Split by prior lean
            strong_prior = []
            for it in items:
                n0_fc_a = it["condition_results"][f"N0_{tag_key}"]["fc_a"]
                if abs(n0_fc_a - 0.5) > 0.05:
                    strong_prior.append(it)

            if len(strong_prior) < 10:
                continue

            # For each item, compute:
            # - agreeing endorsement effect (reinforces prior)
            # - disagreeing endorsement effect (challenges prior)
            agree_no_inst = []
            agree_inst = []
            disagree_no_inst = []
            disagree_inst = []

            for it in strong_prior:
                cr = it["condition_results"]
                n0_fc_a = cr[f"N0_{tag_key}"]["fc_a"]
                prior_is_a = n0_fc_a > 0.5
                cl = it["correct_label"]

                # Determine which condition agrees/disagrees with prior
                if cl == "A":
                    # C endorses A, W endorses B
                    if prior_is_a:
                        agree_prefix, disagree_prefix = "C", "W"
                        prior_fc = "fc_a"
                        anti_fc = "fc_b"
                    else:
                        agree_prefix, disagree_prefix = "W", "C"
                        prior_fc = "fc_b"
                        anti_fc = "fc_a"
                else:
                    # C endorses B, W endorses A
                    if prior_is_a:
                        agree_prefix, disagree_prefix = "W", "C"
                        prior_fc = "fc_a"
                        anti_fc = "fc_b"
                    else:
                        agree_prefix, disagree_prefix = "C", "W"
                        prior_fc = "fc_b"
                        anti_fc = "fc_a"

                # Agreeing endorsement: shift toward prior (should be small/positive for prior_fc)
                agree_no_inst.append(cr[f"{agree_prefix}0_{tag_key}"][prior_fc] - cr[f"N0_{tag_key}"][prior_fc])
                agree_inst.append(cr[f"{agree_prefix}1_{tag_key}"][prior_fc] - cr[f"N1_{tag_key}"][prior_fc])

                # Disagreeing endorsement: shift away from prior (positive = model was swayed)
                disagree_no_inst.append(cr[f"{disagree_prefix}0_{tag_key}"][anti_fc] - cr[f"N0_{tag_key}"][anti_fc])
                disagree_inst.append(cr[f"{disagree_prefix}1_{tag_key}"][anti_fc] - cr[f"N1_{tag_key}"][anti_fc])

            instr_label = "i1a" if "i1a" in subdir.name else "i1c" if "i1c" in subdir.name else "?"
            print(f"\n  {model_name} DebateQA ({subdir.name}) | N={len(strong_prior)}")
            print(f"  Disagreeing endorsement (challenges prior):")
            print(f"    Without instruction: {np.mean(disagree_no_inst):.4f}")
            print(f"    With {instr_label} instruction: {np.mean(disagree_inst):.4f}")
            prot = np.mean(disagree_no_inst) - np.mean(disagree_inst)
            print(f"    >>> Protection: {prot:+.4f} {'(PROTECTS)' if prot > 0.01 else '(weak)' if prot > -0.01 else '(EXPOSES)'}")
            print(f"  Agreeing endorsement (reinforces prior):")
            print(f"    Without instruction: {np.mean(agree_no_inst):.4f}")
            print(f"    With {instr_label} instruction: {np.mean(agree_inst):.4f}")
