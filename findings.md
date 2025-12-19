# Findings: Do Persona Vectors Utilize Attention Circuits?

**Date:** December 14, 2025
**Models:** TinyLlama-1.1B, Llama-3.1-8B-Instruct
**Status:** Critical Pivot Required

---

## 1. Executive Summary

We set out to determine whether **persona steering vectors** (specifically for sycophancy) achieve their effect by activating the model's **natural sycophancy circuit**, or by bypassing it.

**The Core Finding:** We observed a consistent "Bypass" phenomenon (RAS ≈ 0) where steering vectors remained effective even when the identified "sycophancy heads" were ablated.

**The Critical Caveat:** Upon rigorous validation, we found that our **circuit identification methodology (Path Patching) failed to locate a causal circuit**. Ablating the "sycophancy heads" in the baseline model did *not* reduce sycophancy.

**Conclusion:** We cannot claim the vector "bypasses the circuit" because we never successfully found the circuit. Instead, our results point to a **methodological finding**: simple path patching correlates with, but does not identify the causal mechanism of, sycophancy in Llama-3.1. The steering vector likely operates through an independent, distributed pathway (likely MLP/Residual) that is robust to attention-head-level interventions.

---

## 2. Part I: TinyLlama-1.1B Experiments (The "Partial Mediation" Era)

**Context:** Initial experiments were conducted on the smaller TinyLlama model to establish a pipeline.

### 2.1 Baseline & Vector
- **Baseline:** Sycophancy rate ≈ 0.53.
- **Steering Vector:** Extracted via CAA. Layer 1, $\alpha=2.0$ proved optimal.
- **Effectiveness:** Mean effect size ≈ 0.17. The vector successfully shifted behavior.

### 2.2 Circuit Identification (Exp 2)
- **Method:** Single-position path patching (last prompt token).
- **Finding:** Sycophancy appeared localized to late layers (Layer 21), specifically heads L21H11 and L21H4.
- **Issue:** The set was extremely narrow (only 2-5 heads with positive signal).

### 2.3 Mediation (Exp 3)
- **Result:**
    - **Sycophancy Heads (RAS ≈ 0.36):** Ablating the identified heads reduced the steering vector's effectiveness by ~36%.
    - **Random Heads (RAS ≈ 0.04):** Random ablation had negligible effect.
- **Interpretation:** This suggested **partial mediation**. The steering vector seemed to rely *somewhat* on the natural late-layer attention heads.
- **Retrospective Flaw:** We used "exploratory" heads (selected by absolute magnitude per layer) which likely included general "processing" heads that the vector needed, rather than specific sycophancy heads.

---

## 3. Part II: Llama-3.1-8B Experiments (The "Bypass" Era)

**Context:** We scaled up to Llama-3.1-8B to verify findings on a capable, modern model.

### 3.1 Exp 0: Baseline Sycophancy
- **Metric:** $D_{syc} = \log P(\text{wrong}) - \log P(\text{correct})$
- **Result:** Sycophancy Rate = 0.52. Mean $D_{syc} \approx 0.40$ (std 3.20).
- **Insight:** The model is naturally sycophantic, but with high variance.

### 3.2 Exp 1: Steering Vector
- **Selection:** Layer 1, $\alpha=2.0$.
- **Effectiveness:** Mean shift of **0.046**.
- **Critical Weakness:** This effect is tiny (1.4% of standard deviation). While "statistically" present, the vector is far weaker than in TinyLlama.

### 3.3 Exp 2: Circuit Identification
- **Method:** Multi-position path patching (`prompt_last` and `first_answer`).
- **Selection:** `abs_top_k_per_layer` (Top 2 heads per layer by absolute delta).
- **Result:** Found 64 "sycophancy heads" distributed across all layers.
- **Flaw:** The `abs_top_k` criterion selected heads with **mixed signs**. Some promoted sycophancy, others suppressed it.

### 3.4 Exp 3: Mediation Analysis
- **Result:**
    - **RAS (Syc Heads):** 0.099
    - **RAS (Rand Heads):** 0.171
    - **Conclusion:** No significant difference ($p \approx 0.51$). The steering vector worked equally well (or poorly) whether sycophancy heads were ablated or not.
- **Initial Interpretation:** "Mechanism Decoupling" / "Bypass". The vector ignores the heads.

---

## 4. Part III: Validation Campaign (The "Null Circuit" Reality)

To rigorously test the "Bypass" hypothesis, we ran three validation experiments.

### 4.1 Circuit Verification (The Smoking Gun)
**Question:** Does ablating the identified heads actually reduce natural sycophancy?
- **Expectation:** $D_{syc}$ should drop significantly.
- **Result:**
    - Baseline $D_{syc}$: 0.223
    - Ablated $D_{syc}$: 0.250
    - **Change:** +0.026 (Sycophancy **increased**).
- **Implication:** The heads we identified were **not causal**. They were likely "suppressor" heads or simply noise. This invalidates the premise of the mediation test.

### 4.2 Sign-Aware Mediation
**Question:** If we fix selection to only include heads that *promote* sycophancy ($\Delta D_{syc} > 0$), do we see mediation?
- **Method:** Filtered for positive-impact heads only.
- **Result:**
    - RAS (Syc): 0.067
    - RAS (Rand): 0.005
    - **Conclusion:** Still a null result. Even with sign-correct heads, the steering vector is not mediated by them.

### 4.3 Position-Split Mediation
**Question:** Does the vector strike early (prompt processing) or late (answer generation)?
- **Prompt Last:** RAS ≈ 0.11.
- **First Answer:** RAS ≈ 0.01.
- **Conclusion:** Neither position shows strong mediation. The vector is not "using" the attention heads at either critical timestep.

---

## 5. Synthesis & Novelty Assessment

### 5.1 What We Can Claim (Novelty)
1.  **Negative Result on Path Patching:** Standard path patching (swapping neutral activations) is insufficient to identify the causal circuit for sycophancy in Llama-3.1-8B. The identified heads are correlational or redundant.
2.  **Steering Robustness (Bypass):** Behavioral steering vectors (injected at Layer 1) are remarkably robust to attention head ablation. This supports the "Small Vectors, Big Effects" hypothesis that steering often acts via **MLP/Residual pathways**, bypassing interpretable attention mechanisms.
3.  **Safety Implication:** "Circuit-breaking" safety methods (identifying and deleting 'bad' heads) may fail to stop steering attacks, because the attack vector doesn't need the natural circuit to function.

### 5.2 What We Cannot Claim
- We cannot claim to have "dissected the sycophancy circuit," because our verification failed.
- We cannot definitively prove "mechanism decoupling" without a verified circuit to decouple *from*.

---

## 6. Recommendations for "Conference-Ready" Science

To save this for ICML, we must address the root cause: **We don't have a verified circuit.**

1.  **Pivot to MLP Analysis:** Stop chasing attention heads. If the vector bypasses heads, it *must* be using MLPs or the residual stream directly.
    - **Action:** Run "Logit Lens" analysis to see *where* the vector pushes the logits towards the sycophantic answer.
    - **Action:** Run MLP ablation (layer-wise) to see if *that* stops the vector.

2.  **Try Causal Contrastive Mediation (CCM):**
    - Path patching failed. CCM is the gold-standard for finding causal mediators.
    - If CCM finds a circuit that *does* verify (ablation works), we can re-run mediation.
    - If CCM *also* fails, we have a very strong paper: "Sycophancy is not circuit-localized."

3.  **Strengthen the Steering:**
    - A 0.046 effect size is too small. We need a stronger vector (maybe from a different dataset or layer) to make any mediation claims robust.
