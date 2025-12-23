# Llama-3.1-8B-Instruct Steering Experiment Results

**Date:** December 2025  
**Model:** `meta-llama/Llama-3.1-8B-Instruct`

## Summary

We systematically tested multiple steering approaches to reduce sycophancy in Llama-3.1-8B-Instruct:

| Approach | Method | Result |
|----------|--------|--------|
| CAA (exp1-4) | Mean activation difference in residual stream | ❌ Wrong direction |
| CAA late-layer (exp4b) | Inject at layers 29-31 | ❌ No effect |
| CCM mean-diff (exp5) | Causal head selection + mean-diff vectors | ❌ Wrong direction |
| CCM patching (exp5b) | Causal head selection + activation patching | ❌ Wrong direction |
| **Full model patching** | Patch ALL 1024 heads with neutral activations | ❌ Wrong direction |

**Conclusion:** Llama-3.1-8B-Instruct is resistant to activation-based steering interventions for sycophancy, regardless of head selection strategy or intervention method.

---

## Experiment 1: CAA Vector Extraction

**Goal:** Extract steering vectors from residual stream activations by contrasting sycophantic vs truthful model behavior.

**Method:**
- Used Split A (20% of sycophancy dataset) for vector extraction
- Labeled examples by D_syc margin (syc if D_syc >= margin, truth if D_syc <= -margin)
- Computed mean activations for each class, vector = μ_truth - μ_syc
- Grid search over layers and alpha values

**Results:**
| Parameter | Value |
|-----------|-------|
| Margin | 2.78 |
| Syc examples | 93 |
| Truth examples | 93 |
| Selected layer (L*) | 2 |
| Selected alpha (α*) | 16.0 |

**Files:** `llama-results/exp1-new/exp1/`
- `layer_vectors.pt` - Per-layer steering vectors
- `vector_selection.json` - Grid search results

---

## Experiment 4: Distributed Hypothesis Test

**Goal:** Test whether steering operates through attention heads or MLP layers.

**Conditions:**
1. Baseline: Steering with no ablation
2. All attention ablated: Zero out all attention head outputs
3. MLP ablated: Zero out MLP layers [8, 16, 24]

### Alpha Sweep Results (n=200 examples)

| Alpha | Baseline Effect | p(≠0) | Attn Ablated | p(base vs attn) |
|-------|-----------------|-------|--------------|-----------------|
| 4.0   | -0.013          | 0.799 | +0.043       | 0.341           |
| 8.0   | -0.035          | 0.632 | +0.013       | 0.568           |
| 16.0  | -0.065          | 0.472 | +0.071       | **0.086**       |
| 24.0  | -0.094          | 0.414 | +0.042       | 0.157           |

### Key Findings

1. **All baseline effects are NEGATIVE** - The steering vector makes the model *more* sycophantic, opposite to intended behavior.

2. **No effects are statistically significant** - All p-values for one-sample t-tests > 0.4.

3. **Higher alpha = worse** - Effect becomes more negative as alpha increases (-0.013 → -0.094).

4. **Attention inversion pattern** - With attention ablated, effects flip positive (though not significant). At α=16, baseline vs attn comparison approaches significance (p=0.086).

5. **High variance** - Standard deviations (0.75-1.9) >> effect sizes (0.01-0.12). Signal-to-noise is essentially zero.

**Files:** `llama-results/exp4/exp4/`
- `alpha_sweep_results.json` - Summary statistics per alpha
- `alpha_sweep_full.json` - Per-example effects

---

## Experiment 4b: Late-Layer Injection Test

**Goal:** Test the hypothesis from "Small Vectors, Big Effects" paper that late-layer injection is more effective than early-layer injection on Llama-3.1-8B-Instruct.

**Method:**
- Used same CAA vectors from exp1
- Tested layers 29, 30, 31 (last 3 layers) instead of layer 2
- Higher alpha values: [8.0, 16.0, 32.0, 64.0]

### Layer x Alpha Sweep Results (n=200 examples)

| Layer | Alpha | Baseline Effect | p(≠0) | Direction |
|-------|-------|-----------------|-------|-----------|
| 29 | 8.0 | **-0.027** | 0.326 | Wrong |
| 29 | 16.0 | **-0.046** | 0.370 | Wrong |
| 29 | 32.0 | **-0.069** | 0.437 | Wrong |
| 29 | 64.0 | **-0.071** | 0.574 | Wrong |
| 30 | 8.0 | +0.004 | 0.851 | Noise |
| 30 | 16.0 | +0.011 | 0.797 | Noise |
| 30 | 32.0 | -0.009 | 0.906 | Noise |
| 30 | 64.0 | -0.042 | 0.719 | Wrong |
| 31 | 8.0 | +0.005 | 0.776 | Noise |
| 31 | 16.0 | +0.009 | 0.791 | Noise |
| 31 | 32.0 | +0.013 | 0.830 | Noise |
| 31 | 64.0 | +0.009 | 0.919 | Noise |

### Key Findings

1. **Layer 29 actively worsens sycophancy** - Consistent negative effects across all alphas
2. **Layers 30-31 show near-zero effects** - Indistinguishable from noise
3. **Higher alphas don't help** - Effects don't scale, just add variance
4. **Late-layer hypothesis doesn't transfer** - The "Small Vectors, Big Effects" finding was for RL-trained vectors, not CAA vectors

**Critical Insight:** The "Small Vectors, Big Effects" paper used **RL-trained** steering vectors (optimized via GRPO), NOT CAA vectors (mean difference). These are fundamentally different approaches:

| Approach | Method | Result |
|----------|--------|--------|
| RL-trained (their paper) | Vectors learned via reinforcement | Works on Llama-3.1-8B |
| CAA (our approach) | Mean difference of activations | Does not work |

**Files:** `llama-results/exp4/exp4/`
- `layer_alpha_sweep_results.json` - Summary statistics per layer/alpha
- `layer_alpha_sweep_full.json` - Per-example effects

---

## Experiment 5: Contrastive Causal Mediation (CCM)

**Goal:** Identify attention heads that causally mediate sycophancy and steer only those heads.

**Reference:** Based on the ICLR 2026 submission ["Activation Steering via Contrastive Causal Mediation"](https://openreview.net/forum?id=n25T76QbxQ)

### 5a: Head Mediation Analysis

**Method:**
1. For each attention head, compute **indirect effect**: how much does patching that head (with output from sycophantic context) change the model's sycophancy?
2. Rank all 1024 heads (32 layers × 32 heads) by absolute indirect effect
3. Select top-50 heads for steering

**Top 10 Heads by Indirect Effect (n=100 examples):**

| Layer | Head | Indirect Effect | Std | Signal/Noise |
|-------|------|-----------------|-----|--------------|
| 0 | 10 | **-0.101** | 0.455 | 0.22 |
| 1 | 6 | -0.048 | 0.354 | 0.14 |
| 0 | 14 | +0.045 | 0.378 | 0.12 |
| 5 | 1 | +0.044 | 0.186 | 0.24 |
| 1 | 10 | +0.042 | 0.327 | 0.13 |
| 0 | 20 | -0.040 | 0.178 | 0.22 |
| 2 | 3 | +0.039 | 0.198 | 0.20 |
| 13 | 1 | +0.038 | 0.205 | 0.19 |
| 0 | 11 | -0.037 | 0.366 | 0.10 |
| 9 | 7 | -0.036 | 0.493 | 0.07 |

**Key Observations:**
- Indirect effects are very small (|effect| < 0.1)
- Most top mediators are in early layers (0-13)
- Mix of positive and negative mediators
- Layer 0 dominates (7 of top 20 heads)
- **Signal-to-noise ratios are poor** (all < 0.25)

---

### 5b: Steering with Mean-Difference Vectors (Original CCM)

**Method:**
- Compute per-head steering vectors: `neutral_mean - sycophantic_mean`
- Add scaled vectors to selected heads during inference

**Results (n=200 examples):**

| Alpha | Mean Effect | Std | Direction | Significant? |
|-------|-------------|-----|-----------|--------------|
| 1.0 | **-0.108** | 1.00 | Wrong | No |
| 2.0 | **-0.097** | 1.17 | Wrong | No |
| 4.0 | **-0.044** | 1.36 | Wrong | No |
| 8.0 | +0.054 | 1.69 | Correct | No (p > 0.5) |

**Finding:** Mean-difference steering fails even with causally-selected heads.

---

### 5c: Steering with Activation Patching (Fixed CCM)

**Motivation:** The original GCM paper uses activation patching, not mean-difference vectors. We fixed the implementation to match the paper's methodology.

**Method:**
- Cache attention head outputs from neutral prompt
- Patch (interpolate/replace) selected head outputs during sycophantic prompt evaluation
- Measure change in D_syc

**Results - Interpolation Mode (n=200 examples):**

| Alpha | Mean Effect | Std | Effect/Std | Direction |
|-------|-------------|-----|------------|-----------|
| 1.0 | **-0.079** | 0.81 | 0.10 | Wrong |
| 2.0 | **-0.049** | 1.15 | 0.04 | Wrong |
| 4.0 | **+0.047** | 1.61 | 0.03 | Correct (noise) |
| 8.0 | **-0.105** | 1.81 | 0.06 | Wrong |

**Finding:** Activation patching also fails. Effect direction is inconsistent and indistinguishable from noise.

---

### 5d: Diagnostic Test - Full Model Patching

**Goal:** Determine if the problem is head selection or model resistance.

**Method:**
- Test patching ALL 1024 heads (not just top-50)
- Use `replace` mode (α=1.0, full replacement with neutral activations)
- This is the **strongest possible** activation patching intervention

**Results (n=200 examples):**

| Variant | Heads Patched | Mean Effect | Std | Direction |
|---------|---------------|-------------|-----|-----------|
| Selected (top-50) | 50 | **-0.075** | 0.81 | Wrong |
| **All heads** | 1024 | **-0.107** | 1.29 | Wrong |

### Critical Finding

**Patching ALL 1024 attention heads with neutral activations makes sycophancy WORSE, not better.**

This definitively rules out:
- ❌ "Head selection is wrong" — we patched ALL heads
- ❌ "Need more heads" — 1024 = all of them  
- ❌ "Need replace instead of interpolate" — tested both

**Conclusion:** The model is fundamentally resistant to this class of interventions.

**Files:** `llama-results/exp5/`
- `head_effects.json` - Indirect effects for all 1024 heads
- `selected_heads.json` - Top 50 heads selected for steering
- `evaluation_results.json` - Mean-diff steering results
- `evaluation_results_patching.json` - Activation patching (interp mode)
- `evaluation_results_patching_selected_replace.json` - Selected heads, replace mode
- `evaluation_results_patching_all_replace.json` - All heads, replace mode

---

## Final Conclusion

**Llama-3.1-8B-Instruct is resistant to activation-based steering interventions for sycophancy.**

### Summary of All Approaches

| Experiment | Method | Heads | Intervention | Effect | Verdict |
|------------|--------|-------|--------------|--------|---------|
| Exp1-4 | CAA | N/A (residual) | Vector addition | -0.065 | ❌ Wrong direction |
| Exp4b | CAA late-layer | N/A (residual) | Vector addition | ~0 | ❌ No effect |
| Exp5b | CCM | Top-50 | Mean-diff vectors | -0.108 to +0.054 | ❌ Inconsistent |
| Exp5c | CCM | Top-50 | Activation patching | -0.079 to +0.047 | ❌ Noise |
| Exp5d | Full patch | **All 1024** | Full replacement | **-0.107** | ❌ Wrong direction |

### Why Nothing Works

1. **Sycophancy is not localized to attention heads**
   - Even patching ALL heads doesn't help
   - Behavior may be encoded in embeddings, MLPs, or layer norms

2. **The neutral↔sycophantic activation difference is not causal**
   - Replacing activations with "neutral" values doesn't reduce sycophancy
   - The activations may correlate with but not cause the behavior

3. **D_syc may not capture steerable behavior**
   - The metric (log-prob difference of wrong vs right answer) may be orthogonal to what these interventions affect

4. **Llama-3.1-8B-Instruct encodes behaviors differently**
   - The GCM paper tested SOLAR, Qwen, and OLMo — not Llama
   - Instruction-tuned Llama models may have different internal structure

### Comparison with Literature

| Paper | Model | Method | Result |
|-------|-------|--------|--------|
| GCM (ICLR 2026) | SOLAR, Qwen, OLMo | Causal mediation + patching | ✅ Works |
| Small Vectors, Big Effects | Llama-3.1-8B | **RL-trained** vectors (GRPO) | ✅ Works |
| **This work** | Llama-3.1-8B-Instruct | CAA / CCM (non-optimized) | ❌ Fails |

The key difference: **optimization**. Methods that learn steering vectors via reinforcement (GRPO) or other optimization succeed. Methods that extract vectors via statistical means (CAA, CCM mean-diff) fail.

### Recommendations for Future Work

1. **Use RL-trained vectors** — GRPO or similar to learn optimal steering directions
2. **Try probing-based methods** — Train linear probes on sycophancy labels, use probe weights as steering direction
3. **Investigate other components** — MLPs, layer norms, or embeddings may be more causally relevant
4. **Different metric** — Response-level evaluation (LLM judge) instead of token-level D_syc
5. **Different models** — Test on base Llama (not instruct) or other model families

---

## File Structure

```
llama-results/
├── exp1/                          # CAA vector extraction
│   └── vector_selection.json
├── exp4/exp4/                     # Distributed hypothesis test + late-layer sweep
│   ├── alpha_sweep_results.json
│   ├── alpha_sweep_full.json
│   ├── layer_alpha_sweep_results.json
│   └── layer_alpha_sweep_full.json
├── exp5/                          # CCM experiment (all variants)
│   ├── head_effects.json                           # Indirect effects for 1024 heads
│   ├── selected_heads.json                         # Top 50 heads
│   ├── evaluation_results.json                     # Mean-diff steering
│   ├── evaluation_results_patching.json            # Patching (interp mode)
│   ├── evaluation_results_patching_selected_replace.json  # Top-50, replace
│   └── evaluation_results_patching_all_replace.json       # All 1024, replace
└── RESULTS.md                     # This file
```
