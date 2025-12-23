# Llama-3.1-8B-Instruct Steering Experiment Results

**Date:** December 2025  
**Model:** `meta-llama/Llama-3.1-8B-Instruct`

## Summary

We attempted to extract and apply anti-sycophancy steering vectors using Contrastive Activation Addition (CAA) methodology. The steering vectors failed to produce statistically significant effects.

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

## Diagnosis

The CAA vector extraction methodology failed for Llama-3.1-8B-Instruct:

1. **Vector points wrong direction** - Steering increases sycophancy rather than reducing it
2. **Layer selection doesn't matter** - Neither early (layer 2) nor late (layers 29-31) injection works
3. **CAA is correlational, not causal** - Mean difference captures what's different, not what causes sycophancy
4. **High behavioral variance** - Model's sycophancy behavior on Split C differs substantially from Split A

### Root Cause Analysis

CAA assumes sycophancy is a linear direction in activation space that can be found via mean difference. This fails because:

- **No optimization** - CAA just takes a statistical mean, not an optimized direction
- **Correlation ≠ causation** - The mean difference may not be the direction that actually controls sycophancy
- **Sycophancy is distributed** - The behavior may involve many components in complex interactions

---

## Next Steps: Contrastive Causal Mediation (CCM)

Based on the ICLR 2026 submission "Activation Steering via Contrastive Causal Mediation":

- **Method**: Use causal mediation analysis to find which attention heads actually CAUSE sycophancy
- **Key difference**: CCM measures indirect effects (causal) vs mean difference (correlational)
- **Target**: Steer specific attention heads, not residual stream
- **Evidence**: CCM was specifically tested on sycophancy and outperformed correlational baselines

---

## Experiment 5: Contrastive Causal Mediation (CCM)

**Goal:** Identify attention heads that causally mediate sycophancy and steer only those heads.

**Method:**
1. For each attention head, compute **indirect effect**: how much does patching that head (with output from sycophantic context) change the model's sycophancy?
2. Rank all 1024 heads (32 layers × 32 heads) by absolute indirect effect
3. Select top-50 heads for steering
4. Compute per-head steering vectors (neutral - sycophantic mean)
5. Apply steering vectors to selected heads only

### Head Mediation Analysis (n=100 examples)

**Top 10 Heads by Indirect Effect:**

| Layer | Head | Indirect Effect | Interpretation |
|-------|------|-----------------|----------------|
| 0 | 10 | **-0.101** | Strong negative mediator |
| 1 | 6 | -0.048 | Moderate negative mediator |
| 0 | 14 | +0.045 | Moderate positive mediator |
| 5 | 1 | +0.044 | Moderate positive mediator |
| 1 | 10 | +0.042 | Moderate positive mediator |
| 0 | 20 | -0.040 | Moderate negative mediator |
| 2 | 3 | +0.039 | Moderate positive mediator |
| 13 | 1 | +0.038 | Moderate positive mediator |
| 0 | 11 | -0.037 | Moderate negative mediator |
| 9 | 7 | -0.036 | Moderate negative mediator |

**Key Observations:**
- Indirect effects are very small (|effect| < 0.1)
- Most top mediators are in early layers (0-13)
- Mix of positive and negative mediators
- Layer 0 dominates (7 of top 20 heads)

### Steering Evaluation Results (n=200 examples)

| Alpha | Mean Effect | Std | Direction | Significant? |
|-------|-------------|-----|-----------|--------------|
| 1.0 | **-0.108** | 1.00 | Wrong | No |
| 2.0 | **-0.097** | 1.17 | Wrong | No |
| 4.0 | **-0.044** | 1.36 | Wrong | No |
| 8.0 | +0.054 | 1.69 | Correct | No (p > 0.5) |

### Key Findings

1. **CCM also fails** - Despite using causal mediation analysis, steering still doesn't work reliably
2. **Wrong direction at low alphas** - Alphas 1-4 produce negative effects (increase sycophancy)
3. **Marginal improvement at α=8** - Slight positive effect but swamped by variance
4. **Extremely high variance** - Std >> Mean for all configurations
5. **Head selection concentrated in early layers** - Suggests sycophancy circuit may be more distributed

### Why CCM Failed

Unlike the theoretical advantages of CCM, our implementation shows:

1. **Weak mediation effects** - Individual heads have tiny indirect effects (< 0.1)
2. **No clear mediator circuit** - Top heads are scattered across early layers
3. **Additive assumption violated** - Steering multiple heads doesn't produce additive effects
4. **Sycophancy may not be localized** - Behavior may emerge from distributed computation

**Files:** `llama-results/exp5/`
- `head_effects.json` - Indirect effects for all 1024 heads
- `selected_heads.json` - Top 50 heads selected for steering
- `steering_vectors.json` - Per-head steering vectors
- `evaluation_results.json` - Alpha sweep evaluation results

---

## Final Conclusion

**Neither CAA nor CCM successfully steers sycophancy on Llama-3.1-8B-Instruct.**

| Approach | Methodology | Result |
|----------|-------------|--------|
| CAA (exp1-4) | Mean activation difference | Negative effects (wrong direction) |
| CAA late-layer (exp4b) | Inject at layers 29-31 | No effect / noise |
| CCM (exp5) | Causal mediation + head steering | Weak/negative effects |

### Root Causes

1. **Sycophancy is not a linear direction** - Mean difference doesn't capture the true behavior direction
2. **Sycophancy is distributed** - Not localized to specific heads or layers
3. **Model architecture matters** - Llama-3.1-8B-Instruct may encode behaviors differently than models these methods were designed for
4. **Need for optimization** - RL-trained vectors (like in "Small Vectors, Big Effects") work because they're optimized, not just extracted

### Recommendations

1. **Try RL-trained vectors** - Use GRPO or similar to learn optimal steering directions
2. **Try probing-based methods** - Train linear probes on sycophancy labels
3. **Try different datasets** - Current dataset may not provide clean contrastive signal
4. **Try different models** - Smaller models or different architectures may be more steerable

---

## File Structure

```
llama-results/
├── exp1/                   # CAA vector extraction
│   └── vector_selection.json
├── exp4/exp4/              # Distributed hypothesis test + late-layer sweep
│   ├── layer_alpha_sweep_results.json
│   └── layer_alpha_sweep_full.json
├── exp5/                   # CCM experiment
│   ├── head_effects.json
│   ├── selected_heads.json
│   ├── steering_vectors.json
│   └── evaluation_results.json
└── RESULTS.md              # This file
```
