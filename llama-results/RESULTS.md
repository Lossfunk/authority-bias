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

## Diagnosis

The CAA vector extraction methodology failed for Llama-3.1-8B-Instruct:

1. **Vector points wrong direction** - Steering increases sycophancy rather than reducing it
2. **Layer selection may be wrong** - Layer 2 was selected but effects are minimal
3. **High behavioral variance** - Model's sycophancy behavior on Split C differs substantially from Split A (train/test mismatch)

### Possible Causes

- CAA methodology assumes linear separability of sycophantic vs truthful behavior in residual stream
- Llama-3.1-8B-Instruct may encode sycophancy differently than models CAA was designed for
- The sycophancy circuit may be more distributed/nonlinear than CAA can capture

---

## Next Steps

Consider implementing **Contrastive Causal Mediation (CCM)** from the ICLR 2026 submission:
- CCM was specifically evaluated on sycophancy
- Uses causal mediation analysis to identify steerable components
- Claims to outperform correlational baselines
- Focuses on attention head selection with better generalization

---

## File Structure

```
llama-results/
├── exp1-new/exp1/          # CAA vector extraction results
│   ├── layer_vectors.pt
│   └── vector_selection.json
├── exp4/exp4/              # Distributed hypothesis test
│   ├── alpha_sweep_results.json
│   └── alpha_sweep_full.json
└── RESULTS.md              # This file
```
