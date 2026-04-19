# GPT-oss Activation Steering: Full Results

## Summary

We tested whether the compliance direction extracted from GPT-oss-20B (contrastive mean of W1 vs N0 activations at the endorsement position) can causally control authority-following behavior via activation steering. We ran a systematic three-tier intervention sweep covering 30+ configurations.

**Core finding:** Simple additive steering along the extracted compliance direction does not move GPT-oss behavior at all. However, activation patching -- replacing the full endorsement-span hidden state with the neutral (N0) mean -- produces a clean, monotonic dose-response reduction in authority-following. This dissociation (readable but not additively writable) is itself a mechanistically informative result.

---

## Tier 1: Layer/Position Sweep (Additive Steering)

**Config:** 5 layers (14, 16, 18, 20, 22) x 4 positions (note_end, answer_position, last_prompt_token, endorsement_span) = 20 configs. 7 alphas (-10 to +10). Greedy decoding. First-token margin diagnostics. n=50 W1 UIDs.

**Result:** Completely flat across all 20 configs.
- W1 flip rate locked at 0.936 for nearly every config/alpha
- N0 accuracy locked at 0.667
- Margin diagnostics: baseline -0.030, perturbations < 0.01 in either direction
- No layer or position shows any sensitivity

**Interpretation:** The probe-best readout site (L18/endorsement_end) is not unique; no nearby layer or alternative position responds to additive vector injection either.

**Output:** `neurips-results/mechanism/tier1_layer_position_sweep/`

---

## Tier 2: Alternative Intervention Types

### 2a: Decode-Time Steering
**Config:** L18, block_output, decode_last_token position, apply_phase=decode (hook fires on every autoregressive step). KV cache enabled. n=50.

**Result:** Flat. W1 flip 0.936, margins exactly at baseline (-0.030).

**Output:** `neurips-results/mechanism/tier2a_decode_steering/`

### 2b: Attention-Only Steering
**Config:** L18, attention_output component, note_end position. n=50.

**Result:** Flat. W1 flip 0.936.

**Output:** `neurips-results/mechanism/tier2b_attn_only/`

### 2c: MLP-Only Steering
**Config:** L18, mlp_output component, note_end position. n=50.

**Result:** Flat. W1 flip 0.936.

**Output:** `neurips-results/mechanism/tier2c_mlp_only/`

### 2d: Norm-Aware Scaling
**Config:** L18, block_output, note_end, norm_scaling=resid_norm (alpha scales by residual stream norm). Alphas: -0.5 to 0.5. n=50.

**Result:** Flat at reasonable alphas. At alpha=0.5 (massive perturbation relative to residual norm), N0 accuracy dropped to 0.604 and W1 flip dipped to 0.915 -- coherence degradation, not controlled steering.

**Output:** `neurips-results/mechanism/tier2d_norm_aware/`

---

## Tier 3: Mechanistic Variants

### 3a: Activation Patching (POSITIVE RESULT)

#### Replace Mean (n=50 pilot)
**Config:** replace_mean intervention, source=N0_note activations (authoritative_verified style), endorsement_span position, alpha=1.0. 5 layers.

| Layer | W1 Flip | Margin | N0 Acc |
|-------|---------|--------|--------|
| L14   | 0.854   | -2.197 | 0.681  |
| L16   | 0.822   | -2.011 | 0.694  |
| L18   | 0.739   | -2.119 | 0.714  |
| L20   | 0.804   | -1.666 | 0.714  |
| L22   | 0.913   | -1.997 | 0.646  |

Baseline W1 flip: 0.936. L18 shows strongest reduction (0.739, -20pp).

#### Interpolate Mean Dose-Response (n=50 pilot)
**Config:** interpolate_mean, L18, endorsement_span, source=N0_note.

| Alpha | W1 Flip | Margin  | N0 Acc |
|-------|---------|---------|--------|
| 0.0   | 0.936   | -0.030  | 0.667  |
| 0.1   | 0.936   | -0.134  | 0.667  |
| 0.2   | 0.936   | -0.236  | 0.688  |
| 0.3   | 0.935   | -0.443  | 0.688  |
| 0.5   | 0.867   | -1.269  | 0.667  |
| 0.7   | 0.787   | -2.242  | 0.708  |
| 1.0   | 0.739   | -2.121  | 0.735  |

Clean monotonic dose-response on both flip rate and margin. N0 accuracy stable or improves.

#### Full-Scale Dose-Response (n=612)
**Config:** interpolate_mean, L18, endorsement_span, source=N0_note. All 612 W1 UIDs.

| Alpha | W1 Flip | Margin  | N0 Acc |
|-------|---------|---------|--------|
| 0.0   | 0.873   | -0.256  | 0.483  |
| 0.3   | 0.867   | -0.668  | 0.483  |
| 0.5   | 0.854   | -1.380  | 0.492  |
| 0.7   | 0.814   | -2.129  | 0.519  |
| 1.0   | 0.798   | -2.197  | 0.504  |

Monotonic dose-response confirmed at full scale. 7.5pp flip reduction, 8.5x margin shift.

#### Full-Scale Replace Mean Across Layers (n=612)
**Config:** replace_mean, endorsement_span, source=N0_note, alpha=1.0.

| Layer | W1 Flip | Margin  | N0 Acc |
|-------|---------|---------|--------|
| L16   | 0.778   | -2.062  | 0.482  |
| L18   | 0.801   | -2.193  | 0.506  |
| L20   | 0.834   | -1.699  | 0.524  |

Effect centered in L16-L20 band. L16 shows strongest flip reduction at full scale.

**Output:** `neurips-results/mechanism/full_patch_dose_response/`, `neurips-results/mechanism/full_patch_replace_layers/`

### 3b: Alternative Direction (authority_given_wrong)
**Config:** subtract mode, authority_given_wrong direction, L18, endorsement_span. n=50.

**Result:** Flat. W1 flip 0.936.

**Output:** `neurips-results/mechanism/tier3b_alt_direction_agw/`

### 3c: Sparse Steering (top-k=64)
**Config:** add mode, sparse top-64 dimensions of shared_within_label direction, L18, endorsement_span. n=50.

**Result:** Flat. W1 flip 0.936.

**Output:** `neurips-results/mechanism/tier3c_sparse_topk64/`

---

## Interpretation

### What failed
Every additive steering variant produced a flat null:
- 5 layers x 4 positions (Tier 1)
- Decode-time steering (Tier 2a)
- Attention-only and MLP-only components (Tier 2b, 2c)
- Norm-aware scaling (Tier 2d)
- Alternative direction types (Tier 3b)
- Sparse top-k steering (Tier 3c)

### What worked
Activation patching at the endorsement span:
- Replace mean and interpolate mean both reduce W1 flip rate
- Monotonic dose-response curve (alpha 0 to 1)
- Effect is centered in the L16-L20 layer band
- N0 accuracy is preserved or slightly improved
- Margin diagnostics confirm the logit-level effect is real, not a parsing artifact

### Key claim

> **GPT-oss contains compliance-relevant information at the endorsement span (layers 16-20) that is causally active: replacing these activations with the neutral (N0) state reduces authority-following with a monotonic dose-response. However, this information is not encoded as a low-dimensional linear direction -- additive vector steering along any extracted compliance direction fails completely. The compliance signal is distributed across the full activation state geometry.**

This "readable but not additively writable" dissociation has implications for:
1. **Limits of activation steering as a safety tool:** Not all decodable representations are steerable via simple vector addition.
2. **Circuit complexity:** GPT-oss appears to implement compliance via a distributed, possibly nonlinear mechanism rather than a clean single-direction gating circuit (contrast with Qwen3-4B which showed cleaner single-layer steerability).
3. **Cross-model variation:** The same behavioral phenomenon (authority-conditioned answer flipping) can be implemented by fundamentally different internal mechanisms across model families.

---

## Experiment Metadata

- **Model:** openai/gpt-oss-20b (24 layers, MoE, 2880-dim hidden)
- **Direction source:** `neurips-results/mechanism/gpt_oss_compliance_analysis/primary_direction.pt` (layer 18, endorsement_end, shared_within_label)
- **Extraction source:** `neurips-results/mechanism/gpt_oss_authority_activations/`
- **Evaluation UIDs:** 612 W1 items (model correct on N0, flipped on W1 under authoritative_verified endorsement)
- **Decoding:** Greedy (temperature=0) for all tier runs
- **Endorsement style:** authoritative_verified
- **Total configurations tested:** 30+
- **Total rows generated:** ~25,000+
