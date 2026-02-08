# Exp12 vs Exp12-P0 Comparison: Llama-3.1-8B-Instruct

## Experimental Differences

| Property | exp12_instruct (original) | exp12_p0 (P0 validity) |
|----------|--------------------------|------------------------|
| Tags | Expert, Note, User, Someone online | Expert, Note |
| Instruction conditions | off / on | none / t0 / t1 / t2 |
| Style profile | legacy (unmatched) | matched (imperativeness-controlled) |
| Turns | 2 | 3 |
| Fresh probe scoring | Shared across history branches | Independent per branch |
| torch.compile | enabled | disabled |
| max_length | 768 | 0 (unlimited) |

---

## 1. Initial Turn: Essentially Identical

| Tag | Condition | exp12_instruct | exp12_p0 | Delta |
|-----|-----------|---------------:|----------:|------:|
| Expert | no-instr neutral_margin | 1.768 | 1.759 | -0.009 |
| Expert | no-instr wrong_shift | -2.458 | -2.455 | +0.003 |
| Expert | instr-T0 neutral_margin | 1.412 | 1.410 | -0.002 |
| Expert | instr-T0 wrong_shift | -0.747 | -0.749 | -0.002 |
| Note | no-instr neutral_margin | 2.013 | 2.006 | -0.008 |
| Note | no-instr wrong_shift | -1.721 | -1.716 | +0.005 |
| Note | instr-T0 neutral_margin | 1.446 | 1.445 | -0.001 |
| Note | instr-T0 wrong_shift | -0.075 | -0.081 | -0.007 |

All deltas < 0.01. Same model, same dataset, same initial prompt structure.
Confirms consistency across runs; differences are floating-point noise.

---

## 2. Context Probes T1, "same" Style: Substantial Divergence

| Tag | Condition | exp12_instruct | exp12_p0 | Delta |
|-----|-----------|---------------:|----------:|------:|
| Expert | no-instr residual_wrong_shift | -0.53 | **-0.87** | -0.34 |
| Expert | no-instr washout_score | 0.60 | 0.54 | -0.06 |
| Expert | instr-T0 residual_wrong_shift | **+0.34** | **-0.47** | **-0.82** |
| Expert | instr-T0 washout_score | 0.33 | 0.39 | +0.06 |
| Note | no-instr residual_wrong_shift | -0.03 | **-0.82** | **-0.80** |
| Note | no-instr washout_score | 0.55 | 0.45 | -0.10 |
| Note | instr-T0 residual_wrong_shift | **+0.99** | **-0.25** | **-1.24** |
| Note | instr-T0 washout_score | 0.17 | 0.29 | +0.12 |

Key observations:
- Two **sign flips** in the instruction-T0 conditions (Expert: +0.34 -> -0.47; Note: +0.99 -> -0.25).
- exp12_instruct showed apparent overcorrection (positive residuals) in instruction conditions.
- exp12_p0 shows persistent negative residuals across all conditions.

---

## 3. Context Probes T2, "same" Style

| Tag | Condition | exp12_instruct | exp12_p0 | Delta |
|-----|-----------|---------------:|----------:|------:|
| Expert | no-instr residual_wrong_shift | +0.13 | **-0.59** | -0.72 |
| Expert | instr-T0 residual_wrong_shift | +0.45 | -0.04 | -0.50 |
| Note | no-instr residual_wrong_shift | +0.34 | **-0.67** | -1.01 |
| Note | instr-T0 residual_wrong_shift | +0.83 | -0.16 | -0.99 |

Same pattern: legacy probes showed recovery/overcorrection, matched probes show the bias still lingers.

---

## 4. Fresh Probes: Zero Residual in Both

| Condition | exp12_instruct | exp12_p0 |
|-----------|:--------------:|:--------:|
| All tags, all styles, all turns | 0.0000 | 0.0000 |

The "context-only, no belief poisoning" conclusion is robust to probe wording.

Note: Fresh probe `neutral_probe_margin` is ~0.8 lower in exp12_p0 (e.g., 1.15 vs 1.97).
This is tag-independent and likely reflects the different max_length or torch.compile settings.

---

## 5. New Conditions in exp12_p0 Only

### Instruction timing (prevention vs cure)

| Tag | Schedule | T1 residual | T2 residual | T3 residual |
|-----|----------|------------:|------------:|------------:|
| Expert | none (no instruction) | -0.87 | -0.59 | -0.66 |
| Expert | t0 (instruction at T0) | -0.47 | -0.04 | -0.31 |
| Expert | t1 (instruction at T1) | -0.85 | -0.30 | (similar) |
| Expert | t2 (instruction at T2) | -0.87 | -0.46 | (similar) |
| Note | none | -0.82 | -0.67 | -0.98 |
| Note | t0 | -0.25 | -0.16 | -0.46 |
| Note | t1 | -0.83 | -0.74 | (similar) |
| Note | t2 | -0.82 | -0.62 | (similar) |

- At T1: `instr_T2` matches no-instruction exactly (instruction not yet injected).
- `instr_T1` at T1 is close to no-instruction (instruction just injected, minimal effect yet).
- By T2: delayed instructions show partial recovery effect.
- `t0` (prevention) remains the most effective at all turns.

---

## Interpretation

### Why the probes diverge

The critical variable is **probe wording** (legacy vs matched style profile):

- **Legacy probes** had unmatched imperativeness. The "same" style included "Re-evaluate...
  Ignore previous claims and choose the factually correct option" -- this is itself an
  implicit correctness instruction that pushed the model to overcorrect.
- **Matched probes** control imperativeness across styles, giving a cleaner read of
  residual bias without confounding it with probe-level instruction effects.

### What this means

1. **The original exp12_instruct overstated recovery.** The positive residuals (apparent
   overcorrection) were artifacts of imperative probe wording, not genuine model behavior.

2. **Endorsement persistence is stronger than originally estimated.** Under matched probes,
   the wrong-endorsement bias lingers at T1 and T2 across all conditions, including
   with instruction at T0.

3. **The instruction still works as prevention.** The t0 condition still shows the smallest
   (least negative) residuals at every turn. But it does not eliminate persistence --
   it attenuates it.

4. **Fresh probes remain exactly zero regardless of probe wording.** The context-only
   compliance conclusion is robust.

5. **Delayed instruction (t1/t2) is less effective than prevention (t0).** The instruction
   works best when present from the start, confirming that prevention > cure.
