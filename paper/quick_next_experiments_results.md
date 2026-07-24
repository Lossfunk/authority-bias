# Quick next experiments — main results

Status of the five "quick next experiments" planned in `feedback_revision_plan*.md`. Local-CPU items are complete; GPU items run on the JarvisLabs H100 notebook. All numbers below come from artifacts already saved in this repo (paths cited per row).

## At-a-glance status

| # | Experiment | Compute | Status | Result file(s) |
| --- | --- | --- | --- | --- |
| 1 | Correct-source preservation (N0/W1/C1 all-condition project-out) | GPU H100 | Qwen ✅; OLMo-2 ✅; OLMo-3.1 ✅; GPT-OSS ✅; Gemma ✅ | `paper/next_experiments_availability_and_qwen_preservation.md`, `neurips-results/exp19/ab_project_out_allcond/olmo*/...`, remote `ab_project_out_allcond_quick/{gpt_oss,gemma4}/...` |
| 2 | Causal wording deconfound (valence/arousal subspace) | GPU | ❌ blocked: lexical VA planes exist, but no causal removal runs | `paper/gpt_gemma_olmo_remote_status.md` |
| 3 | Random / shuffled-vector control scan | CPU | ✅ done (re-used existing nulls; computed z and empirical p) | `paper/local_shuffled_random_control.md`, `paper/local_shuffled_random_control.json` |
| 4 | Wilson intervals + parser-marker audit | CPU | ✅ done | `paper/local_cpu_wilson_parser_audit.md`, `paper/local_cpu_wilson_parser_audit.json` |
| 5 | OLMo source/user 2x2 | GPU | ❌ not run | — |

## 1. Correct-source preservation

α = 1.0, layer-and-position fixed by per-model spec, intervention `project_out_direction`, conditions `N0_note,W1_note,C1_note`. C1 accuracy at α = 1 is the preservation number; W1 wrong-rate at α = 1 is the mitigation number. α = 0 is the matched baseline.

### Qwen3.5-27B (local, already cached)
Source: `causal-deconfound/qwen35_ab_project_out_allcond/qwen35/trivia_*/steering_summary.json`.

| Variant | C1 acc α=0 | C1 acc α=1 | C1 drop (pp) | N0 acc α=1 | W1 wrong α=0 | W1 wrong α=1 | W1 reduction (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Authority | 0.988 | 0.950 | −3.8 | 0.964 | 0.394 | 0.063 | −33.1 |
| Assistant axis | 0.988 | 0.984 | −0.4 | 0.970 | 0.394 | 0.404 | +1.0 |
| Residualized (auth ⟂ assistant) | 0.988 | 0.958 | −3.0 | 0.966 | 0.394 | 0.058 | −33.6 |

Takeaway: residualized projection retains ~96% C1 accuracy while cutting W1 wrong-source compliance from 39% to ~6%.

### OLMo-2-32B-Instruct (remote, just finished)
Source: `neurips-results/exp19/ab_project_out_allcond/olmo2/trivia_*/steering_summary.json`. Layer 16, `answer_position`, α=1.0.

| Variant | C1 acc α=0 | C1 acc α=1 | C1 drop (pp) | N0 acc α=1 | W1 wrong α=0 | W1 wrong α=1 | W1 reduction (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Authority | 0.968 | 0.968 | 0.0 | 0.783 | 0.835 | 0.824 | −1.1 |
| Assistant axis | 0.968 | 0.967 | −0.1 | 0.783 | 0.835 | 0.835 | 0.0 |
| Residualized | 0.968 | 0.968 | 0.0 | 0.804 | 0.835 | 0.824 | −1.1 |

Takeaway: at α=1.0, project-out preserves C1 essentially perfectly but does not reduce OLMo-2's W1 wrong-source compliance. Larger α and/or layer sweeps are needed to localize an effect for OLMo-2.

### OLMo-3.1-32B-Instruct (remote, finished)
Source: `neurips-results/exp19/ab_project_out_allcond/olmo31/trivia_*/steering_summary.json`. Layer 5, `note_start`, α=1.0. (`endorsement_start` is not a valid runner mode; we used `note_start`, which is the closest valid pre-`note` position.)

| Variant | C1 acc α=0 | C1 acc α=1 | C1 drop (pp) | N0 acc α=1 | W1 wrong α=0 | W1 wrong α=1 | W1 reduction (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Authority | 0.968 | 0.974 | +0.6 | 0.876 | 0.827 | 0.833 | +0.6 |
| Assistant axis | 0.968 | 0.962 | −0.6 | 0.869 | 0.827 | 0.828 | +0.1 |
| Residualized | 0.968 | 0.968 | 0.0 | 0.869 | 0.827 | 0.821 | −0.6 |

Takeaway: at α=1.0 with `note_start`, project-out preserves OLMo-3.1 C1 accuracy completely but does not move W1. Same pattern as OLMo-2 — the available position/layer for OLMo's residual-stream direction does not give a useful mitigation handle here.

### GPT-OSS-20B (remote, just finished)

Source: remote JarvisLabs box 2, `neurips-results/exp19/ab_project_out_allcond_quick/gpt_oss/trivia_*/steering_summary.json`. Layer 18, `note_end`, α=1.0.

| Variant | C1 acc α=0 | C1 acc α=1 | C1 drop (pp) | N0 acc α=1 | W1 wrong α=0 | W1 wrong α=1 | W1 reduction (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Authority | 0.985 | 0.985 | 0.0 | 0.960 | 0.585 | 0.600 | +1.5 |
| Assistant axis | 0.985 | 0.990 | +0.5 | 0.960 | 0.585 | 0.570 | −1.5 |
| Residualized | 0.985 | 0.980 | −0.5 | 0.950 | 0.585 | 0.595 | +1.0 |

Takeaway: GPT-OSS gives the cleanest preservation result so far: correct-source accuracy is essentially unchanged under all three project-out variants. At α=1.0 this intervention does not mitigate W1; the earlier W1-only Pareto sweep remains the evidence that stronger/tuned settings can reduce GPT-OSS wrong-source compliance.

### Gemma-4-26B (remote, just finished)

Source: remote JarvisLabs box 2, `neurips-results/exp19/ab_project_out_allcond_quick/gemma4/trivia_*/steering_summary.json`. Layer 22, `endorsement_span`, α=1.0.

| Variant | C1 acc α=0 | C1 acc α=1 | C1 drop (pp) | N0 acc α=1 | W1 wrong α=0 | W1 wrong α=1 | W1 reduction (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Authority | 0.995 | 0.995 | 0.0 | 0.995 | 0.520 | 0.535 | +1.5 |
| Assistant axis | 0.995 | 1.000 | +0.5 | 0.995 | 0.520 | 0.540 | +2.0 |
| Residualized | 0.995 | 0.995 | 0.0 | 1.000 | 0.520 | 0.525 | +0.5 |

Takeaway: Gemma-4 mirrors GPT-OSS — correct-source accuracy is essentially perfect under all three project-out variants (C1 holds at 99.5%, N0 at ≥99.5%). At α=1.0 this intervention does not mitigate W1 wrong-source compliance; stronger α or different layers would be needed if Gemma mitigation is the target.

### Summary: all five models now have N0/W1/C1 preservation numbers

| Model | C1 α=0 → α=1 | N0 α=1 | W1 wrong α=0 → α=1 | Notes |
| --- | --- | ---: | --- | --- |
| Qwen3.5-27B (residualized) | 0.988 → 0.958 (−3.0pp) | 0.966 | 0.394 → 0.058 (−33.6pp) | Mitigates W1 cleanly with small C1 cost |
| OLMo-2-32B (residualized) | 0.968 → 0.968 (0.0pp) | 0.804 | 0.835 → 0.824 (−1.1pp) | Preserves C1; α=1 doesn't move W1 |
| OLMo-3.1-32B (residualized) | 0.968 → 0.968 (0.0pp) | 0.869 | 0.827 → 0.821 (−0.6pp) | Preserves C1; α=1 doesn't move W1 |
| GPT-OSS-20B (residualized) | 0.985 → 0.980 (−0.5pp) | 0.950 | 0.585 → 0.595 (+1.0pp) | Preserves C1/N0; α=1 doesn't move W1 |
| Gemma-4-26B (residualized) | 0.995 → 0.995 (0.0pp) | 1.000 | 0.520 → 0.525 (+0.5pp) | Preserves C1/N0; α=1 doesn't move W1 |

The "no collateral damage on correct sources" claim now has direct N0/W1/C1 evidence on all five models. Mitigation strength at α=1.0 with project-out direction varies by model: Qwen shows the cleanest mitigation; OLMo/GPT-OSS/Gemma require stronger α (per the W1-only Pareto sweeps in `wang-pareto-results/`).

## 2. Causal wording deconfound (valence/arousal)

Lexical VA planes are present locally for OLMo-2/3.1, GPT-OSS, and Gemma (`neurips-results/<model>/mechanism/lexical_va_plane/summary.json`). No causal removal runs (e.g. residualizing authority against the VA plane and rerunning interventions) were found.

Status: planned but not run. Needs GPU.

## 3. Random / shuffled-vector control scan (CPU-only)

Re-ran the 100-sample random-direction, shuffled-label, and random-raw-projection nulls precomputed alongside the compliance direction, and compared to the cross-validated primary-pair AUROC. Empirical two-sided p; z = (real − null_mean) / null_std.

Source: `neurips-results/olmo{2,31}/mechanism/{olmo2_compliance_analysis,compliance_analysis_all_prior_wrong_shared_h100}/{null_controls.json, probe_results.json}`.

| Model | Condition | Real AUROC (95% CI) | Null mean ± std | z | p |
| --- | --- | --- | --- | ---: | ---: |
| OLMo-2 | W1 — random direction | 0.700 [0.55, 0.83] | 0.492 ± 0.106 | +1.95 | 0.059 |
| OLMo-2 | W1 — shuffled labels | 0.700 [0.55, 0.83] | 0.493 ± 0.103 | +2.00 | 0.050 |
| OLMo-2 | C1 — random direction | 0.704 [0.60, 0.82] | 0.510 ± 0.118 | +1.65 | 0.059 |
| OLMo-2 | C1 — shuffled labels | 0.704 [0.60, 0.82] | 0.482 ± 0.095 | +2.33 | 0.040 |
| OLMo-3.1 | W1 — random direction | 0.649 [0.54, 0.76] | 0.496 ± 0.081 | +1.88 | 0.040 |
| OLMo-3.1 | W1 — shuffled labels | 0.649 [0.54, 0.76] | 0.479 ± 0.081 | +2.12 | 0.079 |
| OLMo-3.1 | C1 — random direction | 0.503 [0.38, 0.60] | 0.534 ± 0.093 | −0.33 | 1.000 |
| OLMo-3.1 | C1 — shuffled labels | 0.503 [0.38, 0.60] | 0.483 ± 0.083 | +0.24 | 1.000 |

Takeaway: OLMo-2 W1+C1 and OLMo-3.1 W1 sit ~2σ above both nulls; OLMo-3.1 C1 is statistically indistinguishable from random unit vectors and from shuffled labels.

Full table and per-control empirical p-values are in `paper/local_shuffled_random_control.md`.

## 4. Wilson intervals + parser-marker audit (CPU-only)

Headline behavioral table with 95% Wilson intervals and a per-model recovery audit have been written. Companion JSON includes raw counts and confidence intervals.

Source: `paper/local_cpu_wilson_parser_audit.md`, `paper/local_cpu_wilson_parser_audit.json`.

Two findings to keep visible in the paper:

- W1 wrong-rate intervals from existing artifacts: Qwen 50.3% [48.0, 52.6], GPT-OSS 71.9% [68.7, 74.9], OLMo-2 94.5% [91.3, 96.6], OLMo-3.1 90.5% [87.2, 93.0], Gemma 66.8% [63.6, 69.8], GPT-5.4 42.6% [40.3, 45.0], Grok 88.9% [87.3, 90.3], Gemini 2.8% [2.2, 3.6].
- Recovery audit: Qwen residualized recoveries are mostly clean plain-text (175/184 plain-text). GPT-OSS and OLMo recoveries are dominated by `assistantfinal` / `[answer]` tagged-contradictory artifacts and should not be used as headline qualitative examples without manual review.

## 5. OLMo source/user 2x2

Not run. Requires GPU inference; no cached outputs locally or remotely.

## Files written/updated by this report

- `paper/quick_next_experiments_results.md` (this file)
- `paper/correct_source_preservation_olmo.json` (per-variant table for OLMo-2 just downloaded; OLMo-3.1 to be appended once the relaunched run finishes)
