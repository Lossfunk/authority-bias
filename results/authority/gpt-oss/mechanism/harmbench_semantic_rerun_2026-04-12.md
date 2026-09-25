# HarmBench Semantic Rerun (2026-04-12)

This rerun replaces the old single `degenerate` bucket with a semantic parser that separates:

- `clean_refusal`
- `refusal_truncated`
- `refusal_meta`
- `clean_compliance`
- `compliance_meta`
- `prompt_regurgitation`
- `corrupted_generation`
- `meta_only_no_answer`
- `mixed_or_ambiguous`

Key aggregate rates:

- `initial_full_sweep`
  - semantic refusal: `96.47%`
  - semantic compliance: `0.80%`
  - invalid generation: `1.30%`
- `followup_multilayer`
  - semantic refusal: `93.08%`
  - semantic compliance: `0.25%`
  - invalid generation: `4.75%`
- `followup_w1_source`
  - semantic refusal: `81.21%`
  - semantic compliance: `0.77%`
  - invalid generation: `15.75%`
  - prompt regurgitation alone: `15.58%`
- `followup_decode_time`
  - semantic refusal: `0.00%`
  - semantic compliance: `0.00%`
  - invalid generation: `100.00%`

Selected configs:

- `baseline_L16`
  - semantic refusal: `98.00%`
  - semantic compliance: `0.75%`
  - invalid generation: `0.25%`
- `interpolate_mean_a1_L16`
  - initial sweep: refusal `91.50%`, compliance `0.50%`, invalid `6.25%`
  - W1 source: refusal `52.00%`, compliance `1.00%`, invalid `43.00%`
- `replace_mean_a1_L16`
  - initial sweep: refusal `92.00%`, compliance `0.75%`, invalid `5.00%`
  - W1 source: refusal `53.50%`, compliance `0.75%`, invalid `41.75%`
- `interpolate_mean_a1_L16-18-20`
  - multilayer: refusal `92.50%`, compliance `0.75%`, invalid `4.50%`

Interpretation:

- The semantic parser supports the concern that the old `degenerate` bucket was too blunt.
- Most of the disputed W1-source L16 mass is not clean compliance; it is mostly `prompt_regurgitation`.
- Decode-time remains non-interpretable as compliance and is now cleanly isolated as `corrupted_generation`.
- The remaining clean compliance signal is small but non-zero in the prompt-time runs.
