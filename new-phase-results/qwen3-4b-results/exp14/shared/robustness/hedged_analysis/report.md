# Exp14 Evidence-Quality Analysis

This report summarizes monotonic evidence scaling and asymmetry diagnostics across prior-stratified slices.

## Qwen/Qwen3-4B-Instruct-2507
- n_records: 1006
- tags: expert, note
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_expert_bare`

- slice counts: all=1006, prior_correct=795, prior_wrong=209, high_conf_top25=53, top10=21

### Tag: expert
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.416, correct=0.368
  prior-wrong Page p-value: wrong=1.45e-33, correct=5.584e-25
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=2.391, CI95=[1.697, 3.067], positive_frac=0.689
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.418, correct=0.667
  prior-wrong Page p-value: wrong=2.239e-33, correct=6.414e-80
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=7.387, CI95=[6.718, 8.090], positive_frac=0.938

### Tag: note
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.025, correct=0.491
  prior-wrong Page p-value: wrong=0.2181, correct=2.843e-43
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=4.855, CI95=[4.125, 5.623], positive_frac=0.828
- Instruction `I1`
  prior-wrong Kendall tau: wrong=-0.103, correct=0.537
  prior-wrong Page p-value: wrong=0.999, correct=2.006e-52
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=7.034, CI95=[6.273, 7.841], positive_frac=0.890

## Qwen/Qwen3-4B-Thinking-2507
- n_records: 1006
- tags: expert, note
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_expert_bare`

- slice counts: all=1006, prior_correct=772, prior_wrong=223, high_conf_top25=56, top10=23

### Tag: expert
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.599, correct=0.713
  prior-wrong Page p-value: wrong=9.623e-66, correct=1.632e-92
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.865, CI95=[3.619, 4.112], positive_frac=0.978
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.649, correct=0.740
  prior-wrong Page p-value: wrong=1.361e-78, correct=9.95e-98
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=2.135, CI95=[1.987, 2.286], positive_frac=0.982

### Tag: note
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.575, correct=0.800
  prior-wrong Page p-value: wrong=2.027e-62, correct=3.655e-110
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.792, CI95=[3.573, 4.017], positive_frac=0.996
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.525, correct=0.776
  prior-wrong Page p-value: wrong=6.253e-55, correct=9.641e-104
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=2.217, CI95=[2.062, 2.387], positive_frac=0.987

## meta-llama/Llama-3.1-8B-Instruct
- n_records: 1006
- tags: expert, note
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_expert_bare`

- slice counts: all=1006, prior_correct=816, prior_wrong=173, high_conf_top25=44, top10=18

### Tag: expert
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.573, correct=0.677
  prior-wrong Page p-value: wrong=2.771e-46, correct=6.932e-67
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.915, CI95=[3.694, 4.147], positive_frac=0.994
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.602, correct=0.757
  prior-wrong Page p-value: wrong=5.894e-50, correct=8.488e-82
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.233, CI95=[3.054, 3.404], positive_frac=0.994

### Tag: note
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.413, correct=0.724
  prior-wrong Page p-value: wrong=2.025e-27, correct=5.5e-76
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=4.198, CI95=[3.968, 4.430], positive_frac=0.994
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.253, correct=0.811
  prior-wrong Page p-value: wrong=5.997e-11, correct=9.003e-90
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.460, CI95=[3.232, 3.686], positive_frac=1.000
