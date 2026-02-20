# Exp14 Evidence-Quality Analysis

This report summarizes monotonic evidence scaling and asymmetry diagnostics across prior-stratified slices.

## Qwen/Qwen3-4B-Instruct-2507
- n_records: 807
- tags: expert, note
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_expert_bare`

- slice counts: all=807, prior_correct=561, prior_wrong=243, high_conf_top25=61, top10=25

### Tag: expert
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.281, correct=0.422
  prior-wrong Page p-value: wrong=5.611e-19, correct=1.157e-40
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=2.620, CI95=[2.077, 3.119], positive_frac=0.782
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.358, correct=0.544
  prior-wrong Page p-value: wrong=1.911e-28, correct=1.472e-63
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=5.774, CI95=[5.145, 6.420], positive_frac=0.852

### Tag: note
- Instruction `I0`
  prior-wrong Kendall tau: wrong=-0.039, correct=0.463
  prior-wrong Page p-value: wrong=0.8594, correct=3.038e-47
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=4.359, CI95=[3.704, 5.020], positive_frac=0.811
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.126, correct=0.463
  prior-wrong Page p-value: wrong=7.566e-05, correct=1.183e-45
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=5.684, CI95=[4.928, 6.428], positive_frac=0.815

## Qwen/Qwen3-4B-Thinking-2507
- n_records: 807
- tags: expert, note
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_expert_bare`

- slice counts: all=807, prior_correct=571, prior_wrong=218, high_conf_top25=55, top10=22

### Tag: expert
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.469, correct=0.630
  prior-wrong Page p-value: wrong=2.245e-44, correct=7.853e-75
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=2.659, CI95=[2.434, 2.857], positive_frac=0.963
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.467, correct=0.536
  prior-wrong Page p-value: wrong=2.206e-43, correct=2.827e-57
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=1.369, CI95=[1.228, 1.519], positive_frac=0.927

### Tag: note
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.530, correct=0.732
  prior-wrong Page p-value: wrong=5.727e-53, correct=1.088e-94
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.212, CI95=[3.025, 3.413], positive_frac=0.991
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.449, correct=0.665
  prior-wrong Page p-value: wrong=2.084e-40, correct=9.861e-81
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=1.909, CI95=[1.752, 2.059], positive_frac=0.968

## meta-llama/Llama-3.1-8B-Instruct
- n_records: 807
- tags: expert, note
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_expert_bare`

- slice counts: all=807, prior_correct=510, prior_wrong=279, high_conf_top25=70, top10=28

### Tag: expert
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.697, correct=0.815
  prior-wrong Page p-value: wrong=1.056e-108, correct=1.013e-139
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.531, CI95=[3.358, 3.700], positive_frac=0.996
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.798, correct=0.829
  prior-wrong Page p-value: wrong=5.216e-132, correct=1.222e-144
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.050, CI95=[2.907, 3.186], positive_frac=0.989

### Tag: note
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.649, correct=0.798
  prior-wrong Page p-value: wrong=2.259e-95, correct=2.875e-139
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.913, CI95=[3.728, 4.097], positive_frac=0.993
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.552, correct=0.868
  prior-wrong Page p-value: wrong=2.245e-69, correct=7.877e-153
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.490, CI95=[3.313, 3.670], positive_frac=0.989
