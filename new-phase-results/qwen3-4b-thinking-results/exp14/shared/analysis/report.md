# Exp14 Evidence-Quality Analysis

This report summarizes monotonic evidence scaling and asymmetry diagnostics across prior-stratified slices.

## Qwen/Qwen3-4B-Instruct-2507
- n_records: 1813
- tags: expert, note
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_expert_bare`

- slice counts: all=1813, prior_correct=1356, prior_wrong=452, high_conf_top25=113, top10=46

### Tag: expert
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.343, correct=0.397
  prior-wrong Page p-value: wrong=7.81e-49, correct=4.269e-63
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=2.514, CI95=[2.068, 2.926], positive_frac=0.739
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.386, correct=0.601
  prior-wrong Page p-value: wrong=2.064e-59, correct=5.839e-140
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=6.520, CI95=[6.037, 7.005], positive_frac=0.892

### Tag: note
- Instruction `I0`
  prior-wrong Kendall tau: wrong=-0.009, correct=0.476
  prior-wrong Page p-value: wrong=0.6028, correct=2.262e-88
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=4.588, CI95=[4.102, 5.054], positive_frac=0.819
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.020, correct=0.497
  prior-wrong Page p-value: wrong=0.2469, correct=1.999e-95
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=6.308, CI95=[5.774, 6.874], positive_frac=0.850

## Qwen/Qwen3-4B-Thinking-2507
- n_records: 1813
- tags: expert, note
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_expert_bare`

- slice counts: all=1813, prior_correct=1343, prior_wrong=441, high_conf_top25=111, top10=45

### Tag: expert
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.535, correct=0.672
  prior-wrong Page p-value: wrong=5.478e-107, correct=1.066e-164
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.269, CI95=[3.086, 3.444], positive_frac=0.971
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.559, correct=0.639
  prior-wrong Page p-value: wrong=2.613e-117, correct=3.095e-150
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=1.756, CI95=[1.647, 1.870], positive_frac=0.955

### Tag: note
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.553, correct=0.766
  prior-wrong Page p-value: wrong=4.661e-113, correct=2.497e-202
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.506, CI95=[3.353, 3.655], positive_frac=0.993
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.488, correct=0.721
  prior-wrong Page p-value: wrong=1.053e-92, correct=1.398e-181
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=2.065, CI95=[1.952, 2.176], positive_frac=0.977

## meta-llama/Llama-3.1-8B-Instruct
- n_records: 1813
- tags: expert, note
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_expert_bare`

- slice counts: all=1813, prior_correct=1326, prior_wrong=452, high_conf_top25=113, top10=46

### Tag: expert
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.650, correct=0.762
  prior-wrong Page p-value: wrong=2.017e-151, correct=1.888e-203
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.678, CI95=[3.540, 3.817], positive_frac=0.996
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.723, correct=0.802
  prior-wrong Page p-value: wrong=4.051e-177, correct=5.523e-224
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.120, CI95=[3.006, 3.233], positive_frac=0.991

### Tag: note
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.559, correct=0.769
  prior-wrong Page p-value: wrong=1.299e-116, correct=1.037e-212
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=4.022, CI95=[3.877, 4.166], positive_frac=0.993
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.438, correct=0.846
  prior-wrong Page p-value: wrong=4.648e-71, correct=3.253e-240
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.479, CI95=[3.334, 3.617], positive_frac=0.993
