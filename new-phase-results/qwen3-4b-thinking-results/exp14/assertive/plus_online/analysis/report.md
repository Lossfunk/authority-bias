# Exp14 Evidence-Quality Analysis

This report summarizes monotonic evidence scaling and asymmetry diagnostics across prior-stratified slices.

## Qwen/Qwen3-4B-Thinking-2507
- n_records: 1813
- tags: expert, note, someone_online
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_expert_bare`

- slice counts: all=1813, prior_correct=1348, prior_wrong=442, high_conf_top25=111, top10=45

### Tag: expert
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.548, correct=0.640
  prior-wrong Page p-value: wrong=1.982e-114, correct=4.312e-153
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.197, CI95=[3.027, 3.370], positive_frac=0.964
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.541, correct=0.602
  prior-wrong Page p-value: wrong=1.343e-111, correct=6.836e-142
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=1.660, CI95=[1.559, 1.758], positive_frac=0.948

### Tag: note
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.614, correct=0.745
  prior-wrong Page p-value: wrong=3.131e-136, correct=4.237e-195
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.415, CI95=[3.274, 3.559], positive_frac=0.993
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.534, correct=0.705
  prior-wrong Page p-value: wrong=6.401e-112, correct=4.167e-178
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=2.016, CI95=[1.911, 2.117], positive_frac=0.982

### Tag: someone_online
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.461, correct=0.670
  prior-wrong Page p-value: wrong=1.211e-80, correct=3.229e-161
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.306, CI95=[3.111, 3.509], positive_frac=0.973
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.203, correct=0.447
  prior-wrong Page p-value: wrong=9.305e-18, correct=1.738e-75
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=0.965, CI95=[0.862, 1.070], positive_frac=0.799
