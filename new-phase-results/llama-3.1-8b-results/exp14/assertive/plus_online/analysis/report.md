# Exp14 Evidence-Quality Analysis

This report summarizes monotonic evidence scaling and asymmetry diagnostics across prior-stratified slices.

## meta-llama/Llama-3.1-8B-Instruct
- n_records: 1813
- tags: expert, note, someone_online
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_expert_bare`

- slice counts: all=1813, prior_correct=1327, prior_wrong=447, high_conf_top25=112, top10=45

### Tag: expert
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.705, correct=0.718
  prior-wrong Page p-value: wrong=9.569e-177, correct=2.533e-185
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.556, CI95=[3.415, 3.694], positive_frac=0.987
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.805, correct=0.777
  prior-wrong Page p-value: wrong=2.566e-219, correct=3.579e-214
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.058, CI95=[2.944, 3.175], positive_frac=0.996

### Tag: note
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.724, correct=0.737
  prior-wrong Page p-value: wrong=3.767e-186, correct=2.975e-198
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.971, CI95=[3.818, 4.124], positive_frac=0.984
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.680, correct=0.817
  prior-wrong Page p-value: wrong=6.031e-165, correct=4.303e-225
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.441, CI95=[3.292, 3.587], positive_frac=0.987

### Tag: someone_online
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.675, correct=0.619
  prior-wrong Page p-value: wrong=1.252e-165, correct=4.897e-143
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.716, CI95=[3.535, 3.885], positive_frac=0.964
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.585, correct=0.682
  prior-wrong Page p-value: wrong=2.992e-129, correct=2.465e-172
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.071, CI95=[2.929, 3.212], positive_frac=0.987
