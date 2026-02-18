# Exp14 Evidence-Quality Analysis

This report summarizes monotonic evidence scaling and asymmetry diagnostics across prior-stratified slices.

## Qwen/Qwen3-4B-Instruct-2507
- n_records: 1813
- tags: expert, note
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_expert_bare`

- slice counts: all=1813, prior_correct=1357, prior_wrong=450, high_conf_top25=113, top10=45

### Tag: expert
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.410, correct=0.402
  prior-wrong Page p-value: wrong=4.425e-66, correct=2.064e-65
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=2.593, CI95=[2.159, 3.018], positive_frac=0.767
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.465, correct=0.605
  prior-wrong Page p-value: wrong=4.872e-85, correct=3.439e-137
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=6.234, CI95=[5.756, 6.735], positive_frac=0.869

### Tag: note
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.124, correct=0.481
  prior-wrong Page p-value: wrong=9.907e-08, correct=1.298e-88
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=4.624, CI95=[4.141, 5.118], positive_frac=0.820
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.181, correct=0.497
  prior-wrong Page p-value: wrong=4.794e-14, correct=1.186e-96
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=6.207, CI95=[5.616, 6.753], positive_frac=0.867

## Qwen/Qwen3-4B-Thinking-2507
- n_records: 1813
- tags: expert, note
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_expert_bare`

- slice counts: all=1813, prior_correct=1344, prior_wrong=441, high_conf_top25=111, top10=45

### Tag: expert
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.557, correct=0.646
  prior-wrong Page p-value: wrong=2.613e-117, correct=3.431e-154
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.188, CI95=[3.021, 3.358], positive_frac=0.959
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.538, correct=0.595
  prior-wrong Page p-value: wrong=1.216e-110, correct=1.133e-139
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=1.661, CI95=[1.560, 1.764], positive_frac=0.946

### Tag: note
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.610, correct=0.740
  prior-wrong Page p-value: wrong=1.715e-134, correct=2.419e-192
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.410, CI95=[3.273, 3.558], positive_frac=0.995
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.534, correct=0.709
  prior-wrong Page p-value: wrong=1.323e-111, correct=2.534e-179
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=2.012, CI95=[1.912, 2.109], positive_frac=0.984

## meta-llama/Llama-3.1-8B-Instruct
- n_records: 1813
- tags: expert, note
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_expert_bare`

- slice counts: all=1813, prior_correct=1328, prior_wrong=451, high_conf_top25=113, top10=46

### Tag: expert
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.703, correct=0.718
  prior-wrong Page p-value: wrong=3.271e-178, correct=6.878e-187
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.571, CI95=[3.432, 3.713], positive_frac=0.987
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.799, correct=0.778
  prior-wrong Page p-value: wrong=7.605e-219, correct=2.153e-217
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.063, CI95=[2.944, 3.182], positive_frac=0.993

### Tag: note
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.734, correct=0.732
  prior-wrong Page p-value: wrong=3.286e-190, correct=1.411e-197
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.957, CI95=[3.805, 4.107], positive_frac=0.980
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.698, correct=0.821
  prior-wrong Page p-value: wrong=1.057e-173, correct=1.848e-228
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.445, CI95=[3.300, 3.585], positive_frac=0.987
