# Exp14 Evidence-Quality Analysis

This report summarizes monotonic evidence scaling and asymmetry diagnostics across prior-stratified slices.

## Qwen/Qwen3-4B-Instruct-2507
- n_records: 1813
- tags: expert, note, someone_online
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_expert_bare`

- slice counts: all=1813, prior_correct=1356, prior_wrong=451, high_conf_top25=113, top10=46

### Tag: expert
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.408, correct=0.399
  prior-wrong Page p-value: wrong=2.85e-65, correct=6.567e-65
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=2.599, CI95=[2.184, 3.023], positive_frac=0.772
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.477, correct=0.607
  prior-wrong Page p-value: wrong=1.744e-89, correct=5.119e-139
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=6.233, CI95=[5.725, 6.715], positive_frac=0.874

### Tag: note
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.125, correct=0.493
  prior-wrong Page p-value: wrong=7.181e-08, correct=3.786e-93
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=4.641, CI95=[4.164, 5.135], positive_frac=0.827
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.169, correct=0.490
  prior-wrong Page p-value: wrong=9.16e-13, correct=8.393e-94
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=6.207, CI95=[5.615, 6.778], positive_frac=0.858

### Tag: someone_online
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.273, correct=0.581
  prior-wrong Page p-value: wrong=1.577e-32, correct=8.772e-128
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=6.883, CI95=[6.322, 7.433], positive_frac=0.878
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.221, correct=0.544
  prior-wrong Page p-value: wrong=7.173e-21, correct=1.977e-112
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=10.014, CI95=[9.311, 10.733], positive_frac=0.898
