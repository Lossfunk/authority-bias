# Exp14 Evidence-Quality Analysis

This report summarizes monotonic evidence scaling and asymmetry diagnostics across prior-stratified slices.

## Qwen/Qwen3-4B-Instruct-2507
- n_records: 1813
- tags: user
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_user_bare`

- slice counts: all=1813, prior_correct=1330, prior_wrong=478, high_conf_top25=120, top10=48

### Tag: user
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.225, correct=0.546
  prior-wrong Page p-value: wrong=1.054e-22, correct=3.54e-121
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=5.753, CI95=[5.205, 6.284], positive_frac=0.837
- Instruction `I1`
  prior-wrong Kendall tau: wrong=-0.088, correct=0.336
  prior-wrong Page p-value: wrong=1, correct=5.383e-46
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=3.801, CI95=[3.123, 4.484], positive_frac=0.686
