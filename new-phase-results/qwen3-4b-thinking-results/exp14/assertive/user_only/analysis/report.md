# Exp14 Evidence-Quality Analysis

This report summarizes monotonic evidence scaling and asymmetry diagnostics across prior-stratified slices.

## Qwen/Qwen3-4B-Thinking-2507
- n_records: 1813
- tags: user
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_user_bare`

- slice counts: all=1813, prior_correct=1272, prior_wrong=518, high_conf_top25=130, top10=52

### Tag: user
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.375, correct=0.281
  prior-wrong Page p-value: wrong=3.172e-67, correct=1.059e-41
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=0.697, CI95=[0.564, 0.834], positive_frac=0.649
- Instruction `I1`
  prior-wrong Kendall tau: wrong=-0.167, correct=-0.414
  prior-wrong Page p-value: wrong=1, correct=1
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=-1.140, CI95=[-1.218, -1.061], positive_frac=0.106
