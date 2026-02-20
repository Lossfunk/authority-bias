# Exp14 Evidence-Quality Analysis

This report summarizes monotonic evidence scaling and asymmetry diagnostics across prior-stratified slices.

## meta-llama/Llama-3.1-8B-Instruct
- n_records: 1813
- tags: user
- evidence_levels: bare, reason1, reason2, reason_data
- prior code: `N0_user_bare`

- slice counts: all=1813, prior_correct=1300, prior_wrong=472, high_conf_top25=118, top10=48

### Tag: user
- Instruction `I0`
  prior-wrong Kendall tau: wrong=0.749, correct=0.516
  prior-wrong Page p-value: wrong=6.754e-212, correct=2.708e-110
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=2.233, CI95=[2.046, 2.408], positive_frac=0.860
- Instruction `I1`
  prior-wrong Kendall tau: wrong=0.572, correct=0.527
  prior-wrong Page p-value: wrong=3.063e-131, correct=1.652e-114
  key diagnostic (E_last - E_first correct shift, prior-wrong): mean=1.916, CI95=[1.790, 2.045], positive_frac=0.915
