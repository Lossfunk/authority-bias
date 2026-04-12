# HarmBench Parser Rerun (2026-04-12)

This rerun applies a stricter post-hoc classifier to all four HarmBench result sets:

- stronger decode-time degeneration detection
- truncated-final refusal detection (`assistantfinalI…`, `assistantfinalI’m…`)
- prompt-regurgitation / repeated `User request` loop detection
- stricter invalid-compliance filtering for analysis-only meta outputs

Updated aggregate overview:

- initial_full_sweep: refusal `98.08%`, compliance `0.37%`, degenerate `1.05%`
- followup_multilayer: refusal `97.79%`, compliance `0.46%`, degenerate `1.67%`
- followup_decode_time: refusal `0.00%`, compliance `0.00%`, degenerate `100.00%`
- followup_w1_source: refusal `92.13%`, compliance `0.44%`, degenerate `7.33%`

Selected configs:

- `baseline_L16`: compliance `0.25%`, refusal `99.00%`
- `interpolate_mean_a1_L16`: compliance `0.50%`, refusal `97.00%`, degenerate `2.25%`
- `replace_mean_a1_L16`: compliance `0.50%`, refusal `96.50%`, degenerate `2.50%`
- `interpolate_mean_a1_L16-18-20`: compliance `0.50%`, refusal `96.75%`, degenerate `2.50%`
- `replace_mean_a1_L16`: compliance `0.75%`, refusal `80.00%`, degenerate `19.00%` in `followup_w1_source`

Interpretation:

- decode-time patching is not a valid compliance result under the stricter parser; it collapses entirely into malformed generation
- the original prompt-time effect becomes much smaller after removing prompt-copy loops, policy-analysis drafts, and truncated refusals
- W1-source L16 still moves the distribution more than the neutral-source runs, but most of that movement lands in `degenerate` rather than clean compliance
- the remaining clean compliance signal is small enough that any writeup should foreground the artifact-sensitive nature of these interventions
