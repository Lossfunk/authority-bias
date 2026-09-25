# OLMo-3.1-32B-Instruct — persona-vectors runs

## Active mechanism pipeline (2026-04-20)

All live directories use the **`all_prior_wrong`** subset of Exp10 UIDs.
The slice `subset_mode=all_prior_wrong` means: every UID that OLMo-3.1 got
wrong at N0_note in the Exp10 seed is included. This is the intended
mechanism slice going forward; it gives 504 selected UIDs vs 126 for the
earlier `high_conf_top25` and 252 for `high_conf_top50`.

Dir layout (all tagged `all_prior_wrong`):

- `exp10_n0_seed/` — Exp10 behavioral seed (slice-independent)
- `authoritative_verified_all_prior_wrong_shared_h100/` — v2 style rows (N0/C1/W1)
- `gradient_weak_all_prior_wrong_shared_h100/` — gradient weak style
- `gradient_uncertain_all_prior_wrong_shared_h100/` — gradient uncertain style
- `gradient_assertive_all_prior_wrong_shared_h100/` — gradient assertive style
- `mechanism/authority_activations_all_prior_wrong_shared_h100/` — step 5 activations
- `mechanism/compliance_analysis_all_prior_wrong_shared_h100/` — step 6 directions + probes

All rows were generated with `max_new_tokens=256`, `disable_thinking=True`,
`temperature=0.2`, `top_p=0.95`, `top_k=20`, `batch_size=8`, seed 42,
on an H100 via box 217.18.55.140.

## Mask counts (step 5 extraction)

- `selected_uids`: 504
- `mask_primary`: 311
- `mask_primary_w1`: 158
- `mask_primary_c1`: 153
- `mask_cross`: 262

## Archived earlier runs

- `_archive_top50_20260420_131604/` — prior `high_conf_top50` mechanism tree
  plus the original `high_conf_top25` behavioral dirs. Kept verbatim for
  reproducibility; **not used downstream**. See per-dir `summary.json` for
  slice metadata. The diagnostic comparison (top25 vs top50 vs all_prior_wrong)
  that motivated the slice change is documented in the accompanying session
  summary.

## Why we switched slices

The `high_conf_top{25,50}` slices were seeded from exp10 N0 results of a
different reference model, so "high-confidence correct on the seed" did not
correspond to "easy for OLMo-3.1." Consequences:
- OLMo-3.1 N0 accuracy on top25/top50 dropped to ~0.19/0.23 — below
  reasonable floor and artificially inflating the W1 error rate.
- Only 21 (top25) or 56 (top50) clean N0-correct rows — too few to anchor
  a W1 compliance direction.

Switching to `all_prior_wrong` restored N0 accuracy to **0.47** and gave
**158** clean primary_w1 rows. A manual 30-row sample of N0 responses
confirmed that remaining errors are genuine model mistakes plus some
TruthfulQA-style dataset ambiguity, not parser or formatting bugs.
