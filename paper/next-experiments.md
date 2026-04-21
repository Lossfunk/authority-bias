# Next Experiments — Strengthening Authority-Compliance Paper

These are the experiments that most directly answer the likely NeurIPS reviews (see `mock-review.md`).

## Full List

### 1. Per-model completeness table

Not glamorous, but high value. Add a table with one row per model and columns for:

`behavior`, `steering`, `C1-null`, `shuffled-label`, `assistant-axis`, `valence/arousal`, `PIQA`, `SYCON`, `thinking-mode`.

This directly addresses overclaim and evidence asymmetry.

### 2. Finish missing deconfounds

Run the missing assistant-axis checks, especially for `Qwen3.5` and `OLMo-3.1`.

Also report explicit valence/arousal cosine values per model, not just a family-wide statement.

### 3. Config-freezing audit

Run transfer and `SYCON` with the exact frozen config chosen on the fit split and report those as primary.

If you also want best-case exploratory sweeps, put them in appendix and label them clearly.

This single fix would remove a major reviewer attack.

### 4. Seed / resample stability

Refit the direction across multiple random fit/eval splits or bootstrap subsets and report variance in:

`AUROC`, matched flip, assistant-axis cosine, transfer score.

This directly addresses "is the direction real or a fragile fitting artifact?"

### 5. Replication of thinking-mode result

Replicate the thinking-on vs thinking-off comparison on at least one additional reasoning-capable model.

Even one more matched result would materially strengthen that claim.

### 6. `OLMo-3.1` PIQA transfer

Either run it or explicitly explain why it is excluded.

Right now the omission is too visible.

### 7. Negative steering / mitigation check

Test whether steering in the opposite direction reduces authority-compliance on held-out prompts without severely harming baseline helpfulness or normal instruction following.

This would strengthen both significance and practical relevance.

### 8. Position / token-phase specificity

Show whether the effect depends on applying the patch at authority-note positions versus later prompt tokens.

That would help support the "upstream processing" claim more directly.

### 9. Direction specificity against nearby alternatives

Add one or two extra deconfounds:

- instruction-following strength,
- general refusal tendency,
- or confidence/hedging style.

Reviewers will otherwise say persona and valence are not the only plausible alternatives.

### 10. Circuit follow-up, if feasible

Even a lightweight head-level attribution or patching analysis in one strong model would substantially upgrade the "mechanism" claim.

Not mandatory, but high upside.

---

## Highest-ROI Shortlist

If time is limited, do these first:

1. Finish missing assistant-axis and valence/arousal checks.
2. Clean up frozen-config discipline and rerun/report primary `PIQA`/`SYCON` under that protocol.
3. Add seed/resample stability.
4. Replicate thinking-mode on one more model.
5. Add a per-model audit table.

Those five would directly neutralize most of the review.
