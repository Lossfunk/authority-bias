# Exp14 Reasons Dataset Audit Report (Restored)

**Date**: February 17, 2026  
**Dataset**: `data/exp14_reasons.jsonl` (1,813 items)  
**Scope**: Restored automated contamination audit and robustness follow-up

---

## 1. Hedging/Meta Contamination Split

Using the restored heuristic split over `reasons_wrong` text:

| Category | Count | % |
|---|---:|---:|
| Strong hedging cues | 834 | 46.0% |
| Mild hedging only | 57 | 3.1% |
| Meta-explanatory framing only | 115 | 6.3% |
| Clean (none of the above) | 807 | 44.5% |

Artifacts:

- `new-phase-results/exp14/robustness/subset_summary.json`
- `new-phase-results/exp14/robustness/subsets/clean_uids.txt`
- `new-phase-results/exp14/robustness/subsets/hedged_uids.txt`

---

## 2. Robustness Re-Analysis (Full vs Clean vs Hedged)

Recomputed using `src.exp14.analyze_evidence_quality`:

- Full: `new-phase-results/exp14/robustness/full_analysis`
- Clean: `new-phase-results/exp14/robustness/clean_analysis`
- Hedged: `new-phase-results/exp14/robustness/hedged_analysis`

Prior-wrong slice trend summary (mean Kendall tau over model x tag x instruction cells):

| Subset | Mean tau_wrong | Mean tau_correct | Mean (correct - wrong) |
|---|---:|---:|---:|
| Full | 0.437 | 0.663 | 0.226 |
| Clean | 0.445 | 0.647 | 0.203 |
| Hedged | 0.412 | 0.672 | 0.259 |

Gap sign check (`correct > wrong`) across 12 prior-wrong cells:

- Full: 12/12
- Clean: 12/12
- Hedged: 11/12

Comparison artifact:

- `new-phase-results/exp14/robustness/comparison_prior_wrong_tau.csv`
- `new-phase-results/exp14/robustness/robustness_report.md`

---

## 3. Interpretation

1. The Exp14 evidence-quality asymmetry remains on the clean subset (does not collapse).
2. Hedged items appear to amplify effect magnitude, but are not necessary for direction.
3. The robustness check supports keeping the main Exp14 claim with an explicit contamination caveat.

---

## 4. Caveat

This restored audit is heuristic and automated. It is suitable as a robustness stress test but not a perfect semantic annotation of all wrong-reason rows.
