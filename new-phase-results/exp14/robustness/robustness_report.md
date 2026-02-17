# Exp14 Hedging Robustness Check (Full vs Clean vs Hedged)

## Subset Construction
- Clean UID count: 807 (44.5%)
- Hedged UID count: 1006 (55.5%)
- Heuristic details: `subset_summary.json`

## Prior-Wrong Trend Comparison (Kendall tau)

| Subset | Mean tau_wrong | Mean tau_correct | Mean (correct - wrong) |
|---|---:|---:|---:|
| full | 0.437 | 0.663 | 0.226 |
| clean | 0.445 | 0.647 | 0.203 |
| hedged | 0.412 | 0.672 | 0.259 |

## Gap Sign Count (prior_wrong)

| Subset | #cells where correct>wrong | #cells where correct<=wrong |
|---|---:|---:|
| full | 12 | 0 |
| clean | 12 | 0 |
| hedged | 11 | 1 |
