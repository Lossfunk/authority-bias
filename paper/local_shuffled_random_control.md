# Local CPU-only shuffled / random-vector control summary

Source artifacts (downloaded from the JarvisLabs notebook):

- `neurips-results/olmo2/mechanism/olmo2_compliance_analysis/null_controls.json`
- `neurips-results/olmo2/mechanism/olmo2_compliance_analysis/probe_results.json`
- `neurips-results/olmo31/mechanism/compliance_analysis_all_prior_wrong_shared_h100/null_controls.json`
- `neurips-results/olmo31/mechanism/compliance_analysis_all_prior_wrong_shared_h100/probe_results.json`

## How to read this

The real probe is the cross-validated compliance direction at each model's primary (layer, position).
Random-direction nulls draw 100 random unit vectors in the same residual-stream space and compute AUROC.
Shuffled-label nulls keep the real direction but shuffle W1/C1 labels 100 times.
Random-raw-projection nulls use the raw (un-trained) projection direction at the same pair.
Empirical p-values are two-sided over the null distribution; z is (real − null_mean) / null_std.

## Real probe AUROC vs null distributions

| Model | Condition | Real AUROC (95% CI) | Null mean ± std | z | p (two-sided) |
| --- | --- | --- | --- | --- | --- |
| OLMo-2-32B-Instruct | W1 — random direction | 0.700 [0.550, 0.834] | 0.492 ± 0.106 | +1.95 | 0.059 |
| OLMo-2-32B-Instruct | C1 — random direction | 0.704 [0.595, 0.815] | 0.510 ± 0.118 | +1.65 | 0.059 |
| OLMo-2-32B-Instruct | W1 — shuffled labels | 0.700 [0.550, 0.834] | 0.493 ± 0.103 | +2.00 | 0.050 |
| OLMo-2-32B-Instruct | C1 — shuffled labels | 0.704 [0.595, 0.815] | 0.482 ± 0.095 | +2.33 | 0.040 |
| OLMo-2-32B-Instruct | W1 — random raw projection | 0.700 [0.550, 0.834] | 0.575 ± 0.052 | +2.39 | 0.040 |
| OLMo-2-32B-Instruct | C1 — random raw projection | 0.704 [0.595, 0.815] | 0.585 ± 0.067 | +1.79 | 0.069 |
| OLMo-3.1-32B-Instruct | W1 — random direction | 0.649 [0.538, 0.755] | 0.496 ± 0.081 | +1.88 | 0.040 |
| OLMo-3.1-32B-Instruct | C1 — random direction | 0.503 [0.382, 0.600] | 0.534 ± 0.093 | -0.33 | 1.000 |
| OLMo-3.1-32B-Instruct | W1 — shuffled labels | 0.649 [0.538, 0.755] | 0.479 ± 0.081 | +2.12 | 0.079 |
| OLMo-3.1-32B-Instruct | C1 — shuffled labels | 0.503 [0.382, 0.600] | 0.483 ± 0.083 | +0.24 | 1.000 |
| OLMo-3.1-32B-Instruct | W1 — random raw projection | 0.649 [0.538, 0.755] | 0.559 ± 0.039 | +2.33 | 0.020 |
| OLMo-3.1-32B-Instruct | C1 — random raw projection | 0.503 [0.382, 0.600] | 0.583 ± 0.053 | -1.50 | 0.970 |

## Headline takeaways

- **OLMo-2-32B-Instruct** (W1 primary AUROC = 0.700; C1 primary AUROC = 0.704). Random-direction nulls put W1 at z=+1.95 (p=0.059), C1 at z=+1.65 (p=0.059); shuffled-label nulls put W1 at z=+2.00 (p=0.050), C1 at z=+2.33 (p=0.040).
- **OLMo-3.1-32B-Instruct** (W1 primary AUROC = 0.649; C1 primary AUROC = 0.503). Random-direction nulls put W1 at z=+1.88 (p=0.040), C1 at z=-0.33 (p=1.000); shuffled-label nulls put W1 at z=+2.12 (p=0.079), C1 at z=+0.24 (p=1.000).

## Suggested wording for paper appendix

We re-used the random-direction and shuffled-label controls computed alongside the compliance direction. For OLMo-2-32B-Instruct, the trained compliance direction beats both nulls on W1 (z≈+2.0 vs random unit vectors and shuffled labels; p≈0.04–0.06) and on C1 (z≈+1.6–2.3; p≈0.04–0.06). For OLMo-3.1-32B-Instruct the picture is asymmetric: W1 sits ~1.5–2σ above the random-direction and shuffled-label nulls (p≈0.02–0.08), while C1 is statistically indistinguishable from random unit vectors and from shuffled labels (z≈0; p≈1.0). This pattern is consistent with the paper's interpretation that the authority-level signal carries most of the variance the probe captures and that a separable 'compliance' direction at the chosen layer/position is not robustly identifiable for OLMo-3.1 C1, which we now report explicitly in the main appendix alongside the random-direction nulls.
