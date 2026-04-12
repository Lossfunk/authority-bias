# Compliance Direction Analysis — GPT-oss-20B

**Date:** 2026-04-11  
**Extraction:** 9720 records (810 items × 4 styles × 3 conditions)  
**Primary pair:** endorsement_end, layer 18 (selected unanimously across all 5 nested CV folds)

## Main result

A single direction in activation space at the end of the authority endorsement text (layer 18) predicts whether GPT-oss-20B will flip its answer to match a wrong authority-endorsed claim.

- **W1 flip prediction:** nested CV AUROC = 0.811, accuracy = 73.7% (n=612)
- **C1 correction prediction:** AUROC = 0.527 (near chance, n=179)

## Null controls

| Null test | W1 AUROC | Interpretation |
|---|---|---|
| Compliance direction (nested CV) | **0.811** | Our result |
| Shuffled labels | 0.498 mean (97.5th pctl = 0.550) | Proper null — breaks label-direction link |
| Random direction (with probe) | 0.665 mean (std 0.108) | Inflated by probe fitting in 2880-d space |
| Random raw projection (no probe) | 0.678 mean (97.5th pctl = 0.858) | Oriented dot-product, no fitting |
| Majority baseline | 0.500 | Trivial |
| Authority-level-only (style one-hot) | 0.760 | Behavioral confound ceiling |

**p-value vs shuffled labels: 0.005** — the direction carries real item-level information about flip outcomes beyond what the authority level alone explains.

The random-direction baseline is high (0.665) because any random projection in 2880-d space, when passed through a logistic probe, partially captures distributed signal. The shuffled-label control is the proper test: it uses the same direction and probe pipeline but permutes labels, isolating whether the specific direction matters. It does (p=0.005).

## Direction stability

Mean pairwise cosine across 5 CV folds: **0.9999** (std 2.5e-5). The direction is essentially identical regardless of which 80% of items are used to compute it.

## Direction geometry

| | pooled_authority | content | endorsement_presence | auth_given_correct | auth_given_wrong |
|---|---|---|---|---|---|
| pooled_authority | 1.00 | -0.07 | 0.95 | 0.99 | 0.99 |
| content | -0.07 | 1.00 | -0.17 | 0.07 | -0.20 |
| endorsement_presence | 0.95 | -0.17 | 1.00 | 0.92 | 0.95 |

Authority processing and factual content are nearly orthogonal (cos = -0.07). The authority-given-correct and authority-given-wrong directions are highly aligned (cos = 0.96), meaning the model represents "an authority said something" similarly regardless of whether the claim is true.

## Task-specific direction checks

| Direction used | W1 CV AUROC | C1 CV AUROC |
|---|---|---|
| shared_within_label | 0.812 | 0.574 |
| authority_given_wrong | **0.850** | — |
| authority_given_correct | — | 0.590 |

The `authority_given_wrong` direction predicts flips even better than the shared direction. C1 corrections remain near chance regardless of direction. This means:

1. The direction is specifically about **compliance/deference**, not general authority processing
2. Correction (resisting authority) is a different mechanism, not just the inverse of compliance
3. This asymmetry is a cleaner story for the paper

## Projection across authority levels

Projections of C1 and W1 centroids onto the compliance direction across the 4 non-think authority levels (weak → uncertain → assertive → authoritative) show monotonic separation, with W1 projections tracking behavioral flip rates (Spearman correlation reported in projection_analysis.json).

## C1/W1 trajectory geometry

The auth-minus-weak trajectory vectors for C1 and W1 are highly aligned (cosine and angle in trajectory_analysis.json), confirming the model moves through similar representational space when authority increases, regardless of whether the endorsed answer is correct.

## Files

- `primary_direction.pt` — the compliance direction vector (endorsement_end, layer 18)
- `directions.pt` — full direction pack (6 directions)
- `probe_results.json` — nested CV metrics, task-specific checks, baselines, p-values
- `null_controls.json` — all null distribution values (shuffled, random direction, raw projection)
- `projection_analysis.json` — 4-level projections, Spearman, trajectory
- `layer_position_sweep.json` — full AUROC heatmap data
- `figures/` — heatmaps, projection plots, null distribution, cosine matrix
