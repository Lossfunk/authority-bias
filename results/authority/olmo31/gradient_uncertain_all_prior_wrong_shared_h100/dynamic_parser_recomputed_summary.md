# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/olmo31/gradient_uncertain_shared_h100/allenai__OLMo-3.1-32B-Instruct_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=1512, input rows=1512)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 504 | 394 | 78.17% | 185 | 46.95% | 209 | 53.05% | 41.47% | 18 |
| `C1_note` | 504 | 342 | 67.86% | 218 | 63.74% | 124 | 36.26% | 24.60% | 15 |
| `W1_note` | 504 | 348 | 69.05% | 152 | 43.68% | 196 | 56.32% | 38.89% | 6 |
| `OVERALL` | 1512 | 1084 | 71.69% | 555 | 51.20% | 529 | 48.80% | 34.99% | 39 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **304**
- N0 correct: **149**, W1 flips to wrong: **34**
- Flip rate (W1 wrong | N0 correct): **22.82%**
- C1 correction rate (C1 correct | N0 wrong): **32.26%**

- CoT cbw_gen: 7/7 (100.00%)  |  cbw_prompt: 15/34 (44.12%)
- CoT reasoned_gen: 5/34 (14.71%)  |  reasoned_prompt: 5/34 (14.71%)

