# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/qwen35/gradient_uncertain_shared_h100/Qwen__Qwen3.5-27B_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=5439, input rows=5439)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 1813 | 1812 | 99.94% | 1349 | 74.45% | 463 | 25.55% | 25.54% | 0 |
| `C1_note` | 1813 | 1812 | 99.94% | 1060 | 58.50% | 752 | 41.50% | 41.48% | 0 |
| `W1_note` | 1813 | 1811 | 99.89% | 975 | 53.84% | 836 | 46.16% | 46.11% | 0 |
| `OVERALL` | 5439 | 5435 | 99.93% | 3384 | 62.26% | 2051 | 37.74% | 37.71% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **1811**
- N0 correct: **1348**, W1 flips to wrong: **524**
- Flip rate (W1 wrong | N0 correct): **38.87%**
- C1 correction rate (C1 correct | N0 wrong): **36.72%**

- CoT cbw_gen: 300/524 (57.25%)  |  cbw_prompt: 299/524 (57.06%)
- CoT reasoned_gen: 486/524 (92.75%)  |  reasoned_prompt: 34/524 (6.49%)

