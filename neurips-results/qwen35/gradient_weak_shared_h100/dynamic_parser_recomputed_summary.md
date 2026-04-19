# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/qwen35/gradient_weak_shared_h100/Qwen__Qwen3.5-27B_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=5439, input rows=5439)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 1813 | 1812 | 99.94% | 1361 | 75.11% | 451 | 24.89% | 24.88% | 0 |
| `C1_note` | 1813 | 1812 | 99.94% | 1232 | 67.99% | 580 | 32.01% | 31.99% | 0 |
| `W1_note` | 1813 | 1811 | 99.89% | 902 | 49.81% | 909 | 50.19% | 50.14% | 0 |
| `OVERALL` | 5439 | 5435 | 99.93% | 3495 | 64.31% | 1940 | 35.69% | 35.67% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **1810**
- N0 correct: **1359**, W1 flips to wrong: **609**
- Flip rate (W1 wrong | N0 correct): **44.81%**
- C1 correction rate (C1 correct | N0 wrong): **41.69%**

- CoT cbw_gen: 319/607 (52.55%)  |  cbw_prompt: 325/609 (53.37%)
- CoT reasoned_gen: 496/609 (81.44%)  |  reasoned_prompt: 35/609 (5.75%)

