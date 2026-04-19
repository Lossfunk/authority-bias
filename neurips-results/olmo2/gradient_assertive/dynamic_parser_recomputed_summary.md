# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/olmo2/gradient_assertive/allenai__OLMo-2-0325-32B-Instruct_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=960, input rows=960)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 320 | 294 | 91.88% | 102 | 34.69% | 192 | 65.31% | 60.00% | 0 |
| `C1_note` | 320 | 302 | 94.38% | 279 | 92.38% | 23 | 7.62% | 7.19% | 0 |
| `W1_note` | 320 | 290 | 90.62% | 25 | 8.62% | 265 | 91.38% | 82.81% | 0 |
| `OVERALL` | 960 | 886 | 92.29% | 406 | 45.82% | 480 | 54.18% | 50.00% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **272**
- N0 correct: **92**, W1 flips to wrong: **70**
- Flip rate (W1 wrong | N0 correct): **76.09%**
- C1 correction rate (C1 correct | N0 wrong): **87.78%**

- CoT cbw_gen: 0/1 (0.00%)  |  cbw_prompt: 29/70 (41.43%)
- CoT reasoned_gen: 1/70 (1.43%)  |  reasoned_prompt: 8/70 (11.43%)

