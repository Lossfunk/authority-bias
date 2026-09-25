# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/gemma4/gradient_assertive/google__gemma-4-26B-A4B-it_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so, <|thought|>
- Dedupe policy: `last` (duplicates=0, unique uid+condition=2661, input rows=2661)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 887 | 861 | 97.07% | 777 | 90.24% | 84 | 9.76% | 9.47% | 0 |
| `C1_note` | 887 | 861 | 97.07% | 827 | 96.05% | 34 | 3.95% | 3.83% | 0 |
| `W1_note` | 887 | 858 | 96.73% | 680 | 79.25% | 178 | 20.75% | 20.07% | 0 |
| `OVERALL` | 2661 | 2580 | 96.96% | 2284 | 88.53% | 296 | 11.47% | 11.12% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **838**
- N0 correct: **760**, W1 flips to wrong: **100**
- Flip rate (W1 wrong | N0 correct): **13.16%**
- C1 correction rate (C1 correct | N0 wrong): **71.79%**

- CoT cbw_gen: 17/21 (80.95%)  |  cbw_prompt: 3/100 (3.00%)
- CoT reasoned_gen: 16/100 (16.00%)  |  reasoned_prompt: 3/100 (3.00%)

