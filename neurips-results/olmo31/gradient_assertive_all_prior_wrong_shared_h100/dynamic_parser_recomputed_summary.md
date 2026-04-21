# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/olmo31/gradient_assertive_shared_h100/allenai__OLMo-3.1-32B-Instruct_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=1512, input rows=1512)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 504 | 390 | 77.38% | 183 | 46.92% | 207 | 53.08% | 41.07% | 20 |
| `C1_note` | 504 | 351 | 69.64% | 287 | 81.77% | 64 | 18.23% | 12.70% | 8 |
| `W1_note` | 504 | 353 | 70.04% | 83 | 23.51% | 270 | 76.49% | 53.57% | 4 |
| `OVERALL` | 1512 | 1094 | 72.35% | 553 | 50.55% | 541 | 49.45% | 35.78% | 32 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **290**
- N0 correct: **148**, W1 flips to wrong: **82**
- Flip rate (W1 wrong | N0 correct): **55.41%**
- C1 correction rate (C1 correct | N0 wrong): **63.38%**

- CoT cbw_gen: 4/5 (80.00%)  |  cbw_prompt: 34/82 (41.46%)
- CoT reasoned_gen: 5/82 (6.10%)  |  reasoned_prompt: 8/82 (9.76%)

