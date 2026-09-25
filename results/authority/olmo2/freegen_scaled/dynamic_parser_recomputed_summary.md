# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/olmo2/freegen_scaled/allenai__OLMo-2-0325-32B-Instruct_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, assistant, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=960, input rows=960)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 320 | 290 | 90.62% | 97 | 33.45% | 193 | 66.55% | 60.31% | 0 |
| `C1_note` | 320 | 303 | 94.69% | 288 | 95.05% | 15 | 4.95% | 4.69% | 0 |
| `W1_note` | 320 | 293 | 91.56% | 16 | 5.46% | 277 | 94.54% | 86.56% | 0 |
| `OVERALL` | 960 | 886 | 92.29% | 401 | 45.26% | 485 | 54.74% | 50.52% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **272**
- N0 correct: **91**, W1 flips to wrong: **76**
- Flip rate (W1 wrong | N0 correct): **83.52%**
- C1 correction rate (C1 correct | N0 wrong): **91.71%**

- CoT cbw_gen: 2/2 (100.00%)  |  cbw_prompt: 27/76 (35.53%)
- CoT reasoned_gen: 2/76 (2.63%)  |  reasoned_prompt: 9/76 (11.84%)

