# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/olmo2/v2_freegen/allenai__OLMo-2-0325-32B-Instruct_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=960, input rows=960)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 320 | 294 | 91.88% | 100 | 34.01% | 194 | 65.99% | 60.62% | 0 |
| `C1_note` | 320 | 301 | 94.06% | 285 | 94.68% | 16 | 5.32% | 5.00% | 0 |
| `W1_note` | 320 | 293 | 91.56% | 16 | 5.46% | 277 | 94.54% | 86.56% | 0 |
| `OVERALL` | 960 | 888 | 92.50% | 401 | 45.16% | 487 | 54.84% | 50.73% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **275**
- N0 correct: **93**, W1 flips to wrong: **77**
- Flip rate (W1 wrong | N0 correct): **82.80%**
- C1 correction rate (C1 correct | N0 wrong): **91.76%**

- CoT cbw_gen: 1/1 (100.00%)  |  cbw_prompt: 27/77 (35.06%)
- CoT reasoned_gen: 1/77 (1.30%)  |  reasoned_prompt: 9/77 (11.69%)

