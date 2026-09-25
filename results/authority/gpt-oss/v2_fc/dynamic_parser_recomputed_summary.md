# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/gpt-oss/v2_fc/openai__gpt-oss-20b_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=2430, input rows=2430)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 810 | 22 | 2.72% | 15 | 68.18% | 7 | 31.82% | 0.86% | 0 |
| `C1_note` | 810 | 4 | 0.49% | 3 | 75.00% | 1 | 25.00% | 0.12% | 0 |
| `W1_note` | 810 | 4 | 0.49% | 3 | 75.00% | 1 | 25.00% | 0.12% | 0 |
| `OVERALL` | 2430 | 30 | 1.23% | 21 | 70.00% | 9 | 30.00% | 0.37% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **1**
- N0 correct: **0**, W1 flips to wrong: **0**
- Flip rate (W1 wrong | N0 correct): **NA**
- C1 correction rate (C1 correct | N0 wrong): **0.00%**

- CoT cbw_gen: 0/0 (NA)  |  cbw_prompt: 0/0 (NA)
- CoT reasoned_gen: 0/0 (NA)  |  reasoned_prompt: 0/0 (NA)

