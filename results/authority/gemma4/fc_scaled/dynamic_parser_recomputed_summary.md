# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/gemma4/fc_scaled/google__gemma-4-26B-A4B-it_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=2661, input rows=2661)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 887 | 29 | 3.27% | 20 | 68.97% | 9 | 31.03% | 1.01% | 0 |
| `C1_note` | 887 | 8 | 0.90% | 7 | 87.50% | 1 | 12.50% | 0.11% | 0 |
| `W1_note` | 887 | 39 | 4.40% | 17 | 43.59% | 22 | 56.41% | 2.48% | 0 |
| `OVERALL` | 2661 | 76 | 2.86% | 44 | 57.89% | 32 | 42.11% | 1.20% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **1**
- N0 correct: **1**, W1 flips to wrong: **0**
- Flip rate (W1 wrong | N0 correct): **0.00%**
- C1 correction rate (C1 correct | N0 wrong): **NA**

- CoT cbw_gen: 0/0 (NA)  |  cbw_prompt: 0/0 (NA)
- CoT reasoned_gen: 0/0 (NA)  |  reasoned_prompt: 0/0 (NA)

