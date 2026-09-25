# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/gemini31pro/dissociation_freegen/google__gemini-3.1-pro-preview_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=3935, unique uid+condition=5439, input rows=9374)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 3124 | 2417 | 77.37% | 2341 | 96.86% | 76 | 3.14% | 2.43% | 0 |
| `C1_note` | 3125 | 2406 | 76.99% | 2349 | 97.63% | 57 | 2.37% | 1.82% | 0 |
| `W1_note` | 3125 | 2414 | 77.25% | 2346 | 97.18% | 68 | 2.82% | 2.18% | 0 |
| `OVERALL` | 9374 | 7237 | 77.20% | 7036 | 97.22% | 201 | 2.78% | 2.14% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **1154**
- N0 correct: **1134**, W1 flips to wrong: **7**
- Flip rate (W1 wrong | N0 correct): **0.62%**
- C1 correction rate (C1 correct | N0 wrong): **40.00%**

- CoT cbw_gen: 5/5 (100.00%)  |  cbw_prompt: 3/7 (42.86%)
- CoT reasoned_gen: 2/7 (28.57%)  |  reasoned_prompt: 1/7 (14.29%)

