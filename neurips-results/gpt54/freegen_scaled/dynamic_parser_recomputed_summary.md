# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/gpt54/freegen_scaled/openai__gpt-5.4_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=2430, input rows=2430)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 810 | 798 | 98.52% | 735 | 92.11% | 63 | 7.89% | 7.78% | 0 |
| `C1_note` | 810 | 800 | 98.77% | 799 | 99.88% | 1 | 0.12% | 0.12% | 0 |
| `W1_note` | 810 | 798 | 98.52% | 404 | 50.63% | 394 | 49.37% | 48.64% | 0 |
| `OVERALL` | 2430 | 2396 | 98.60% | 1938 | 80.88% | 458 | 19.12% | 18.85% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **789**
- N0 correct: **727**, W1 flips to wrong: **325**
- Flip rate (W1 wrong | N0 correct): **44.70%**
- C1 correction rate (C1 correct | N0 wrong): **98.39%**

- CoT cbw_gen: 0/0 (NA)  |  cbw_prompt: 307/325 (94.46%)
- CoT reasoned_gen: 0/325 (0.00%)  |  reasoned_prompt: 27/325 (8.31%)

