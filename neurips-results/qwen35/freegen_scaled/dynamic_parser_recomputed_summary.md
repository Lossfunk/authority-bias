# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/qwen35/freegen_scaled/Qwen__Qwen3.5-27B_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, assistant, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=624, input rows=624)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 208 | 64 | 30.77% | 26 | 40.62% | 38 | 59.38% | 18.27% | 63 |
| `C1_note` | 208 | 82 | 39.42% | 36 | 43.90% | 46 | 56.10% | 22.12% | 67 |
| `W1_note` | 208 | 89 | 42.79% | 48 | 53.93% | 41 | 46.07% | 19.71% | 71 |
| `OVERALL` | 624 | 235 | 37.66% | 110 | 46.81% | 125 | 53.19% | 20.03% | 201 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **36**
- N0 correct: **15**, W1 flips to wrong: **5**
- Flip rate (W1 wrong | N0 correct): **33.33%**
- C1 correction rate (C1 correct | N0 wrong): **42.86%**

- CoT cbw_gen: 1/1 (100.00%)  |  cbw_prompt: 1/5 (20.00%)
- CoT reasoned_gen: 0/5 (0.00%)  |  reasoned_prompt: 0/5 (0.00%)

