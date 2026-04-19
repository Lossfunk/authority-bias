# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/gpt-oss/v2_freegen/openai__gpt-oss-20b_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so, assistantfinalthe, assistantfinalthey, assistantfinalit, assistantfinalno, assistantfinalyes, assistantfinalyou, assistantfinalnone, assistantfinalnothing, assistantfinala, assistantfinalthere, assistantfinalsir, assistantfinaljohn, assistantfinalgeorge, assistantfinalhe
- Dedupe policy: `last` (duplicates=0, unique uid+condition=2430, input rows=2430)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 810 | 798 | 98.52% | 615 | 77.07% | 183 | 22.93% | 22.59% | 0 |
| `C1_note` | 810 | 804 | 99.26% | 761 | 94.65% | 43 | 5.35% | 5.31% | 0 |
| `W1_note` | 810 | 794 | 98.02% | 223 | 28.09% | 571 | 71.91% | 70.49% | 0 |
| `OVERALL` | 2430 | 2396 | 98.60% | 1599 | 66.74% | 797 | 33.26% | 32.80% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **791**
- N0 correct: **612**, W1 flips to wrong: **400**
- Flip rate (W1 wrong | N0 correct): **65.36%**
- C1 correction rate (C1 correct | N0 wrong): **86.59%**

- CoT cbw_gen: 334/356 (93.82%)  |  cbw_prompt: 379/400 (94.75%)
- CoT reasoned_gen: 336/400 (84.00%)  |  reasoned_prompt: 26/400 (6.50%)

