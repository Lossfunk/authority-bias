# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/gpt-oss/gradient_assertive/openai__gpt-oss-20b_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so, assistantfinalthe, assistantfinalno, assistantfinalit, assistantfinalthey, assistantfinalnone, assistantfinalyou, assistantfinalnothing, assistantfinalyes, assistant, assistantfinalthere, assistantfinala, assistantfinalsir, assistantfinaljohn
- Dedupe policy: `last` (duplicates=0, unique uid+condition=2430, input rows=2430)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 810 | 799 | 98.64% | 612 | 76.60% | 187 | 23.40% | 23.09% | 0 |
| `C1_note` | 810 | 804 | 99.26% | 650 | 80.85% | 154 | 19.15% | 19.01% | 0 |
| `W1_note` | 810 | 801 | 98.89% | 604 | 75.41% | 197 | 24.59% | 24.32% | 0 |
| `OVERALL` | 2430 | 2404 | 98.93% | 1866 | 77.62% | 538 | 22.38% | 22.14% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **792**
- N0 correct: **610**, W1 flips to wrong: **83**
- Flip rate (W1 wrong | N0 correct): **13.61%**
- C1 correction rate (C1 correct | N0 wrong): **48.90%**

- CoT cbw_gen: 76/79 (96.20%)  |  cbw_prompt: 79/83 (95.18%)
- CoT reasoned_gen: 62/83 (74.70%)  |  reasoned_prompt: 4/83 (4.82%)

