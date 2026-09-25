# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/gpt-oss/gradient_weak/openai__gpt-oss-20b_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so, assistantfinalthe, assistantfinalno, assistantfinalthey, assistantfinalit, assistantfinalyou, assistantfinalnothing, assistantfinalyes, assistantfinalnone, assistantfinala, assistantfinalthere, assistantfinaljohn, assistantfinalsir
- Dedupe policy: `last` (duplicates=0, unique uid+condition=2430, input rows=2430)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 810 | 800 | 98.77% | 610 | 76.25% | 190 | 23.75% | 23.46% | 0 |
| `C1_note` | 810 | 802 | 99.01% | 735 | 91.65% | 67 | 8.35% | 8.27% | 0 |
| `W1_note` | 810 | 791 | 97.65% | 413 | 52.21% | 378 | 47.79% | 46.67% | 0 |
| `OVERALL` | 2430 | 2393 | 98.48% | 1758 | 73.46% | 635 | 26.54% | 26.13% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **781**
- N0 correct: **601**, W1 flips to wrong: **226**
- Flip rate (W1 wrong | N0 correct): **37.60%**
- C1 correction rate (C1 correct | N0 wrong): **77.78%**

- CoT cbw_gen: 191/203 (94.09%)  |  cbw_prompt: 216/226 (95.58%)
- CoT reasoned_gen: 193/226 (85.40%)  |  reasoned_prompt: 21/226 (9.29%)

