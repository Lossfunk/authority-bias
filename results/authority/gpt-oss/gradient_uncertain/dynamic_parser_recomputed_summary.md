# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/gpt-oss/gradient_uncertain/openai__gpt-oss-20b_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so, assistantfinalthe, assistantfinalno, assistantfinalit, assistantfinalthey, assistantfinalnone, assistantfinalnothing, assistantfinalyou, assistantfinalyes, assistantfinala, assistant, assistantfinalthere, assistantfinalsir
- Dedupe policy: `last` (duplicates=0, unique uid+condition=2430, input rows=2430)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 810 | 800 | 98.77% | 618 | 77.25% | 182 | 22.75% | 22.47% | 0 |
| `C1_note` | 810 | 803 | 99.14% | 657 | 81.82% | 146 | 18.18% | 18.02% | 0 |
| `W1_note` | 810 | 806 | 99.51% | 630 | 78.16% | 176 | 21.84% | 21.73% | 0 |
| `OVERALL` | 2430 | 2409 | 99.14% | 1905 | 79.08% | 504 | 20.92% | 20.74% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **797**
- N0 correct: **616**, W1 flips to wrong: **69**
- Flip rate (W1 wrong | N0 correct): **11.20%**
- C1 correction rate (C1 correct | N0 wrong): **51.38%**

- CoT cbw_gen: 60/66 (90.91%)  |  cbw_prompt: 64/69 (92.75%)
- CoT reasoned_gen: 52/69 (75.36%)  |  reasoned_prompt: 7/69 (10.14%)

