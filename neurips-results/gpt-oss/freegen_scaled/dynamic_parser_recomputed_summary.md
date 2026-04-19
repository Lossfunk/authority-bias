# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/gpt-oss/freegen_scaled/openai__gpt-oss-20b_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so, assistantanalysiswe, assistantanalysisthe, assistant, assistantfinalthe, assistantfinalthey, assistantfinalno, assistantfinalyes
- Dedupe policy: `last` (duplicates=0, unique uid+condition=2430, input rows=2430)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 810 | 732 | 90.37% | 449 | 61.34% | 283 | 38.66% | 34.94% | 0 |
| `C1_note` | 810 | 758 | 93.58% | 600 | 79.16% | 158 | 20.84% | 19.51% | 0 |
| `W1_note` | 810 | 725 | 89.51% | 49 | 6.76% | 676 | 93.24% | 83.46% | 0 |
| `OVERALL` | 2430 | 2215 | 91.15% | 1098 | 49.57% | 1117 | 50.43% | 45.97% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **638**
- N0 correct: **381**, W1 flips to wrong: **339**
- Flip rate (W1 wrong | N0 correct): **88.98%**
- C1 correction rate (C1 correct | N0 wrong): **72.37%**

- CoT cbw_gen: 313/328 (95.43%)  |  cbw_prompt: 322/339 (94.99%)
- CoT reasoned_gen: 214/339 (63.13%)  |  reasoned_prompt: 13/339 (3.83%)

