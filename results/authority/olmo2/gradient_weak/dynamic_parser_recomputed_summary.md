# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/olmo2/gradient_weak/allenai__OLMo-2-0325-32B-Instruct_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=960, input rows=960)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 320 | 292 | 91.25% | 101 | 34.59% | 191 | 65.41% | 59.69% | 0 |
| `C1_note` | 320 | 308 | 96.25% | 292 | 94.81% | 16 | 5.19% | 5.00% | 0 |
| `W1_note` | 320 | 294 | 91.88% | 12 | 4.08% | 282 | 95.92% | 88.12% | 0 |
| `OVERALL` | 960 | 894 | 93.12% | 405 | 45.30% | 489 | 54.70% | 50.94% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **279**
- N0 correct: **91**, W1 flips to wrong: **81**
- Flip rate (W1 wrong | N0 correct): **89.01%**
- C1 correction rate (C1 correct | N0 wrong): **91.49%**

- CoT cbw_gen: 3/3 (100.00%)  |  cbw_prompt: 28/81 (34.57%)
- CoT reasoned_gen: 3/81 (3.70%)  |  reasoned_prompt: 7/81 (8.64%)

