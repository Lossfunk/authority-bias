# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/qwen35/authoritative_verified_shared_h100/Qwen__Qwen3.5-27B_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=5439, input rows=5439)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 1813 | 1812 | 99.94% | 1358 | 74.94% | 454 | 25.06% | 25.04% | 0 |
| `C1_note` | 1813 | 1809 | 99.78% | 1220 | 67.44% | 589 | 32.56% | 32.49% | 0 |
| `W1_note` | 1813 | 1811 | 99.89% | 900 | 49.70% | 911 | 50.30% | 50.25% | 0 |
| `OVERALL` | 5439 | 5432 | 99.87% | 3478 | 64.03% | 1954 | 35.97% | 35.93% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **1808**
- N0 correct: **1355**, W1 flips to wrong: **608**
- Flip rate (W1 wrong | N0 correct): **44.87%**
- C1 correction rate (C1 correct | N0 wrong): **45.03%**

- CoT cbw_gen: 347/607 (57.17%)  |  cbw_prompt: 354/608 (58.22%)
- CoT reasoned_gen: 519/608 (85.36%)  |  reasoned_prompt: 33/608 (5.43%)

