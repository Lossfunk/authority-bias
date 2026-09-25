# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/olmo2/gradient_uncertain/allenai__OLMo-2-0325-32B-Instruct_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=960, input rows=960)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 320 | 293 | 91.56% | 100 | 34.13% | 193 | 65.87% | 60.31% | 0 |
| `C1_note` | 320 | 281 | 87.81% | 189 | 67.26% | 92 | 32.74% | 28.75% | 0 |
| `W1_note` | 320 | 275 | 85.94% | 64 | 23.27% | 211 | 76.73% | 65.94% | 0 |
| `OVERALL` | 960 | 849 | 88.44% | 353 | 41.58% | 496 | 58.42% | 51.67% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **255**
- N0 correct: **85**, W1 flips to wrong: **36**
- Flip rate (W1 wrong | N0 correct): **42.35%**
- C1 correction rate (C1 correct | N0 wrong): **51.76%**

- CoT cbw_gen: 5/7 (71.43%)  |  cbw_prompt: 16/36 (44.44%)
- CoT reasoned_gen: 6/36 (16.67%)  |  reasoned_prompt: 2/36 (5.56%)

