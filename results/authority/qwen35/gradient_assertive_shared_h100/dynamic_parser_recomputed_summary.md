# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/qwen35/gradient_assertive_shared_h100/Qwen__Qwen3.5-27B_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=5439, input rows=5439)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 1813 | 1811 | 99.89% | 1367 | 75.48% | 444 | 24.52% | 24.49% | 0 |
| `C1_note` | 1813 | 1811 | 99.89% | 1130 | 62.40% | 681 | 37.60% | 37.56% | 0 |
| `W1_note` | 1813 | 1812 | 99.94% | 949 | 52.37% | 863 | 47.63% | 47.60% | 0 |
| `OVERALL` | 5439 | 5434 | 99.91% | 3446 | 63.42% | 1988 | 36.58% | 36.55% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **1810**
- N0 correct: **1366**, W1 flips to wrong: **567**
- Flip rate (W1 wrong | N0 correct): **41.51%**
- C1 correction rate (C1 correct | N0 wrong): **38.74%**

- CoT cbw_gen: 303/565 (53.63%)  |  cbw_prompt: 318/567 (56.08%)
- CoT reasoned_gen: 428/567 (75.49%)  |  reasoned_prompt: 36/567 (6.35%)

