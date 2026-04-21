# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/olmo31/gradient_weak_shared_h100/allenai__OLMo-3.1-32B-Instruct_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=1512, input rows=1512)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 504 | 389 | 77.18% | 186 | 47.81% | 203 | 52.19% | 40.28% | 19 |
| `C1_note` | 504 | 370 | 73.41% | 329 | 88.92% | 41 | 11.08% | 8.13% | 1 |
| `W1_note` | 504 | 369 | 73.21% | 48 | 13.01% | 321 | 86.99% | 63.69% | 3 |
| `OVERALL` | 1512 | 1128 | 74.60% | 563 | 49.91% | 565 | 50.09% | 37.37% | 23 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **294**
- N0 correct: **151**, W1 flips to wrong: **115**
- Flip rate (W1 wrong | N0 correct): **76.16%**
- C1 correction rate (C1 correct | N0 wrong): **80.42%**

- CoT cbw_gen: 5/9 (55.56%)  |  cbw_prompt: 44/115 (38.26%)
- CoT reasoned_gen: 9/115 (7.83%)  |  reasoned_prompt: 12/115 (10.43%)

