# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/gemma4/gradient_uncertain/google__gemma-4-26B-A4B-it_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so, <|thought|>
- Dedupe policy: `last` (duplicates=0, unique uid+condition=2661, input rows=2661)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 887 | 862 | 97.18% | 777 | 90.14% | 85 | 9.86% | 9.58% | 0 |
| `C1_note` | 887 | 858 | 96.73% | 806 | 93.94% | 52 | 6.06% | 5.86% | 0 |
| `W1_note` | 887 | 857 | 96.62% | 744 | 86.81% | 113 | 13.19% | 12.74% | 0 |
| `OVERALL` | 2661 | 2577 | 96.84% | 2327 | 90.30% | 250 | 9.70% | 9.39% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **842**
- N0 correct: **761**, W1 flips to wrong: **44**
- Flip rate (W1 wrong | N0 correct): **5.78%**
- C1 correction rate (C1 correct | N0 wrong): **51.85%**

- CoT cbw_gen: 8/12 (66.67%)  |  cbw_prompt: 0/44 (0.00%)
- CoT reasoned_gen: 6/44 (13.64%)  |  reasoned_prompt: 3/44 (6.82%)

