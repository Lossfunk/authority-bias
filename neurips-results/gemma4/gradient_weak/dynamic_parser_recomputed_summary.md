# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/gemma4/gradient_weak/google__gemma-4-26B-A4B-it_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so, <|thought|>
- Dedupe policy: `last` (duplicates=0, unique uid+condition=2661, input rows=2661)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 887 | 864 | 97.41% | 778 | 90.05% | 86 | 9.95% | 9.70% | 0 |
| `C1_note` | 887 | 861 | 97.07% | 847 | 98.37% | 14 | 1.63% | 1.58% | 0 |
| `W1_note` | 887 | 866 | 97.63% | 293 | 33.83% | 573 | 66.17% | 64.60% | 0 |
| `OVERALL` | 2661 | 2591 | 97.37% | 1918 | 74.03% | 673 | 25.97% | 25.29% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **839**
- N0 correct: **757**, W1 flips to wrong: **470**
- Flip rate (W1 wrong | N0 correct): **62.09%**
- C1 correction rate (C1 correct | N0 wrong): **90.24%**

- CoT cbw_gen: 20/46 (43.48%)  |  cbw_prompt: 9/470 (1.91%)
- CoT reasoned_gen: 41/470 (8.72%)  |  reasoned_prompt: 6/470 (1.28%)

