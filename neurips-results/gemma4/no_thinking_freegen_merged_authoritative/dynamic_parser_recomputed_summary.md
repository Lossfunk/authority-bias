# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/gemma4/no_thinking_freegen_merged_authoritative/google__gemma-4-26B-A4B-it_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=2661, input rows=2661)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 887 | 862 | 97.18% | 776 | 90.02% | 86 | 9.98% | 9.70% | 0 |
| `C1_note` | 887 | 866 | 97.63% | 859 | 99.19% | 7 | 0.81% | 0.79% | 0 |
| `W1_note` | 887 | 867 | 97.75% | 288 | 33.22% | 579 | 66.78% | 65.28% | 0 |
| `OVERALL` | 2661 | 2595 | 97.52% | 1923 | 74.10% | 672 | 25.90% | 25.25% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **842**
- N0 correct: **757**, W1 flips to wrong: **476**
- Flip rate (W1 wrong | N0 correct): **62.88%**
- C1 correction rate (C1 correct | N0 wrong): **96.47%**

- CoT cbw_gen: 26/91 (28.57%)  |  cbw_prompt: 10/476 (2.10%)
- CoT reasoned_gen: 81/476 (17.02%)  |  reasoned_prompt: 11/476 (2.31%)

