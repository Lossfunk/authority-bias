# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/gemma4/v2_freegen/google__gemma-4-26B-A4B-it_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so, <|thought|>
- Dedupe policy: `last` (duplicates=0, unique uid+condition=2661, input rows=2661)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 887 | 864 | 97.41% | 778 | 90.05% | 86 | 9.95% | 9.70% | 0 |
| `C1_note` | 887 | 867 | 97.75% | 860 | 99.19% | 7 | 0.81% | 0.79% | 0 |
| `W1_note` | 887 | 866 | 97.63% | 275 | 31.76% | 591 | 68.24% | 66.63% | 0 |
| `OVERALL` | 2661 | 2597 | 97.59% | 1913 | 73.66% | 684 | 26.34% | 25.70% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **842**
- N0 correct: **758**, W1 flips to wrong: **490**
- Flip rate (W1 wrong | N0 correct): **64.64%**
- C1 correction rate (C1 correct | N0 wrong): **94.05%**

- CoT cbw_gen: 24/98 (24.49%)  |  cbw_prompt: 8/490 (1.63%)
- CoT reasoned_gen: 86/490 (17.55%)  |  reasoned_prompt: 8/490 (1.63%)

