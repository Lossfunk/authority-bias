# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/grok/grok420_dissociation_freegen/x-ai__grok-4.20_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=5439, input rows=5439)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 1813 | 1712 | 94.43% | 1552 | 90.65% | 160 | 9.35% | 8.83% | 0 |
| `C1_note` | 1813 | 1717 | 94.70% | 1697 | 98.84% | 20 | 1.16% | 1.10% | 0 |
| `W1_note` | 1813 | 1688 | 93.11% | 187 | 11.08% | 1501 | 88.92% | 82.79% | 0 |
| `OVERALL` | 5439 | 5117 | 94.08% | 3436 | 67.15% | 1681 | 32.85% | 30.91% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **1608**
- N0 correct: **1462**, W1 flips to wrong: **1279**
- Flip rate (W1 wrong | N0 correct): **87.48%**
- C1 correction rate (C1 correct | N0 wrong): **89.73%**

- CoT cbw_gen: 8/8 (100.00%)  |  cbw_prompt: 700/1279 (54.73%)
- CoT reasoned_gen: 2/1279 (0.16%)  |  reasoned_prompt: 61/1279 (4.77%)

