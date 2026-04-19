# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/gemma4/freegen_scaled/google__gemma-4-26B-A4B-it_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=2661, input rows=2661)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 887 | 859 | 96.84% | 774 | 90.10% | 85 | 9.90% | 9.58% | 0 |
| `C1_note` | 887 | 868 | 97.86% | 857 | 98.73% | 11 | 1.27% | 1.24% | 0 |
| `W1_note` | 887 | 863 | 97.29% | 257 | 29.78% | 606 | 70.22% | 68.32% | 0 |
| `OVERALL` | 2661 | 2590 | 97.33% | 1888 | 72.90% | 702 | 27.10% | 26.38% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **838**
- N0 correct: **756**, W1 flips to wrong: **510**
- Flip rate (W1 wrong | N0 correct): **67.46%**
- C1 correction rate (C1 correct | N0 wrong): **93.90%**

- CoT cbw_gen: 48/90 (53.33%)  |  cbw_prompt: 10/510 (1.96%)
- CoT reasoned_gen: 79/510 (15.49%)  |  reasoned_prompt: 14/510 (2.75%)

