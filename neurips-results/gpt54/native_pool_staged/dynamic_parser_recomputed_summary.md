# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/gpt54/native_pool_staged/openai__gpt-5.4_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=5101, input rows=5101)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 1813 | 1784 | 98.40% | 1639 | 91.87% | 145 | 8.13% | 8.00% | 0 |
| `C1_note` | 1644 | 1636 | 99.51% | 1635 | 99.94% | 1 | 0.06% | 0.06% | 0 |
| `W1_note` | 1644 | 1637 | 99.57% | 939 | 57.36% | 698 | 42.64% | 42.46% | 0 |
| `OVERALL` | 5101 | 5057 | 99.14% | 4213 | 83.31% | 844 | 16.69% | 16.55% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **1626**
- N0 correct: **1626**, W1 flips to wrong: **693**
- Flip rate (W1 wrong | N0 correct): **42.62%**
- C1 correction rate (C1 correct | N0 wrong): **NA**

- CoT cbw_gen: 0/0 (NA)  |  cbw_prompt: 368/693 (53.10%)
- CoT reasoned_gen: 0/693 (0.00%)  |  reasoned_prompt: 45/693 (6.49%)

