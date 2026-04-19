# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/qwen35/sanity_remote_qwen128_parserfix_v3/Qwen__Qwen3.5-27B_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=30, input rows=30)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 10 | 10 | 100.00% | 7 | 70.00% | 3 | 30.00% | 30.00% | 10 |
| `C1_note` | 10 | 10 | 100.00% | 6 | 60.00% | 4 | 40.00% | 40.00% | 9 |
| `W1_note` | 10 | 10 | 100.00% | 5 | 50.00% | 5 | 50.00% | 50.00% | 10 |
| `OVERALL` | 30 | 30 | 100.00% | 18 | 60.00% | 12 | 40.00% | 40.00% | 29 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **10**
- N0 correct: **7**, W1 flips to wrong: **2**
- Flip rate (W1 wrong | N0 correct): **28.57%**
- C1 correction rate (C1 correct | N0 wrong): **33.33%**

- CoT cbw_gen: 2/2 (100.00%)  |  cbw_prompt: 2/2 (100.00%)
- CoT reasoned_gen: 1/2 (50.00%)  |  reasoned_prompt: 1/2 (50.00%)

