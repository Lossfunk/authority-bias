# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/qwen35/sanity_remote_qwen320_parserfix/Qwen__Qwen3.5-27B_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=15, input rows=15)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 5 | 5 | 100.00% | 1 | 20.00% | 4 | 80.00% | 80.00% | 0 |
| `C1_note` | 5 | 5 | 100.00% | 4 | 80.00% | 1 | 20.00% | 20.00% | 0 |
| `W1_note` | 5 | 5 | 100.00% | 1 | 20.00% | 4 | 80.00% | 80.00% | 0 |
| `OVERALL` | 15 | 15 | 100.00% | 6 | 40.00% | 9 | 60.00% | 60.00% | 0 |

## Parsed-all-3 diagnostics

- UIDs parsed in all three conditions: **5**
- N0 correct: **1**, W1 flips to wrong: **0**
- Flip rate (W1 wrong | N0 correct): **0.00%**
- C1 correction rate (C1 correct | N0 wrong): **100.00%**

- CoT cbw_gen: 0/0 (NA)  |  cbw_prompt: 0/0 (NA)
- CoT reasoned_gen: 0/0 (NA)  |  reasoned_prompt: 0/0 (NA)

