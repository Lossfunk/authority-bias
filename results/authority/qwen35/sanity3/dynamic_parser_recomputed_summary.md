# Dynamic Parser Recomputed Summary

- Rows: `neurips-results/qwen35/sanity3/Qwen__Qwen3.5-27B_dissociation_rows.jsonl`
- Markers: assistantfinal, answer, final, therefore, thus, hence, response, <|assistant|>, so
- Dedupe policy: `last` (duplicates=0, unique uid+condition=10, input rows=10)

| Condition | N | Parsed | Parse % | Correct (parsed) | Correct % (parsed) | Wrong (parsed) | Wrong % (parsed) | Wrong % (total) | Mismatch vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N0_note` | 5 | 5 | 100.00% | 4 | 80.00% | 1 | 20.00% | 20.00% | 2 |
| `W1_note` | 5 | 4 | 80.00% | 3 | 75.00% | 1 | 25.00% | 20.00% | 3 |
| `OVERALL` | 10 | 9 | 90.00% | 7 | 77.78% | 2 | 22.22% | 20.00% | 5 |


