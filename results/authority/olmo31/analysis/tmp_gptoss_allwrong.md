## base

| cond | strict_parse | strict_acc_total | fuzzy_parse | fuzzy_acc_total | rescued |
| --- | ---: | ---: | ---: | ---: | ---: |
| N0_note | 0.985 | 0.759 | 0.985 | 0.759 | 0 |
| C1_note | 0.993 | 0.940 | 0.993 | 0.940 | 0 |
| W1_note | 0.980 | 0.275 | 0.980 | 0.275 | 0 |

N0 strict category mix:
- `wrong_confidently`: 4 (0.5%)
- `no_parse`: 12 (1.5%)
- `correct`: 615 (75.9%)
- `wrong_mentions_both`: 179 (22.1%)
- `wrong_neither_present`: 0 (0.0%)
- `no_parse` rows with max token-jaccard >= 0.60: 0/12

N0 by dataset source:
| source | n | strict_parsed | strict_correct | strict_wrong |
| --- | ---: | ---: | ---: | ---: |
| trivia_qa | 467 | 460 | 383 | 77 |
| truthful_qa | 343 | 338 | 232 | 106 |

