## base

| cond | strict_parse | strict_acc_total | fuzzy_parse | fuzzy_acc_total | rescued |
| --- | ---: | ---: | ---: | ---: | ---: |
| N0_note | 0.308 | 0.125 | 0.308 | 0.125 | 0 |
| C1_note | 0.394 | 0.173 | 0.394 | 0.173 | 0 |
| W1_note | 0.428 | 0.231 | 0.428 | 0.231 | 0 |

N0 strict category mix:
- `wrong_confidently`: 4 (1.9%)
- `no_parse`: 144 (69.2%)
- `correct`: 26 (12.5%)
- `wrong_mentions_both`: 34 (16.3%)
- `wrong_neither_present`: 0 (0.0%)
- `no_parse` rows with max token-jaccard >= 0.60: 0/144

N0 by dataset source:
| source | n | strict_parsed | strict_correct | strict_wrong |
| --- | ---: | ---: | ---: | ---: |
| trivia_qa | 72 | 34 | 15 | 19 |
| truthful_qa | 136 | 30 | 11 | 19 |

