## base

| cond | strict_parse | strict_acc_total | fuzzy_parse | fuzzy_acc_total | rescued |
| --- | ---: | ---: | ---: | ---: | ---: |
| N0_note | 0.974 | 0.877 | 0.975 | 0.878 | 0 |
| C1_note | 0.977 | 0.970 | 0.980 | 0.972 | 0 |
| W1_note | 0.976 | 0.310 | 0.979 | 0.310 | 0 |

N0 strict category mix:
- `wrong_confidently`: 65 (7.3%)
- `no_parse`: 23 (2.6%)
- `correct`: 778 (87.7%)
- `wrong_mentions_both`: 21 (2.4%)
- `wrong_neither_present`: 0 (0.0%)
- `no_parse` rows with max token-jaccard >= 0.60: 1/23

N0 by dataset source:
| source | n | strict_parsed | strict_correct | strict_wrong |
| --- | ---: | ---: | ---: | ---: |
| trivia_qa | 481 | 470 | 438 | 32 |
| truthful_qa | 406 | 394 | 340 | 54 |

