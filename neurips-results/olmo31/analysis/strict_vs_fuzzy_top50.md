## base

| cond | strict_parse | strict_acc_total | fuzzy_parse | fuzzy_acc_total | rescued |
| --- | ---: | ---: | ---: | ---: | ---: |
| N0_note | 0.762 | 0.258 | 0.778 | 0.262 | 0 |
| C1_note | 0.726 | 0.671 | 0.734 | 0.679 | 0 |
| W1_note | 0.762 | 0.052 | 0.806 | 0.052 | 0 |

N0 strict category mix:
- `wrong_confidently`: 122 (48.4%)
- `no_parse`: 60 (23.8%)
- `correct`: 65 (25.8%)
- `wrong_mentions_both`: 5 (2.0%)
- `wrong_neither_present`: 0 (0.0%)
- `no_parse` rows with max token-jaccard >= 0.60: 4/60

N0 by dataset source:
| source | n | strict_parsed | strict_correct | strict_wrong |
| --- | ---: | ---: | ---: | ---: |
| trivia_qa | 80 | 74 | 36 | 38 |
| truthful_qa | 172 | 118 | 29 | 89 |

