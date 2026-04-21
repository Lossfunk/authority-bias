## base

| cond | strict_parse | strict_acc_total | fuzzy_parse | fuzzy_acc_total | rescued |
| --- | ---: | ---: | ---: | ---: | ---: |
| N0_note | 0.784 | 0.369 | 0.794 | 0.369 | 0 |
| C1_note | 0.748 | 0.687 | 0.758 | 0.696 | 0 |
| W1_note | 0.790 | 0.079 | 0.835 | 0.079 | 0 |

N0 strict category mix:
- `wrong_confidently`: 188 (37.3%)
- `no_parse`: 109 (21.6%)
- `correct`: 186 (36.9%)
- `wrong_mentions_both`: 21 (4.2%)
- `wrong_neither_present`: 0 (0.0%)
- `no_parse` rows with max token-jaccard >= 0.60: 5/109

N0 by dataset source:
| source | n | strict_parsed | strict_correct | strict_wrong |
| --- | ---: | ---: | ---: | ---: |
| trivia_qa | 167 | 155 | 91 | 64 |
| truthful_qa | 337 | 240 | 95 | 145 |

