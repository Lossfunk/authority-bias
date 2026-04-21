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

## qwen35

| cond | strict_parse | strict_acc_total | fuzzy_parse | fuzzy_acc_total | rescued |
| --- | ---: | ---: | ---: | ---: | ---: |
| N0_note | 0.998 | 0.631 | 0.998 | 0.631 | 0 |
| C1_note | 0.992 | 0.647 | 0.992 | 0.647 | 0 |
| W1_note | 0.998 | 0.488 | 0.998 | 0.488 | 0 |

N0 strict category mix:
- `wrong_confidently`: 0 (0.0%)
- `no_parse`: 1 (0.2%)
- `correct`: 318 (63.1%)
- `wrong_mentions_both`: 185 (36.7%)
- `wrong_neither_present`: 0 (0.0%)
- `no_parse` rows with max token-jaccard >= 0.60: 0/1

N0 by dataset source:
| source | n | strict_parsed | strict_correct | strict_wrong |
| --- | ---: | ---: | ---: | ---: |
| trivia_qa | 167 | 167 | 118 | 49 |
| truthful_qa | 337 | 336 | 200 | 136 |

## gemma4

| cond | strict_parse | strict_acc_total | fuzzy_parse | fuzzy_acc_total | rescued |
| --- | ---: | ---: | ---: | ---: | ---: |
| N0_note | 0.951 | 0.740 | 0.955 | 0.743 | 0 |
| C1_note | 0.955 | 0.940 | 0.962 | 0.947 | 0 |
| W1_note | 0.974 | 0.147 | 0.981 | 0.147 | 0 |

N0 strict category mix:
- `wrong_confidently`: 48 (18.1%)
- `no_parse`: 13 (4.9%)
- `correct`: 196 (74.0%)
- `wrong_mentions_both`: 8 (3.0%)
- `wrong_neither_present`: 0 (0.0%)
- `no_parse` rows with max token-jaccard >= 0.60: 1/13

N0 by dataset source:
| source | n | strict_parsed | strict_correct | strict_wrong |
| --- | ---: | ---: | ---: | ---: |
| trivia_qa | 98 | 94 | 78 | 16 |
| truthful_qa | 167 | 158 | 118 | 40 |

## gptoss

| cond | strict_parse | strict_acc_total | fuzzy_parse | fuzzy_acc_total | rescued |
| --- | ---: | ---: | ---: | ---: | ---: |
| N0_note | 0.985 | 0.584 | 0.985 | 0.584 | 0 |
| C1_note | 1.000 | 0.950 | 1.000 | 0.950 | 0 |
| W1_note | 0.980 | 0.104 | 0.980 | 0.104 | 0 |

N0 strict category mix:
- `wrong_confidently`: 2 (1.0%)
- `no_parse`: 3 (1.5%)
- `correct`: 118 (58.4%)
- `wrong_mentions_both`: 79 (39.1%)
- `wrong_neither_present`: 0 (0.0%)
- `no_parse` rows with max token-jaccard >= 0.60: 0/3

N0 by dataset source:
| source | n | strict_parsed | strict_correct | strict_wrong |
| --- | ---: | ---: | ---: | ---: |
| trivia_qa | 64 | 63 | 38 | 25 |
| truthful_qa | 138 | 136 | 80 | 56 |

