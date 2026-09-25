# OLMo Additive Steering Parser Audit

- Total rows: 1674
- Mismatch vs stored parser: 109
- Rows where stored label contradicts explicit leading `A)`/`B)` token: 0
- `neither` / invalid-option rows: 11
- Rows containing both `A)` and `B)` style labels: 0
- Parse methods: {"dynamic:consensus:answer": 127, "dynamic:consensus:final": 18, "dynamic:consensus:no_marker": 915, "dynamic:consensus:response": 10, "dynamic:none:no_marker": 17, "lead_correct_text": 496, "lead_wrong_text": 80, "neither": 11}

## By Config/Alpha

| Config | Alpha | N | Parsed | Parse % | Correct % (parsed) | Wrong % (parsed) | Wrong % (total) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cfg001_L10_Pendorsement_span_Cblock_output_Madd` | 0.0 | 93 | 91 | 97.85% | 89.01% | 10.99% | 10.75% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Madd` | 1.0 | 93 | 91 | 97.85% | 92.31% | 7.69% | 7.53% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Madd` | 3.0 | 93 | 91 | 97.85% | 92.31% | 7.69% | 7.53% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Madd` | 5.0 | 93 | 92 | 98.92% | 90.22% | 9.78% | 9.68% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Madd` | 10.0 | 93 | 91 | 97.85% | 90.11% | 9.89% | 9.68% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Madd` | 15.0 | 93 | 91 | 97.85% | 87.91% | 12.09% | 11.83% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Madd` | 0.0 | 93 | 91 | 97.85% | 89.01% | 10.99% | 10.75% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Madd` | 1.0 | 93 | 91 | 97.85% | 89.01% | 10.99% | 10.75% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Madd` | 3.0 | 93 | 93 | 100.00% | 90.32% | 9.68% | 9.68% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Madd` | 5.0 | 93 | 92 | 98.92% | 91.30% | 8.70% | 8.60% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Madd` | 10.0 | 93 | 91 | 97.85% | 91.21% | 8.79% | 8.60% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Madd` | 15.0 | 93 | 91 | 97.85% | 90.11% | 9.89% | 9.68% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Madd` | 0.0 | 93 | 91 | 97.85% | 89.01% | 10.99% | 10.75% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Madd` | 1.0 | 93 | 92 | 98.92% | 91.30% | 8.70% | 8.60% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Madd` | 3.0 | 93 | 92 | 98.92% | 90.22% | 9.78% | 9.68% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Madd` | 5.0 | 93 | 92 | 98.92% | 88.04% | 11.96% | 11.83% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Madd` | 10.0 | 93 | 92 | 98.92% | 88.04% | 11.96% | 11.83% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Madd` | 15.0 | 93 | 91 | 97.85% | 90.11% | 9.89% | 9.68% |

## Flip Rates On OLMo Parser

| Config | Alpha | Baseline-correct n | Parsed common n | Flips to wrong | Flip % |
|---|---:|---:|---:|---:|---:|
| `cfg001_L10_Pendorsement_span_Cblock_output_Madd` | 0.0 | 81 | 81 | 0 | 0.00% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Madd` | 1.0 | 81 | 81 | 0 | 0.00% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Madd` | 3.0 | 81 | 80 | 0 | 0.00% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Madd` | 5.0 | 81 | 81 | 1 | 1.23% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Madd` | 10.0 | 81 | 81 | 3 | 3.70% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Madd` | 15.0 | 81 | 81 | 4 | 4.94% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Madd` | 0.0 | 81 | 81 | 0 | 0.00% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Madd` | 1.0 | 81 | 81 | 0 | 0.00% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Madd` | 3.0 | 81 | 81 | 1 | 1.23% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Madd` | 5.0 | 81 | 81 | 1 | 1.23% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Madd` | 10.0 | 81 | 81 | 3 | 3.70% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Madd` | 15.0 | 81 | 81 | 5 | 6.17% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Madd` | 0.0 | 81 | 81 | 0 | 0.00% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Madd` | 1.0 | 81 | 81 | 0 | 0.00% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Madd` | 3.0 | 81 | 81 | 1 | 1.23% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Madd` | 5.0 | 81 | 81 | 2 | 2.47% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Madd` | 10.0 | 81 | 81 | 3 | 3.70% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Madd` | 15.0 | 81 | 81 | 4 | 4.94% |
