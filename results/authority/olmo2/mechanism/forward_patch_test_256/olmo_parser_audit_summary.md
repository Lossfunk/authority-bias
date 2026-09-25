# OLMo Additive Steering Parser Audit

- Total rows: 1395
- Mismatch vs stored parser: 127
- Rows where stored label contradicts explicit leading `A)`/`B)` token: 0
- `neither` / invalid-option rows: 11
- Rows containing both `A)` and `B)` style labels: 1
- Parse methods: {"dynamic:consensus:answer": 81, "dynamic:consensus:final": 15, "dynamic:consensus:no_marker": 761, "dynamic:consensus:response": 11, "dynamic:consensus:therefore": 11, "dynamic:none:answer": 7, "dynamic:none:no_marker": 14, "dynamic:none:therefore": 1, "lead_correct_text": 394, "lead_wrong_text": 87, "neither": 11, "standalone": 2}

## By Config/Alpha

| Config | Alpha | N | Parsed | Parse % | Correct % (parsed) | Wrong % (parsed) | Wrong % (total) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cfg001_L10_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.0 | 93 | 91 | 97.85% | 89.01% | 10.99% | 10.75% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.3 | 93 | 91 | 97.85% | 87.91% | 12.09% | 11.83% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.5 | 93 | 91 | 97.85% | 74.73% | 25.27% | 24.73% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.7 | 93 | 92 | 98.92% | 81.52% | 18.48% | 18.28% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Minterpolate_mean` | 1.0 | 93 | 92 | 98.92% | 82.61% | 17.39% | 17.20% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.0 | 93 | 91 | 97.85% | 89.01% | 10.99% | 10.75% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.3 | 93 | 90 | 96.77% | 88.89% | 11.11% | 10.75% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.5 | 93 | 90 | 96.77% | 78.89% | 21.11% | 20.43% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.7 | 93 | 90 | 96.77% | 82.22% | 17.78% | 17.20% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Minterpolate_mean` | 1.0 | 93 | 92 | 98.92% | 79.35% | 20.65% | 20.43% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.0 | 93 | 91 | 97.85% | 89.01% | 10.99% | 10.75% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.3 | 93 | 91 | 97.85% | 80.22% | 19.78% | 19.35% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.5 | 93 | 88 | 94.62% | 80.68% | 19.32% | 18.28% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.7 | 93 | 91 | 97.85% | 84.62% | 15.38% | 15.05% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Minterpolate_mean` | 1.0 | 93 | 91 | 97.85% | 79.12% | 20.88% | 20.43% |

## Flip Rates On OLMo Parser

| Config | Alpha | Baseline-correct n | Parsed common n | Flips to wrong | Flip % |
|---|---:|---:|---:|---:|---:|
| `cfg001_L10_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.0 | 81 | 81 | 0 | 0.00% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.3 | 81 | 80 | 3 | 3.75% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.5 | 81 | 80 | 13 | 16.25% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.7 | 81 | 80 | 10 | 12.50% |
| `cfg001_L10_Pendorsement_span_Cblock_output_Minterpolate_mean` | 1.0 | 81 | 80 | 9 | 11.25% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.0 | 81 | 81 | 0 | 0.00% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.3 | 81 | 80 | 3 | 3.75% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.5 | 81 | 80 | 13 | 16.25% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.7 | 81 | 80 | 9 | 11.25% |
| `cfg002_L16_Pendorsement_span_Cblock_output_Minterpolate_mean` | 1.0 | 81 | 80 | 10 | 12.50% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.0 | 81 | 81 | 0 | 0.00% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.3 | 81 | 80 | 9 | 11.25% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.5 | 81 | 78 | 11 | 14.10% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Minterpolate_mean` | 0.7 | 81 | 80 | 9 | 11.25% |
| `cfg003_L22_Pendorsement_span_Cblock_output_Minterpolate_mean` | 1.0 | 81 | 79 | 10 | 12.66% |
