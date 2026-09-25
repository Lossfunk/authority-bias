# OLMo PIQA Forward Patch Parser Audit

- Total rows: 3000
- Mismatch vs stored parser: 204
- Parse methods: {"dynamic:consensus:answer": 2, "dynamic:consensus:no_marker": 572, "dynamic:consensus:therefore": 3, "dynamic:none:final": 7, "dynamic:none:no_marker": 470, "dynamic:none:therefore": 6, "dynamic:none:thus": 1, "lead_correct_text": 1500, "lead_wrong_text": 180, "neither": 5, "standalone": 242, "start_label": 12}

## Corrected Summary

| Config | Layer | Alpha | N | Parsed | Parse % | Acc % (parsed) | Wrong % (parsed) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `interpolate_mean_a0.3_L10` | 10 | 0.3 | 200 | 171 | 85.50% | 85.96% | 14.04% |
| `interpolate_mean_a0.3_L16` | 16 | 0.3 | 200 | 171 | 85.50% | 85.96% | 14.04% |
| `interpolate_mean_a0.3_L22` | 22 | 0.3 | 200 | 170 | 85.00% | 84.12% | 15.88% |
| `interpolate_mean_a0.5_L10` | 10 | 0.5 | 200 | 158 | 79.00% | 79.75% | 20.25% |
| `interpolate_mean_a0.5_L16` | 16 | 0.5 | 200 | 161 | 80.50% | 75.16% | 24.84% |
| `interpolate_mean_a0.5_L22` | 22 | 0.5 | 200 | 168 | 84.00% | 78.57% | 21.43% |
| `interpolate_mean_a0.7_L10` | 10 | 0.7 | 200 | 176 | 88.00% | 84.66% | 15.34% |
| `interpolate_mean_a0.7_L16` | 16 | 0.7 | 200 | 170 | 85.00% | 85.29% | 14.71% |
| `interpolate_mean_a0.7_L22` | 22 | 0.7 | 200 | 169 | 84.50% | 84.02% | 15.98% |
| `interpolate_mean_a0_L10` | 10 | 0.0 | 200 | 166 | 83.00% | 85.54% | 14.46% |
| `interpolate_mean_a0_L16` | 16 | 0.0 | 200 | 166 | 83.00% | 85.54% | 14.46% |
| `interpolate_mean_a0_L22` | 22 | 0.0 | 200 | 166 | 83.00% | 85.54% | 14.46% |
| `interpolate_mean_a1_L10` | 10 | 1.0 | 200 | 177 | 88.50% | 85.88% | 14.12% |
| `interpolate_mean_a1_L16` | 16 | 1.0 | 200 | 163 | 81.50% | 84.05% | 15.95% |
| `interpolate_mean_a1_L22` | 22 | 1.0 | 200 | 159 | 79.50% | 82.39% | 17.61% |

## Matched Baseline-Correct Flip Rates

| Layer | Alpha | Common parsed | Baseline-correct | Flips to wrong | Flip % | Wrong delta |
|---|---:|---:|---:|---:|---:|---:|
| 10 | 0.3 | 164 | 140 | 0 | 0.00% | -0.61 pp |
| 10 | 0.5 | 148 | 125 | 9 | 7.20% | 4.73 pp |
| 10 | 0.7 | 159 | 136 | 4 | 2.94% | -0.63 pp |
| 10 | 1.0 | 160 | 138 | 4 | 2.90% | -0.63 pp |
| 16 | 0.3 | 165 | 141 | 0 | 0.00% | -0.61 pp |
| 16 | 0.5 | 154 | 131 | 16 | 12.21% | 9.74 pp |
| 16 | 0.7 | 158 | 135 | 3 | 2.22% | -0.63 pp |
| 16 | 1.0 | 151 | 129 | 4 | 3.10% | 0.66 pp |
| 22 | 0.3 | 166 | 142 | 2 | 1.41% | 1.20 pp |
| 22 | 0.5 | 161 | 137 | 12 | 8.76% | 6.21 pp |
| 22 | 0.7 | 158 | 135 | 4 | 2.96% | 0.63 pp |
| 22 | 1.0 | 149 | 127 | 6 | 4.72% | 2.01 pp |
