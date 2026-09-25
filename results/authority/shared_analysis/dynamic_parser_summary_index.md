# Dynamic Parser — All Runs Index

Per-run recomputed metrics produced by `scripts/rerun_dynamic_parser_all.py`. Each run's detailed breakdown lives in `dynamic_parser_recomputed_summary.json` and `.md` alongside the source rows.

| Kind | Model | Run | N | Baseline | Primary Metric | Secondary Metric | Notes |
|---|---|---|---:|---|---|---|---|
| `dissociation` | `allenai/OLMo-2-0325-32B-Instruct` | `gradient_olmo2_assertive` | 960 | N0 acc 34.7% | W1 flips 70/92 (76.1%) | W1 acc 8.6% | mismatch 0 |
| `dissociation` | `allenai/OLMo-2-0325-32B-Instruct` | `gradient_olmo2_uncertain` | 960 | N0 acc 34.1% | W1 flips 36/85 (42.4%) | W1 acc 23.3% | mismatch 0 |
| `dissociation` | `allenai/OLMo-2-0325-32B-Instruct` | `gradient_olmo2_weak` | 960 | N0 acc 34.6% | W1 flips 81/91 (89.0%) | W1 acc 4.1% | mismatch 0 |
| `dissociation` | `allenai/OLMo-2-0325-32B-Instruct` | `olmo2_freegen_scaled` | 960 | N0 acc 33.4% | W1 flips 76/91 (83.5%) | W1 acc 5.5% | mismatch 0 |
| `dissociation` | `allenai/OLMo-2-0325-32B-Instruct` | `v2_olmo2_freegen` | 960 | N0 acc 34.0% | W1 flips 77/93 (82.8%) | W1 acc 5.5% | mismatch 0 |
| `dissociation` | `google/gemini-3.1-pro-preview` | `gemini31pro__dissociation_freegen` | 9374 | N0 acc 96.9% | W1 flips 7/1134 (0.6%) | W1 acc 97.2% | mismatch 0 |
| `dissociation` | `google/gemini-3.1-pro-preview` | `gemini31pro__n0_baseline_freegen` | 1813 | N0 acc 96.3% | W1 flips 0/0 (NA) | W1 acc NA | mismatch 0 |
| `dissociation` | `google/gemma-4-26B-A4B-it` | `gemma4__no_thinking_freegen_authority` | 1774 | N0 acc NA | W1 flips 0/0 (NA) | W1 acc 33.2% | mismatch 0 |
| `dissociation` | `google/gemma-4-26B-A4B-it` | `gemma4__no_thinking_freegen_baseline` | 887 | N0 acc 90.0% | W1 flips 0/0 (NA) | W1 acc NA | mismatch 0 |
| `dissociation` | `google/gemma-4-26B-A4B-it` | `gemma4__no_thinking_freegen_merged_authoritative` | 2661 | N0 acc 90.0% | W1 flips 476/757 (62.9%) | W1 acc 33.2% | mismatch 0 |
| `dissociation` | `google/gemma-4-26B-A4B-it` | `gemma4_freegen_scaled` | 2661 | N0 acc 90.1% | W1 flips 510/756 (67.5%) | W1 acc 29.8% | mismatch 0 |
| `dissociation` | `google/gemma-4-26B-A4B-it` | `gradient_gemma4_assertive` | 2661 | N0 acc 90.2% | W1 flips 100/760 (13.2%) | W1 acc 79.3% | mismatch 0 |
| `dissociation` | `google/gemma-4-26B-A4B-it` | `gradient_gemma4_uncertain` | 2661 | N0 acc 90.1% | W1 flips 44/761 (5.8%) | W1 acc 86.8% | mismatch 0 |
| `dissociation` | `google/gemma-4-26B-A4B-it` | `gradient_gemma4_weak` | 2661 | N0 acc 90.0% | W1 flips 470/757 (62.1%) | W1 acc 33.8% | mismatch 0 |
| `dissociation` | `google/gemma-4-26B-A4B-it` | `v2_gemma4_freegen` | 2661 | N0 acc 90.0% | W1 flips 490/758 (64.6%) | W1 acc 31.8% | mismatch 0 |
| `dissociation` | `openai/gpt-5.4` | `gpt54_freegen_scaled` | 2430 | N0 acc 92.1% | W1 flips 325/727 (44.7%) | W1 acc 50.6% | mismatch 0 |
| `dissociation` | `openai/gpt-5.4` | `gpt54_native_pool_staged` | 5101 | N0 acc 91.9% | W1 flips 693/1626 (42.6%) | W1 acc 57.4% | mismatch 0 |
| `dissociation` | `openai/gpt-oss-20b` | `gpt_oss_freegen_scaled` | 2430 | N0 acc 61.3% | W1 flips 339/381 (89.0%) | W1 acc 6.8% | mismatch 0 |
| `dissociation` | `openai/gpt-oss-20b` | `gradient_gpt_oss_assertive` | 2430 | N0 acc 76.6% | W1 flips 83/610 (13.6%) | W1 acc 75.4% | mismatch 0 |
| `dissociation` | `openai/gpt-oss-20b` | `gradient_gpt_oss_uncertain` | 2430 | N0 acc 77.2% | W1 flips 69/616 (11.2%) | W1 acc 78.2% | mismatch 0 |
| `dissociation` | `openai/gpt-oss-20b` | `gradient_gpt_oss_weak` | 2430 | N0 acc 76.2% | W1 flips 226/601 (37.6%) | W1 acc 52.2% | mismatch 0 |
| `dissociation` | `openai/gpt-oss-20b` | `v2_gpt_oss_freegen` | 2430 | N0 acc 77.1% | W1 flips 400/612 (65.4%) | W1 acc 28.1% | mismatch 0 |
| `dissociation` | `x-ai/grok-4.20` | `grok__grok420_dissociation_freegen` | 5439 | N0 acc 90.7% | W1 flips 1279/1462 (87.5%) | W1 acc 11.1% | mismatch 0 |
| `dissociation` | `x-ai/grok-4.20` | `grok__grok420_n0_baseline_freegen` | 1813 | N0 acc 90.5% | W1 flips 0/0 (NA) | W1 acc NA | mismatch 0 |
| `dissociation` | `x-ai/grok-4.20-multi-agent` | `grok__n0_baseline_freegen` | 1813 | N0 acc NA | W1 flips 0/0 (NA) | W1 acc NA | mismatch 0 |
| `piqa` | `Qwen/Qwen3.5-27B` | `qwen35__mechanism__piqa_forward_patch_200_freegen` | 200 | N0 wrong 40.0%, acc 60.0% | matched flips 21/102 (20.6%) | best wrong 48.6%, delta 8.6% | interpolate_mean_a0.7_L5 @ a=0.7 |
| `piqa` | `allenai/OLMo-2-0325-32B-Instruct` | `olmo2__mechanism__piqa_forward_patch_200_freegen` | 200 | N0 wrong 10.0%, acc 90.0% | matched flips 15/144 (10.4%) | best wrong 19.7%, delta 9.7% | interpolate_mean_a0.5_L16 @ a=0.5 |
| `piqa` | `google/gemma-4-26B-A4B-it` | `gemma4__mechanism__piqa_forward_patch_200_freegen_no_thinking` | 200 | N0 wrong 8.1%, acc 91.9% | matched flips 2/148 (1.4%) | best wrong 8.2%, delta 0.2% | interpolate_mean_a0.5_L20 @ a=0.5 |
| `piqa` | `openai/gpt-oss-20b` | `gpt_oss__mechanism__piqa_forward_patch_200_freegen` | 200 | N0 wrong 24.6%, acc 75.4% | matched flips 19/101 (18.8%) | best wrong 34.4%, delta 9.7% | interpolate_mean_a1_L16 @ a=1.0 |
| `steering` | `Qwen/Qwen3.5-27B` | `qwen35__mechanism__forward_patch_expanded_512_c1source_null` | 512 | N0 wrong 0.0%, acc 100.0% | matched flips 0/512 (0.0%) | best wrong 0.0%, delta 0.0% | cfg001_L2_Pendorsement_span_Cblock_output_Minterpolate_mean @ a=0.3 |
| `steering` | `Qwen/Qwen3.5-27B` | `qwen35__mechanism__forward_patch_expanded_512_w1_minus_c1` | 512 | N0 wrong 9.8%, acc 90.2% | matched flips 75/462 (16.2%) | best wrong 19.7%, delta 10.0% | cfg001_L2_Pendorsement_span_Cblock_output_Madd @ a=4.0 |
| `steering` | `Qwen/Qwen3.5-27B` | `qwen35__mechanism__forward_patch_expanded_512_w1source` | 512 | N0 wrong 8.8%, acc 91.2% | matched flips 75/467 (16.1%) | best wrong 19.1%, delta 10.4% | cfg001_L2_Pendorsement_span_Cblock_output_Minterpolate_mean @ a=1.0 |
| `steering` | `Qwen/Qwen3.5-27B` | `qwen35__mechanism__forward_patch_pilot_256` | 256 | N0 wrong 9.0%, acc 91.0% | matched flips 40/233 (17.2%) | best wrong 20.7%, delta 11.7% | cfg001_L2_Pendorsement_span_Cblock_output_Minterpolate_mean @ a=1.0 |
| `steering` | `allenai/OLMo-2-0325-32B-Instruct` | `olmo2__mechanism__forward_patch_test_256` | 93 | N0 wrong 8.7%, acc 91.3% | matched flips 16/84 (19.0%) | best wrong 23.9%, delta 15.2% | cfg002_L16_Pendorsement_span_Cblock_output_Minterpolate_mean @ a=0.5 |
| `steering` | `allenai/OLMo-2-0325-32B-Instruct` | `olmo2__mechanism__forward_patch_test_256_c1source_null` | 93 | N0 wrong 0.0%, acc 100.0% | matched flips 0/92 (0.0%) | best wrong 0.0%, delta 0.0% | cfg001_L10_Pendorsement_span_Cblock_output_Minterpolate_mean @ a=0.3 |
| `steering` | `google/gemma-4-26B-A4B-it` | `gemma4__mechanism__forward_patch_test_256` | 758 | N0 wrong 17.0%, acc 83.0% | matched flips 49/623 (7.9%) | best wrong 19.4%, delta 2.4% | cfg001_L20_Pendorsement_span_Cblock_output_Minterpolate_mean @ a=0.7 |
| `steering` | `google/gemma-4-26B-A4B-it` | `gemma4__mechanism__forward_patch_test_256_no_thinking` | 757 | N0 wrong 16.8%, acc 83.2% | matched flips 49/623 (7.9%) | best wrong 19.1%, delta 2.2% | cfg001_L20_Pendorsement_span_Cblock_output_Minterpolate_mean @ a=1.0 |
| `steering` | `google/gemma-4-26B-A4B-it` | `gemma4__mechanism__forward_patch_test_256_no_thinking_l15_l18` | 757 | N0 wrong 16.8%, acc 83.2% | matched flips 66/623 (10.6%) | best wrong 18.1%, delta 1.3% | cfg001_L15_Pendorsement_span_Cblock_output_Minterpolate_mean @ a=1.0 |
| `steering` | `google/gemma-4-26B-A4B-it` | `gemma4__mechanism__forward_patch_test_256_no_thinking_l15_l18_c1source_null` | 758 | N0 wrong 0.9%, acc 99.1% | matched flips 11/747 (1.5%) | best wrong 2.1%, delta 1.2% | cfg001_L15_Pendorsement_span_Cblock_output_Minterpolate_mean @ a=1.0 |
| `steering` | `google/gemma-4-26B-A4B-it` | `gemma4__mechanism__forward_patch_test_256_no_thinking_l15_l18_clean_freegen` | 758 | N0 wrong 1.3%, acc 98.7% | matched flips 13/745 (1.7%) | best wrong 2.5%, delta 1.2% | cfg001_L15_Pendorsement_span_Cblock_output_Minterpolate_mean @ a=0.7 |
| `steering` | `google/gemma-4-26B-A4B-it` | `gemma4__mechanism__forward_patch_test_256_no_thinking_multilayer_L16_18_20_22` | 758 | N0 wrong 1.6%, acc 98.4% | matched flips 14/745 (1.9%) | best wrong 2.4%, delta 0.8% | cfg001_L15-18-20-22_Pendorsement_span_Cblock_output_Minterpolate_mean @ a=1.0 |
| `steering` | `openai/gpt-oss-20b` | `gpt_oss__mechanism__forward_patch_test_256` | 212 | N0 wrong 20.8%, acc 79.2% | matched flips 55/168 (32.7%) | best wrong 37.3%, delta 16.5% | cfg001_L16_Pendorsement_span_Cblock_output_Minterpolate_mean @ a=1.0 |
| `steering` | `openai/gpt-oss-20b` | `gpt_oss__mechanism__forward_patch_test_256_c1source_null` | 212 | N0 wrong 2.4%, acc 97.6% | matched flips 6/207 (2.9%) | best wrong 3.8%, delta 1.4% | cfg001_L16_Pendorsement_span_Cblock_output_Minterpolate_mean @ a=1.0 |

