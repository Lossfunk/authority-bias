# GPT-OSS/Gemma local status and JarvisLabs URL check

Generated from local artifacts and the provided Jupyter URL.

## Remote/JarvisLabs access

- The provided URL `https://b3e94b401415.notebooksn.jarvislabs.net/lab` is reachable, but Jupyter token authentication is enabled.
- `https://b3e94b401415.notebooksn.jarvislabs.net/api/contents` returns `Forbidden`, so I cannot list or download files from that notebook server without the token/cookie.
- The local `jl` CLI is authenticated, but `jl list --json` and `jl filesystem list --json` both return empty lists. So this running notebook is not visible through the current CLI account/session.

To let me inspect/download from it, send the full tokenized URL, e.g. `...?token=...`, or expose SSH/CLI access for that Jarvis instance.

## GPT-OSS and Gemma: correct-source preservation availability

Short version: I do **not** see local all-condition GPT-OSS or Gemma runs for correct-source preservation. The local causal-deconfound outputs for these two are W1-only. They are useful for wrong-source mitigation, but they cannot answer whether C1 correct-source behavior is preserved.

### Existing W1-only principled-correction runs

| run | alpha=0 W1 baseline | best alpha=1 W1 | path |
|---|---:|---:|---|
| GPT-OSS authority | wrong 56.9%, acc 43.1%, parsed 511/512, `cfg001_L16_Pendorsement_span_Cblock_output_Mproject_out_direction` | wrong 57.7%, acc 42.3%, parsed 511/512, `cfg003_L20_Pendorsement_span_Cblock_output_Mproject_out_direction` | `causal-deconfound/principled_correction_w1/gpt_oss/trivia_authority/steering_summary.json` |
| GPT-OSS residualized | n/a | wrong 58.3%, acc 41.7%, parsed 511/512, `cfg003_L20_Pendorsement_span_Cblock_output_Mproject_out_direction` | `causal-deconfound/principled_correction_w1/gpt_oss/trivia_residualized/steering_summary.json` |
| GPT-OSS assistant | n/a | wrong 58.3%, acc 41.7%, parsed 511/512, `cfg003_L20_Pendorsement_span_Cblock_output_Mproject_out_direction` | `causal-deconfound/principled_correction_w1/gpt_oss/trivia_assistant/steering_summary.json` |
| Gemma authority | wrong 57.8%, acc 42.2%, parsed 510/512, `cfg001_L18_Pendorsement_span_Cblock_output_Mproject_out_direction` | wrong 56.0%, acc 44.0%, parsed 511/512, `cfg003_L22_Pendorsement_span_Cblock_output_Mproject_out_direction` | `causal-deconfound/principled_correction_w1/gemma4/trivia_authority/steering_summary.json` |
| Gemma residualized | n/a | wrong 57.5%, acc 42.5%, parsed 511/512, `cfg003_L22_Pendorsement_span_Cblock_output_Mproject_out_direction` | `causal-deconfound/principled_correction_w1/gemma4/trivia_residualized/steering_summary.json` |
| Gemma assistant | n/a | wrong 54.7%, acc 45.3%, parsed 512/512, `cfg001_L18_Pendorsement_span_Cblock_output_Mproject_out_direction` | `causal-deconfound/principled_correction_w1/gemma4/trivia_assistant/steering_summary.json` |

### Existing W1-only Wang/Pareto mitigation runs

These are the mitigation results used elsewhere, but again they are W1-only and do not answer C1 preservation.

| run | best alpha=1 W1 | path |
|---|---:|---|
| GPT-OSS residualized | wrong 44.0%, acc 56.0%, parsed 511/512, `cfg001_L16_Pendorsement_span_Cblock_output_Mproject_out_direction` | `wang-pareto-results/gptoss/mitigation_pareto/gpt_oss/trivia_residualized/steering_summary.json` |
| GPT-OSS assistant | wrong 60.0%, acc 40.0%, parsed 511/512, `cfg001_L16_Pendorsement_span_Cblock_output_Mproject_out_direction` | `wang-pareto-results/gptoss/mitigation_pareto/gpt_oss/trivia_assistant/steering_summary.json` |
| GPT-OSS CAA | wrong 58.0%, acc 42.0%, parsed 511/512, `cfg001_L16_Pendorsement_span_Cblock_output_Msubtract` | `wang-pareto-results/gptoss/mitigation_pareto/gpt_oss/trivia_caa/steering_summary.json` |
| Gemma residualized | wrong 56.6%, acc 43.4%, parsed 511/512, `cfg003_L22_Pendorsement_span_Cblock_output_Mproject_out_direction` | `wang-pareto-results/gemma4/mitigation_pareto/gemma4/trivia_residualized/steering_summary.json` |
| Gemma assistant | wrong 54.8%, acc 45.2%, parsed 511/512, `cfg001_L18_Pendorsement_span_Cblock_output_Mproject_out_direction` | `wang-pareto-results/gemma4/mitigation_pareto/gemma4/trivia_assistant/steering_summary.json` |
| Gemma CAA | wrong 57.8%, acc 42.2%, parsed 510/512, `cfg002_L20_Pendorsement_span_Cblock_output_Msubtract` | `wang-pareto-results/gemma4/mitigation_pareto/gemma4/trivia_caa/steering_summary.json` |

## Lexical valence/arousal artifacts

The geometric VA planes exist locally for GPT-OSS, Gemma, OLMo-2, and OLMo-3.1. I do not see completed causal wording-deconfound intervention runs that remove VA alone and compare against authority-minus-VA.

| model/artifact | layers | position | key overlap numbers | file |
|---|---|---|---|---|
| openai/gpt-oss-20b | [16, 18] | endorsement_end | L16: val_corr=0.282, aro_corr=0.178, pre_cos=0.354; L18: val_corr=0.263, aro_corr=0.185, pre_cos=0.261 | `neurips-results/gpt-oss/mechanism/lexical_va_plane/summary.json` |
| google/gemma-4-26B-A4B-it | [15, 18] | endorsement_end | L15: val_corr=0.151, aro_corr=0.069, pre_cos=0.342; L18: val_corr=0.160, aro_corr=0.113, pre_cos=0.256 | `neurips-results/gemma4/mechanism/lexical_va_plane/summary.json` |
| allenai/OLMo-2-0325-32B-Instruct | [16, 22] | endorsement_end | L16: val_corr=0.196, aro_corr=0.145, pre_cos=0.593; L22: val_corr=0.197, aro_corr=0.180, pre_cos=0.486 | `neurips-results/olmo2/mechanism/lexical_va_plane/summary.json` |
| allenai/OLMo-3.1-32B-Instruct | [5, 15, 18, 22] | endorsement_start | L5: val_corr=0.270, aro_corr=0.238, pre_cos=-0.194; L15: val_corr=0.268, aro_corr=0.210, pre_cos=-0.126; L18: val_corr=0.266, aro_corr=0.221, pre_cos=0.074; L22: val_corr=0.246, aro_corr=0.214, pre_cos=0.029 | `neurips-results/olmo31/mechanism/lexical_va_plane/summary.json` |

## GPU commands for GPT-OSS/Gemma C1 preservation

Once GPU access works, this should run the missing all-condition correct-source preservation for GPT-OSS and Gemma:

```bash
uv run python -m src.exp19.run_assistant_axis_causal_deconfound \
  --repo-root "$(pwd)" \
  --models gpt_oss,gemma4 \
  --variants authority,residualized \
  --run-trivia --execute \
  --output-root causal-deconfound/gpt_gemma_ab_project_out_allcond \
  --trivia-conditions "N0_note,W1_note,C1_note" \
  --trivia-intervention-mode project_out_direction \
  --trivia-norm-scaling none \
  --trivia-alpha-override "1.0" \
  --trivia-position-mode auto \
  --trivia-max-items 512 \
  --trivia-batch-size 96 \
  --no-compile \
  --dedupe-alpha-zero \
  --skip-missing-axis \
  --allow-missing-frozen-settings \
  --checkpoint-path causal-deconfound/gpt_gemma_ab_project_out_allcond/manifest.json
```

For OLMo via the notebook URL, I need the Jupyter token or a CLI-visible Jarvis machine. Without that, I cannot inspect/download files or launch the OLMo run there.
