# Next-experiment availability check and local Qwen correct-source preservation

Generated from local artifacts after attempting the provided SSH host.

## Remote check

- `ssh -o StrictHostKeyChecking=no root@217.18.55.101` failed with `Permission denied (publickey)`.
- Retrying with the explicit local key `~/.ssh/id_rsa` also failed with `Permission denied (publickey)`.
- I could not inspect or run on that GPU box from this session without a working key/agent for that host.

## Local availability

- Local project environment has Torch under `uv`, but `torch.cuda.is_available()` is `False` and `device_count=0`.
- OLMo-2 local artifacts needed to launch intervention runs are present: authority direction, assistant axis, masks, and cached activations.
- OLMo-3.1 local authority direction, masks, and cached activations are present, but the assistant-axis file expected by `src.exp19.common.default_model_specs()` is missing locally.
- Existing OLMo-2/OLMo-3.1 causal-deconfound outputs under `causal-deconfound/principled_correction_w1/` are W1-only. They do not contain the C1 correct-source preservation condition.
- Existing Qwen all-condition project-out runs are already local under `causal-deconfound/qwen35_ab_project_out_allcond/`; those are enough to answer correct-source preservation for Qwen without new GPU inference.

## Local result: Qwen correct-source preservation after authority removal

Trivia, project-out direction, layer 5, endorsement span. Rates use parsed denominators with 95% Wilson intervals.

| variant | condition | alpha | accuracy | wrong rate | parse rate | source |
|---|---|---:|---:|---:|---:|---|
| authority | N0_note | 0 | 96.6% [94.6, 97.9] (485/502) | 3.4% [2.1, 5.4] (17/502) | 98.0% (502/512) | `causal-deconfound/qwen35_ab_project_out_allcond/qwen35/trivia_authority/steering_summary.json` |
| authority | N0_note | 1 | 96.4% [94.4, 97.7] (482/500) | 3.6% [2.3, 5.6] (18/500) | 97.7% (500/512) | `causal-deconfound/qwen35_ab_project_out_allcond/qwen35/trivia_authority/steering_summary.json` |
| authority | C1_note | 0 | 98.8% [97.4, 99.4] (494/500) | 1.2% [0.6, 2.6] (6/500) | 97.7% (500/512) | `causal-deconfound/qwen35_ab_project_out_allcond/qwen35/trivia_authority/steering_summary.json` |
| authority | C1_note | 1 | 95.0% [92.7, 96.6] (474/499) | 5.0% [3.4, 7.3] (25/499) | 97.5% (499/512) | `causal-deconfound/qwen35_ab_project_out_allcond/qwen35/trivia_authority/steering_summary.json` |
| authority | W1_note | 0 | 60.6% [56.3, 64.8] (303/500) | 39.4% [35.2, 43.7] (197/500) | 97.7% (500/512) | `causal-deconfound/qwen35_ab_project_out_allcond/qwen35/trivia_authority/steering_summary.json` |
| authority | W1_note | 1 | 93.7% [91.2, 95.5] (463/494) | 6.3% [4.5, 8.8] (31/494) | 96.5% (494/512) | `causal-deconfound/qwen35_ab_project_out_allcond/qwen35/trivia_authority/steering_summary.json` |
| assistant | N0_note | 1 | 97.0% [95.1, 98.2] (487/502) | 3.0% [1.8, 4.9] (15/502) | 98.0% (502/512) | `causal-deconfound/qwen35_ab_project_out_allcond/qwen35/trivia_assistant/steering_summary.json` |
| assistant | C1_note | 1 | 98.4% [96.9, 99.2] (493/501) | 1.6% [0.8, 3.1] (8/501) | 97.9% (501/512) | `causal-deconfound/qwen35_ab_project_out_allcond/qwen35/trivia_assistant/steering_summary.json` |
| assistant | W1_note | 1 | 59.6% [55.2, 63.8] (298/500) | 40.4% [36.2, 44.8] (202/500) | 97.7% (500/512) | `causal-deconfound/qwen35_ab_project_out_allcond/qwen35/trivia_assistant/steering_summary.json` |
| residualized | N0_note | 1 | 96.6% [94.6, 97.9] (482/499) | 3.4% [2.1, 5.4] (17/499) | 97.5% (499/512) | `causal-deconfound/qwen35_ab_project_out_allcond/qwen35/trivia_residualized/steering_summary.json` |
| residualized | C1_note | 1 | 95.8% [93.6, 97.2] (477/498) | 4.2% [2.8, 6.4] (21/498) | 97.3% (498/512) | `causal-deconfound/qwen35_ab_project_out_allcond/qwen35/trivia_residualized/steering_summary.json` |
| residualized | W1_note | 1 | 94.2% [91.7, 95.9] (468/497) | 5.8% [4.1, 8.3] (29/497) | 97.1% (497/512) | `causal-deconfound/qwen35_ab_project_out_allcond/qwen35/trivia_residualized/steering_summary.json` |

### Interpretation

- Qwen correct-source accuracy drops from **98.8%** at alpha=0 to **95.0%** after raw authority removal, a drop of about **3.8 pp**.
- With authority residualized against the assistant axis, C1 accuracy is **95.8%** at alpha=1. This is still high, while W1 wrong-source compliance falls to **5.8%**.
- For Qwen, the local all-condition run supports the stronger mitigation framing: the intervention sharply reduces wrong-source compliance while mostly preserving correct-source behavior. It is not a pure source-reliance deletion, although there is a small C1 drop.

## What still needs GPU

1. **OLMo-2 / OLMo-3.1 correct-source preservation**: not locally complete because existing OLMo outputs are W1-only. Needs new generation runs.
2. **Causal wording deconfound for Qwen/GPT-OSS**: needs intervention generation with valence/arousal subspace removal and authority-minus-valence/arousal.
3. **OLMo source/user 2x2**: needs new model inference.

## Exact commands I would run once SSH works

From the repo root on the GPU box, for OLMo-2 correct-source preservation only:

```bash
uv run python -m src.exp19.run_assistant_axis_causal_deconfound \
  --repo-root "$(pwd)" \
  --models olmo2 \
  --variants authority,residualized \
  --run-trivia --execute \
  --output-root neurips-results/exp19/olmo2_ab_project_out_allcond \
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
  --checkpoint-path neurips-results/exp19/olmo2_ab_project_out_allcond/manifest.json
```

For OLMo-3.1, first fix/provide the expected assistant-axis file if residualized runs are needed. Authority-only C1 preservation can run without the assistant variant by using `--variants authority`.
