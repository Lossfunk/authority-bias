# Gemma Failure-Analysis — Experiment Plan

**Context.** In Step 5 (PIQA transfer), three of four models give matched flips of
10–20%. Gemma-4-26B-A4B flips only 1.4%. The current red-thread note labels this
a "Gemma caveat." This is too thin. We want a real diagnosis.

## What the local artifacts already say

- **Direction is weak and unstable.** `neurips-results/gemma4/mechanism/gemma4_compliance_analysis/probe_results.json`
  - oriented held-out `w1_auroc_oriented = 0.598` vs GPT-OSS 0.806, Qwen 0.668
  - fold-selected pairs jump: L2/answer_position, L22/endorsement_mean (×2), L28/endorsed_answer, L2/endorsement_start
- **Prompt-metadata baseline already beats the learned direction.**
  - `authority_level_only_w1 AUROC = 0.790`, stronger than the learned 0.60
- **In-domain flip is modest, transfer is basically dead.**
  - Trivia matched flip 10.6%, net wrong-rate change +1.3 pp
  - PIQA matched flip 1.4%, net change +0.2 pp
- **Multilayer shared-vector patch didn't rescue transfer.**
  - But this only rules out *the same vector repeated at several layers*, not distributed mechanism.

## Why this matters for the paper

Gemma-4-26B-A4B is a **Mixture-of-Experts** model (~26B total, ~4B active per forward pass).
Confirmed via:
- HF card: https://huggingface.co/google/gemma-4-26B-A4B
- Google Cloud blog: https://cloud.google.com/blog/products/ai-machine-learning/gemma-4-available-on-google-cloud
- Cloudflare changelog (2026-04-04), Vercel changelog (April 2026)

Qwen3.5-27B is also MoE but transfers fine (20.6% PIQA). The distinguishing factor
may be layer depth of the authority mechanism: Qwen L2 is upstream of most routing,
Gemma L15–16 is deep inside the MoE stack.

## Hypotheses

1. **MoE routing locality** — residual patch in mid-stack reshuffles experts on PIQA
   but not on trivia; direction was fit inside trivia's routing regime.
   Reference: "The Myth of Expert Specialization in MoEs," arXiv:2604.09780.
2. **Single-direction underfits a multi-feature mechanism** — weak oriented AUROC,
   unstable fold selection, and `authority_level_only = 0.79` all point to this.
   Reference: Shafran, Geiger, Geva 2025, SNMF decomposition, arXiv:2506.10920.
3. **Direction partly tracks note-style intensity, not portable compliance content**
   — consistent with (2).
4. **Linear residual steering is off-manifold on PIQA for Gemma specifically.**
   References: Curveball Steering (arXiv:2603.09313); Non-Surjective Steered
   Activations (arXiv:2604.09839).
5. **Trivial explanation: PIQA answer-margin is larger than trivia margin on Gemma,**
   so the same perturbation strength can't cross the boundary. Must rule out first.

## Experiments (ordered by cost/value)

### E1. Shallow-layer transfer on Gemma
- **Why:** tests whether MoE-routing-locality is the culprit.
- **How:** reuse the existing L15 w1−c1 direction, but patch at L2, L5, L8, L10
  with matched residual norms.
- **Outcomes:**
  - If PIQA flip rises at shallow layers → routing-locality hypothesis wins.
  - If PIQA stays flat at every depth → mechanism is not recoverable by this direction.

### E2. Baseline-margin test on PIQA vs trivia
- **Why:** kills or strengthens the trivial explanation cheaply.
- **How:** log answer-token logit margin on baseline PIQA and baseline trivia for Gemma.
  Compare to Qwen/GPT-OSS/OLMo-2.
- **Outcomes:**
  - If Gemma's PIQA margin is much larger than its trivia margin → perturbation strength is the bottleneck.
  - If margins are comparable → margin is not the issue, mechanism explanation stands.

### E3. Per-layer refit (L15–22) with matched-norm patching
- **Why:** tests distributed-but-not-single-summary hypothesis.
- **How:** fit a separate `w1 − c1` direction at each of L15, L16, L18, L20, L22.
  Patch each at matched residual norm and sign on PIQA.
- **Outcomes:**
  - If any one transfers at ≥5–8% → distributed mechanism, wrong summary.
  - If none transfer → mechanism is genuinely non-portable under linear residual patch.

### E4. Residualize against authority-level metadata, then refit
- **Why:** isolates the "authority content" signal from the "note-style intensity" signal.
- **How:** regress activations against the authority-level indicator at each layer;
  fit `w1 − c1` on the residuals; rerun trivia and PIQA steering.
- **Outcomes:**
  - Improved AUROC and steering → original direction was contaminated by style.
  - No change → style contamination is not the bottleneck.

### E5. SNMF feature decomposition on Gemma L15–22 MLP activations
- **Why:** Shafran et al. 2025 is designed exactly for this symptom profile.
- **How:** run SNMF on trivia activations; pick feature(s) that separate W1 vs C1;
  use those as the intervention targets on both trivia and PIQA.
- **Outcomes:**
  - If sparse feature set beats the single direction and transfers → "Gemma's
    authority signal is multi-feature, poorly summarized by one residual direction."
  - If not → linear-residual story is not salvageable for Gemma.

### E6. Expert-routing analysis
- **Why:** direct test of the MoE-locality hypothesis.
- **How:** log expert selections per token at every MoE layer for
  (trivia baseline, trivia patched, PIQA baseline, PIQA patched).
- **Outcomes:**
  - If patched trivia and patched PIQA route through different experts relative
    to their baselines → patch fails to reproduce the same routing regime OOD.

### E7. Router-level or expert-gate intervention (if accessible)
- **Why:** if E6 confirms routing-locality, the honest intervention is at the router.
- **How:** patch router logits (or expert gate pre-activations) directly.
  Only attempt after E6.

## Order of operations

1. E1 and E2 first — cheapest, highest information value.
2. E3 and E4 next — still cheap, refine the mechanism picture.
3. E5 if (1)–(4) haven't resolved it.
4. E6 / E7 only if the residual-space story is clearly insufficient.

## What goes in the paper regardless

Replace the "Gemma caveat" line with something honest:

> In Gemma-4, a prompt-metadata baseline (authority-level indicator alone) predicts
> W1 compliance at AUROC 0.79, while our extracted single-direction classifier
> reaches only 0.60 (oriented). Fold-level layer selection is also unstable. We
> read this as evidence that our current single-residual-direction pipeline
> underfits Gemma's authority mechanism, not as evidence that the mechanism is
> absent — consistent with (a) the modest in-domain matched flip (10.6%) and
> (b) the failure to transfer to PIQA (1.4%). Gemma-4-26B-A4B is an MoE model
> (~4B active), and the authority layer we identified (L15–16) sits deep inside
> the MoE stack, unlike Qwen (L2) whose patch remains upstream of most routing.
> We flag this as a Gemma-specific mechanism question and sketch follow-ups in
> [appendix / future-work pointer].
