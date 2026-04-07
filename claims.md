# Claims

Paper: "Instruction-Dependent Evidence Gating in Language Models"

This document lists every claim we intend to make, the evidence behind it, and its current status. If a claim isn't on this list, it doesn't go in the paper.

---

## Core claims

### C1. Accuracy instructions can backfire
Telling a model to "answer correctly even if the speaker is wrong" (i1a) causes it to reject correct endorsements on items where it was initially wrong. A differently-worded instruction with the same intent (i1c: "base your answer on the facts") does not backfire and often helps.

**Evidence:** Qwen3-4B factual QA: i1a effect = -0.68 (entrenches), i1c effect = +0.42 (corrects). CIs don't overlap.
**Replication:** Qwen3-30B (3/4 tags), Llama-3.1-8B (attenuated), PIQA domain, DebateQA (no ground truth). GPT-oss-20B shows the effect after prior-state conditioning.
**Status:** Strong. Multi-model, multi-domain.

### C2. The credulity dial
i1a and i1c don't target specific items. i1a is general skepticism: it protects prior-correct items from wrong endorsements AND blocks prior-wrong items from correct endorsements. i1c is general openness. The backfire on prior-wrong items is a side effect of blanket skepticism, not a targeted failure.

**Evidence:** Monotonic instruction wording gradient from pathological (i1a, selectivity -0.43) to neutral (consider_info_carefully, +0.01). Speaker authority crossover: no single instruction is universally helpful.
**Status:** Strong. Clean gradient, interpretable.

### C3. The correction-gating direction exists at the endorsement position
At the endorsement position (where the speaker states their opinion), layer 23 representations in Qwen3-4B encode whether the model will correct or resist under i1a. This direction is causal, evidence-specific, cross-domain transferable, and orthogonal to known directions (opinion, authority).

**Evidence:** Probe CV accuracy 81% (cross-condition), causal patching shifts margin +1.93 with 8.6% answer flips. Transfer AUROC 0.83 across domains. Confidence regressed out: still 75.5%. Surface features (TF-IDF, length, dataset): null. Cosine to known directions <0.1. Head-level circuit: L24.H8 (endorsement-reader, 46% attention to endorsement) and L23.H7 (instruction-reader, 60% attention to instruction) are the top gating heads. But mechanism is distributed: top-5 head cascade gives only 2% flip rate vs 8.6% full-layer. L22.H3 (low gating divergence) has the highest single-head ablation effect (6.6% flip).
**Status:** Strong (Qwen3-4B). GPT-oss-20B has the representational structure (probe AUROC 0.677, transfer 0.926) but causal patching is weak (distributed computation, see C7).

### C4. Prevention, not suppression
For resisting items under i1a, the correct answer never appears in intermediate layers. The logit lens margin stays negative from embedding through final layer. Under i1c, the same items show positive correction starting at L22. The model doesn't build the correct representation and suppress it. It prevents the representation from forming.

**Evidence:** Logit lens on Qwen3-4B instruct: resisting items under i1a have margin -8.30 at L32 (never positive at any layer). Under i1c, same items reach +1.13 at L32. Temporal commitment analysis: vulnerability window at L18, point of no return at L24. Resisting items maintain higher entropy (0.45 vs 0.25) through final layer. Model commits around L20-24, after which interventions have zero effect.
**Status:** Strong for Qwen3-4B. Needs cross-model replication (GPT-oss logit lens not yet run).

### C5. The prevention mechanism is attention-mediated
At the gating layers (L22-24), the instruction-dependent signal is carried by attention, not MLP. Attention output divergence between i1a and i1c is 8-9x larger than MLP output divergence when projected through the unembedding matrix. Attention heads at these layers determine whether evidence-relevant tokens get read into the residual stream. MLP dominates at later layers (L28-33), amplifying whatever attention let through.

**Evidence:** Attn/MLP decomposition on Qwen3-4B: L24 resisting items Attn div = +8.70 vs MLP div = +1.01 (ratio 8.6x). L31 reverses: MLP div = +22.87 vs Attn div = -0.42. Now with head-level resolution: instruction-reading and endorsement-reading heads show complementary specialization. Functional separation: L23.H7 reads instruction (60%), L24.H8 reads endorsement (46%).
**Status:** Strong for Qwen3-4B. Needs cross-model replication.

### C6. RLHF enhanced a latent mechanism, not created one from scratch
The base model (Qwen3-4B-Base) has no instruction-dependent gating. Resisting items end at +0.76 under i1a (positive, i.e. it corrects). i1a-i1c divergence is 34x smaller than in the instruct model. But CMAP shows the base model's downstream layers (L25-35) can process gating signals if they receive them: patching instruct L22-24 into the base model recovers 41% of the full gating effect. Reverse CMAP (replacing instruct L22-24 with base activations) destroys 99.3% of gating.

**Evidence:** Base model logit lens + forward CMAP (41% recovery) + reverse CMAP (99.3% loss). Consistent with Prakash et al. (ICLR 2024) "fine-tuning enhances existing mechanisms."
**Status:** Strong for Qwen3-4B.

### C7. Cross-architecture divergence in gating implementation
Different model families implement evidence gating differently:
- **Qwen3-4B:** Sharply localized at L23. Single-layer patching produces 8.6% flips.
- **GPT-oss-20B:** Distributed across layers 1-12. Signal spread (top probe layers: L9, L8, L1, L10). Multi-layer patching (k=5) needed for 59% positive rate, but only 0.9% flips even at k=8.
- **Gemma-4-26B:** No confidence-independent signal (N0 probe collapses to chance after confidence removal). Behavior is almost entirely confidence-driven. Near-zero entrenchment (1/887 resist), confidence-driven.
- **Qwen3.5-27B:** Probe AUROC 0.923 at L43, near-zero entrenchment (7/208 resist).
- **OLMo-2-32B:** Probe AUROC 0.941 at L46, patching NULL.

**Evidence:** Qwen probe peaks at L23 (AUROC 0.81). GPT-oss spread across L1-12 (AUROC 0.678). Gemma N0 three-way decomposition: 0.526 after confidence removal (near chance). Multi-layer patching comparison. Now 5 model families with mechanism data: Qwen, GPT-oss, OLMo, Gemma, Llama.
**Status:** Strong. Five-way comparison across model families.

### C8. The N0 pre-endorsement signal
Before any instruction or endorsement, the model's representation of the question already predicts whether it will accept or resist future correction. This signal is confidence-independent (survives regression) and position-independent. It correlates with actual update magnitude (partial r = 0.32 Qwen, 0.31 GPT-oss, both p < 1e-11).

**Evidence:** N0 probe three-way decomposition: Qwen 0.612, GPT-oss 0.659 after removing confidence and position. Residual predicts actual update magnitude controlling for confidence.
**Status:** Strong. Cross-model replication on Qwen + GPT-oss.

### C9. The N0 signal is not reducible to observable features
Semantic distance between correct/wrong answers (r=-0.02), confidence extremity (r=-0.09), question length (r=-0.10), answer length ratio (r=-0.15), dataset source -- combined R^2 = 0.055. Observable features explain 5.5% of the N0 residual. The signal is a genuinely internal representation.

**Evidence:** N0 signal exploration experiment with 6 feature types.
**Status:** Solid for Qwen3-4B. Needs GPT-oss replication of the exploration.

### C10. Gating is distributed, not localized
Head surgery shows top-5 divergence heads capture only 2% of the 8.6% full-layer effect. The highest causal-importance head (L22.H3) ranks 65th by divergence. The mechanism requires coordination across many heads, not a small circuit of "gating heads."

**Evidence:** Head-level ablation: top-5 head cascade gives 2% flip rate vs 8.6% full-layer. L22.H3 has highest single-head ablation effect (6.6% flip) but ranks 65th by gating divergence.
**Status:** Strong for Qwen3-4B.

### C11. Correction is processed but overridden
The model internally processes correction instructions even for resisting items, but downstream computation overrides the signal. Multiplicative gating experiment shows the correction instruction (i1c) shifts gate values even for resisting items, and resisting items maintain higher entropy through the final layer.

**Evidence:** Multiplicative gating: correction instruction (i1c) shifts gate values even for resisting items (L24: 1.62 -> 3.23). Resisting items maintain higher entropy through final layer (0.45 vs 0.25).
**Status:** Strong for Qwen3-4B.

### C12. Evidence gating is persona-invariant
6 personas (default, scientist, skeptic, judge, empath, therapist) produce near-identical correction/resistance rates. Gating is a representational phenomenon, not modulated by persona conditioning.

**Evidence:** Persona sweep across 6 personas: near-identical correction/resistance rates. Empath shows marginal margin compression but no asymmetric evidence discrimination.
**Status:** Strong for Qwen3-4B.

---

## Supporting claims (not centerpiece, but included)

### S1. Resistance is model-specific, not item-specific
The same items don't resist across models. Qwen vs GPT-oss kappa = 0.046. Holds within confidence bins.

### S2. Common misconceptions are easier to correct, not harder
Consensus-wrong items (all 3 models get wrong) have lower resist rates than unique-wrong items. TruthfulQA (common misconceptions) corrects at 74% vs TriviaQA (obscure facts) at 56%.

### S3. Forced-choice resistance doesn't survive into generation
GPT-oss confabulation pilot: 424/439 items self-corrected when asked to defend their answer. The forced-choice A/B format may overstate resistance.

### S4. GPT-oss DebateQA is decodable but not steerable
Representational structure exists (probe AUROC 0.603, transfer 0.883) but causal patching is null. "Thermometer not heater."

### S5. Precision-weighting partial fit
i1c/i1a flip-rate ratio scales with confidence (1.44 -> 1.75 -> 2.21). But instruction has effect even on near-zero-confidence items (+6.4 logits), which pure precision-weighting doesn't predict. Instructions set mode, not just precision.

---

## Null results (included as honest reporting)

### N1. CMAP forward sweep is trivial
Correcting vs resisting profiles are nearly identical (r=0.989). The monotonic increase is "later layers = more processed information," not gating-specific.

### N2. Surface features don't predict resistance
TF-IDF, question length, source dataset all get null probe accuracy. The gating direction is not a surface confound.

### N3. The gating direction is orthogonal to known directions
Cosine similarity to opinion-like (0.093), authority-like (-0.005, 0.060) directions is negligible. Vector norm drops 0.6% after projection.

---

## What's missing (experiments needed before paper)

### M1. Cross-model replication of prevention finding
Run logit lens on GPT-oss-20B to verify prevention (not suppression) holds in a different model family. Currently Qwen-only for C4.

### M2. Cross-model replication of attention-mediated mechanism
Run attn/MLP decomposition on GPT-oss-20B. Currently Qwen-only for C5.

### M3. Scale
All mechanism work is on 4B (Qwen) and 20B (GPT-oss). Need at least one experiment at larger scale to make claims about generality. Qwen3-30B has some behavioral data but no logit lens or decomposition.

### M4. ~~Head-level analysis at L22-24~~ DONE
Head-level analysis completed. L24.H8 (endorsement-reader), L23.H7 (instruction-reader), L22.H3 (highest causal effect). Results incorporated into C3, C5, C10.

### M5. N0 exploration on GPT-oss
The "N0 is not reducible to surface features" result (C9) is Qwen-only. Need GPT-oss replication.
