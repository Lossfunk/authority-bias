# Creative Mechanistic Interpretability Experiments

**Purpose**: Novel experiment designs to strengthen the Instruction-Dependent Evidence Gating paper for NeurIPS 2026.  
**Generated**: 2026-04-07  
**Grounded in**: Current results (Qwen3-4B flagship, GPT-oss-20B, Gemma-4-26B, OLMo-2-32B, Qwen3.5-27B, Qwen3-30B), existing infrastructure (`src/mechanism/`), and recent literature (Bigelow 2025, Sarfati 2026, DSAS, activation scaling).

---

## Experiment 1: Gating Head Surgery at L22-24

**RQ**: Which specific attention heads at L22-24 implement the instruction-dependent gating in Qwen3-4B, and is there a minimal "gating circuit" of ≤5 heads that accounts for >80% of the attention-mediated divergence?

**Hypothesis**: The 8.6× attention dominance at L22-24 is concentrated in a sparse subset of heads (likely 2-4 out of 32), not distributed uniformly. These heads will have distinct roles — some reading the instruction token, others reading the endorsement token — forming a two-stage "read instruction, gate evidence" circuit. This follows the pattern seen in induction heads and IOI circuits where attention-mediated computation is head-sparse even when the *layer* is identified.

**Method**:
1. **Model**: Qwen3-4B-Instruct (32 heads per layer, layers 22-24).
2. **Data**: The 151 resisting + 298 correcting items already extracted.
3. **Per-head attribution**: For each head at L22-24, compute the head's output projected through W_U (unembedding) at the last token position. Measure the signed margin contribution per head under i1a vs i1c. This gives a per-head "gating contribution" score.
4. **Attention pattern extraction**: For the top-5 contributing heads, extract full attention patterns. Classify each head as: (a) instruction-reading (high attention to instruction tokens under i1a, low under i1c), (b) endorsement-reading (high attention to "I think it's X" tokens), or (c) mixed.
5. **Ablation cascade**: Zero-ablate each head individually, then the top-k heads jointly (k=1..5). Measure flip rate on resisting items. Compare to full-layer patching baseline (8.6% flips).
6. **Nanda-style attribution patching**: Use gradient-based approximation (∂logit/∂head_output × head_output) to rank all 96 heads (32×3 layers) in a single backward pass, then validate top-5 with actual ablation.

**Connection to literature**: This is the "find the circuit" step that Neel Nanda's attribution patching (2023) was designed for. Connects to the IOI paper's methodology of identifying minimal sufficient head sets. Goes beyond the layer-level attn/MLP decomposition in C5.

**Impact if confirmed**: Transforms C5 from "attention mediates gating" (coarse) to "heads H22.7, H23.14, H24.3 form a gating circuit" (precise). This is the difference between a behavioral-level and a circuit-level mechanistic claim — exactly what NeurIPS reviewers in the interp track expect. If the circuit is sparse, it also opens the door to targeted fine-tuning interventions.

**Feasibility**: High. Qwen3-4B fits on 1× H100 with room for activation caching. Per-head decomposition requires 3 forward passes (one per layer, hooking individual heads) × 449 items = ~1,350 forward passes ≈ 20 minutes. Attribution patching (one backward pass per item) ≈ 30 minutes. Total: < 1 hour.

---

## Experiment 2: Manifold Curvature of the Gating Direction

**RQ**: Does the correction-gating direction lie on a curved belief manifold (Sarfati et al. 2026), and does this explain why linear additive patching fails at 20B+ scale while CAPPING (multiplicative/percentile-based) succeeds?

**Hypothesis**: The gating representation is linear in Qwen3-4B (low-dimensional, sharp probe peak) but sits on a curved manifold in GPT-oss-20B (distributed, high-dimensional). Linear patching pushes GPT-oss representations off-manifold, destroying downstream readout. CAPPING works because percentile-based scaling preserves the manifold's local geometry (it rescales along existing directions rather than adding an off-manifold displacement). If true, geometry-aware steering (Sarfati's method) should outperform both linear patching and CAPPING on GPT-oss.

**Method**:
1. **Model**: GPT-oss-20B (the "decodable but not steerable" case).
2. **Manifold estimation**: Collect gating activations at L8-10 (probe peak region) for all 810 items under i1a. Compute local PCA in neighborhoods of size k=50 (Euclidean nearest neighbors). Estimate intrinsic dimensionality via MLE (Levina-Bickel estimator) and local curvature via the rate of principal angle change between neighboring tangent spaces.
3. **On-manifold vs off-manifold test**: Project the linear patching vector onto the local tangent space at each item's activation. Measure the off-manifold component magnitude. Correlate off-manifold magnitude with patching failure (items where patching made things worse should have larger off-manifold components).
4. **Geometry-aware patching**: Implement Sarfati et al.'s geodesic steering: (a) project the patching vector onto the local tangent plane, (b) transport along the manifold using Riemannian exponential map approximation. Compare flip rates: linear patching vs CAPPING vs geodesic steering.
5. **Cross-model comparison**: Run the same analysis on Qwen3-4B where linear patching works. Predict: low curvature, tangent space stable, off-manifold component negligible.

**Connection to literature**: Directly tests Sarfati et al. 2026's core prediction ("standard linear steering pushes off-manifold") in a setting where we have a natural ablation — one model where linear works and one where it doesn't. Also connects to the activation scaling literature (EMNLP 2024): if CAPPING's success is due to manifold-preservation, multiplicative > additive is a consequence of geometry, not just scale.

**Impact if confirmed**: Solves the paper's central open question (Q2: why decodable-not-steerable?) with a geometric explanation grounded in recent theory. Turns a puzzling negative result into a positive mechanistic finding. Would be a strong NeurIPS contribution on its own — connecting evidence gating to representation geometry.

**Feasibility**: Medium-high. Manifold estimation (PCA + intrinsic dim) is CPU-only on pre-extracted activations. GPT-oss activations are already cached. Geodesic steering requires re-running forward passes with modified activations (~810 items × a few conditions = 2-3K forward passes on 20B model ≈ 1-1.5 hours on H100). Total: ~2 hours, tight but feasible.

---

## Experiment 3: SAE Decomposition of the Gating Direction

**RQ**: Can sparse autoencoders decompose the correction-gating direction into interpretable features, and do these features correspond to recognizable concepts (e.g., "speaker credibility assessment", "evidence-prior conflict detection", "instruction compliance")?

**Hypothesis**: The gating direction at L23 in Qwen3-4B is a superposition of 3-8 monosemantic features. At least one feature will be a "conflict detector" (active when endorsed answer ≠ model's prior), one will be an "instruction compliance" feature (active under i1a but not i1c), and one will be an "evidence weight" feature (modulating how strongly the endorsement shifts the answer). The interaction of these features produces the gating behavior — instruction compliance × conflict detection → evidence suppression.

**Method**:
1. **Model**: Qwen3-4B-Instruct, layer 23 residual stream.
2. **Train SAE**: Train a sparse autoencoder (k=4096 or 8192 features) on layer 23 activations at the endorsement position across all 449×2 item-condition pairs (i1a + i1c). Use the standard Anthropic SAE architecture (ReLU encoder, tied decoder, L1 sparsity). Training data: collect activations across a broader set of prompts (add the PIQA and DebateQA items for diversity, ~3K total activations).
3. **Project gating direction onto SAE features**: Decompose the correction-gating vector (mean_correcting − mean_resisting) in the SAE feature basis. Identify the top-10 features by coefficient magnitude.
4. **Feature interpretation**: For each top feature, find the top-20 maximally activating prompts. Hand-label the pattern. Use automated interpretability (generate descriptions via a larger model).
5. **Feature ablation**: For each top feature, zero its coefficient and reconstruct the activation. Re-run the forward pass from L23 onward with the ablated activation. Measure which individual features, when ablated, most reduce the gating effect (margin shift).
6. **Feature interaction**: Test whether the gating is additive (ablating features independently sums) or multiplicative (ablating one feature eliminates the contribution of others). This distinguishes "gating = sum of independent signals" from "gating = product of detectors."

**Connection to literature**: Follows Anthropic's SAE interpretability methodology (Bricken et al. 2023, Templeton et al. 2024). Goes beyond linear probing (which shows *that* information exists) to show *what* the information consists of. Connects to Bigelow et al.'s decomposition of beliefs into priors and evidence — SAE features might cleanly separate these.

**Impact if confirmed**: This is the "opening the black box" experiment that elevates the paper from "we found a direction" to "we found interpretable features that compose into a gating mechanism." If the features are clearly interpretable (conflict detection × instruction compliance), it provides a mechanistic explanation for WHY the gating exists, not just WHERE.

**Feasibility**: Medium. Training a small SAE (4096 features) on ~3K activations (each 2560-dim for Qwen3-4B) takes ~15 minutes on H100. Feature interpretation is the bottleneck — automated methods help but hand-labeling top-10 features is manual. Feature ablation requires ~4,500 forward passes (10 features × 449 items) ≈ 30 minutes. Total compute: ~1 hour. Interpretation time: additional 1-2 hours (partially manual).

---

## Experiment 4: The N0 Signal as Bayesian Prior (Bigelow Framework)

**RQ**: Is the N0 pre-endorsement signal a Bayesian prior in the sense of Bigelow et al. 2025 — i.e., does it behave additively in log-belief space with the endorsement evidence, and does it predict phase transitions where the model switches from resist to correct?

**Hypothesis**: The N0 residual (confidence-independent component) represents a prior susceptibility to evidence, not just confidence. In Bigelow's framework, the posterior belief = prior + evidence (in log-odds space). If the N0 residual is a prior, then: (a) the sum (N0_residual + endorsement_evidence_strength) should predict the actual posterior margin better than either alone, (b) there should be a sharp phase transition at the decision boundary where the sum crosses zero, and (c) activation steering should shift the prior (changing N0) while ICL should shift the evidence (changing endorsement strength), and these should be additively separable.

**Method**:
1. **Model**: Qwen3-4B-Instruct + GPT-oss-20B (both have strong N0 signals).
2. **Quantify evidence strength**: For each item, define evidence_strength = m_C1 − m_N0 (the logit shift caused by the endorsement). Define prior = N0 probe residual score (confidence-regressed-out).
3. **Additivity test**: Regress actual_posterior_margin ~ prior + evidence_strength. If Bigelow's framework holds, R² should be high and the coefficients should be roughly equal (both in log-odds space). Test for interaction: prior × evidence_strength should be near-zero if additive.
4. **Phase transition**: Bin items by (prior + evidence_strength), look for a sigmoid transition in P(correct). If the transition is sharp (steep sigmoid), the model implements approximate Bayesian integration. If gradual, the computation is sub-Bayesian.
5. **Steering vs ICL separation**: (a) Patch the gating direction at L23 (= shift the "prior" per Bigelow). Measure whether evidence_strength is preserved (additivity prediction). (b) Change the endorsement wording strength ("I think" → "I'm certain" → "Studies prove"). Measure whether prior (N0 residual) is preserved. If both hold, the prior/evidence decomposition is real.

**Connection to literature**: Directly tests Bigelow et al. 2025's core prediction in our setting. They studied generic ICL and steering; we test it for instruction-dependent evidence gating specifically. Our setting is ideal because we have independent measures of both prior (N0) and evidence (endorsement effect).

**Impact if confirmed**: Provides a formal theoretical framework (Bayesian belief updating) for the empirical gating phenomenon. Connects our mechanistic findings to a normative theory. Makes the paper theoretically grounded, not just empirically surprising. If the phase transition exists, it also provides a precise prediction for which items will resist vs correct — moving from statistical regularity to mechanistic law.

**Feasibility**: High. All data already exists (N0 residuals, margins under all conditions). The regression and sigmoid fitting are CPU-only analyses on existing results. The steering/ICL separation test requires ~900 forward passes (449 items × 2 conditions) ≈ 10 minutes. Total: < 30 minutes compute + 1 hour analysis.

---

## Experiment 5: Dynamic Scaling Factors for Instruction-Conditional Steering

**RQ**: Can we learn item-specific scaling factors (à la DSAS, arXiv 2512.03661) that make the gating vector steerable even in models where fixed-alpha patching fails (GPT-oss-20B, OLMo-2-32B)?

**Hypothesis**: Fixed-alpha patching fails on GPT-oss and OLMo because the optimal steering magnitude varies per-item — high-confidence-wrong items need stronger push, borderline items need delicate nudges. A context-dependent scaling factor (predicted from the item's own activation pattern) will outperform any fixed alpha. This scaling factor IS the model's internal "evidence precision weight," connecting to the precision-weighting finding (Section 10.5) and DSAS.

**Method**:
1. **Model**: GPT-oss-20B (the primary "decodable not steerable" case).
2. **Learn dynamic alpha**: Train a small MLP (2 layers, 64 hidden) that takes the item's L8 activation (pre-patching) and predicts the optimal alpha for the gating vector. Target: the alpha that maximizes margin shift toward correct. Train on 80% of items, test on 20%.
3. **Feature importance on alpha**: What activation features predict optimal alpha? Regress optimal_alpha ~ confidence + N0_residual + endorsement_strength. Hypothesis: alpha should scale with confidence (harder items need more steering) and inversely with N0_residual (items already predisposed to correct need less).
4. **Apply dynamic steering**: Re-run patching on the test set with predicted alphas. Compare: (a) fixed alpha=4 (current best), (b) learned dynamic alpha, (c) oracle alpha (exhaustive sweep per item, upper bound).
5. **Cross-model transfer**: Train alpha predictor on Qwen3-4B (where we have good signal), apply to GPT-oss. If the scaling rule transfers, the precision-weighting is universal.

**Connection to literature**: Directly applies DSAS (arXiv 2512.03661) to our setting. Their key insight — decouple WHEN to steer from HOW MUCH to steer — maps perfectly onto our observation that the gating direction is correct but the magnitude needs to be item-specific. Also connects to activation scaling (EMNLP 2024): if multiplicative > additive, then dynamic scaling should be even better because it adapts the multiplier.

**Impact if confirmed**: Converts GPT-oss from "not steerable" to "steerable with the right scaling," dramatically strengthening the universality claim. Provides a prescriptive contribution: not just diagnosing the problem but offering a solution. If the alpha predictor is interpretable (e.g., alpha ≈ f(confidence)), it connects mechanism to application.

**Feasibility**: High. Training an MLP on 810 items × ~650 alpha evaluations each (sweep alpha in [0, 8] at 0.1 increments) requires 810 × 80 = 64,800 forward passes. At ~1.5 sec/pass on GPT-oss-20B, that's ~27 hours — too slow for exhaustive sweep. **Optimization**: Use gradient-based alpha optimization (single backward pass per item to find optimal alpha) ≈ 810 × 2 passes = ~40 minutes. MLP training: < 1 minute (CPU). Total: ~1 hour.

---

## Experiment 6: Instruction Token Causal Tracing

**RQ**: Which tokens in the instruction string causally drive the gating divergence? Specifically, does "even if the speaker is wrong" (the adversarial clause in i1a) vs "facts, not the speaker's opinion" (i1c) have localized causal effects that can be traced to specific attention head interactions at L22-24?

**Hypothesis**: The adversarial clause in i1a ("even if the speaker is wrong") triggers a specific "conflict frame" that activates the gating circuit. The gating heads at L22-24 attend disproportionately to the word "wrong" in i1a and to "facts" in i1c. Token-level causal tracing will show that corrupting just 1-2 tokens in each instruction eliminates the divergence, identifying the precise lexical trigger for the gating mode switch.

**Method**:
1. **Model**: Qwen3-4B-Instruct.
2. **Token-level corruption**: For i1a, systematically corrupt (replace with padding or random token) each token in the instruction while keeping everything else fixed. Measure the change in gating behavior (margin shift on resisting items). Identify which tokens, when corrupted, most reduce the i1a-specific entrenchment.
3. **Attention-to-instruction analysis**: For the gating heads identified in Experiment 1, extract attention weights from the endorsement position back to each instruction token. Compare i1a vs i1c: which instruction tokens receive differential attention?
4. **Minimal instruction ablation**: Construct chimeric instructions: i1a with the adversarial clause swapped for i1c's "facts" clause, and vice versa. This isolates the causal contribution of each clause to the gating behavior. (Partially done with i1d; this extends it to finer-grained swaps.)
5. **Cross-reference with wording gradient**: Map the diagnostic sweep results (Section 2.2 of results) onto the token-level analysis. Do instructions that score higher on the "pathological" end attend more to conflict-framing tokens?

**Connection to literature**: Extends the classic causal tracing methodology (Meng et al. 2022, ROME) from factual recall to instruction processing. Connects to the i1d ablation already done (Section 2.3 of findings.md) but at the mechanistic level — showing not just that the adversarial clause matters behaviorally, but HOW it matters in the circuit.

**Impact if confirmed**: Provides the "why does wording matter?" answer at the mechanistic level. The behavioral finding (wording gradient) is already strong; this adds the circuit-level explanation. Also has practical implications: if specific tokens trigger pathological gating, instruction designers can avoid those triggers.

**Feasibility**: High. Token corruption requires ~15 tokens × 449 items × 2 instructions = ~13,500 forward passes ≈ 20 minutes on Qwen3-4B. Attention extraction is free (single forward pass with hooks). Total: < 45 minutes.

---

## Experiment 7: Gating Direction in Log-Belief Space

**RQ**: Is the correction-gating direction linear in the *logit space* (log-odds of correct answer) rather than in activation space, and does transforming to log-belief space improve probe accuracy, patching effectiveness, and cross-model transfer?

**Hypothesis**: Bigelow et al. predict that beliefs are additive in log-odds space. Our probes operate in raw activation space. If the model's internal "belief" representation is in log-odds (which the logit lens results suggest — margins are logit differences), then: (a) probing in log-belief-transformed space should yield higher AUROC, (b) patching vectors computed in log-belief space should be more effective, and (c) cross-model transfer should improve because log-belief coordinates are scale-invariant.

**Method**:
1. **Transform activations**: For each item, apply the unembedding projection (W_U) to get logit-space activations at each layer. Compute log-odds: log(P(correct)/P(wrong)). This gives a scalar per layer.
2. **Log-belief probing**: Train probes on log-belief-transformed activations rather than raw hidden states. Compare AUROC to standard probes.
3. **Log-belief patching**: Compute the gating vector in log-belief space (difference of means in log-odds). Apply via: (a) transform to logit space, (b) add the log-belief-space vector, (c) project back to activation space (pseudo-inverse of W_U). Compare to standard activation-space patching.
4. **Cross-model transfer in log-belief space**: Transfer the log-belief gating vector from Qwen to GPT-oss (after aligning embedding dimensions via CCA or Procrustes). Log-belief vectors should transfer better than activation vectors because they're in a model-independent coordinate system.

**Connection to literature**: Tests Bigelow et al.'s "additivity in log-belief space" prediction mechanistically. Also relates to the logit lens analysis (C4) — the logit lens IS a log-belief projection, so this experiment formalizes what the logit lens already suggests.

**Impact if confirmed**: Provides a principled coordinate system for belief representations. If log-belief probing and patching are uniformly better, it's a methodological contribution applicable to any activation steering work. The cross-model transfer result would be particularly impactful: a universal "belief space" that transcends architecture.

**Feasibility**: High. Logit lens projections already implemented (`logit_lens_analysis.py`). The additional computation is negligible (matrix multiplication + log-softmax). Probing/patching uses existing infrastructure. Total: < 1 hour.

---

## Experiment 8: Temporal Evolution of the Gating Decision

**RQ**: When during the forward pass does the model "commit" to resisting vs correcting, and is this commitment point different under i1a vs i1c? Can we identify a "point of no return" after which the gating decision is irreversible?

**Hypothesis**: Under i1a, the commitment happens early (around L22-24, the gating region), and once made, downstream layers amplify the decision (consistent with C4 prevention finding). Under i1c, the commitment is later and more gradual (evidence continues to be integrated through later layers). The "point of no return" should be earlier for resisting items than correcting items — the model decides to resist faster than it decides to correct.

**Method**:
1. **Model**: Qwen3-4B-Instruct.
2. **Layer-by-layer decision readout**: At each layer L, read out the model's "current answer" (argmax of logits via logit lens). Track when the answer first becomes correct (for correcting items) or first locks into wrong (for resisting items).
3. **Commitment entropy**: At each layer, compute the entropy of the softmax over A/B logits (via logit lens). Low entropy = committed. Plot entropy trajectories for correcting vs resisting items under i1a vs i1c. Identify the layer where entropy drops below a threshold.
4. **Intervention timing**: Patch the gating vector at different layers (L18, L20, L22, L24, L26, L28). Measure flip rate at each intervention point. Hypothesis: earlier intervention catches more items (before the "point of no return"). Late intervention catches only borderline cases.
5. **Information-theoretic measure**: Compute the mutual information I(layer_L_activation ; final_answer) at each layer. The layer where MI saturates is the commitment point.

**Connection to literature**: Extends the logit lens analysis (C4) from a static snapshot to a dynamic trajectory analysis. Connects to the "prevention not suppression" finding — if prevention means "never forms," the commitment point should coincide with the prevention onset (~L22). Relates to Truthfulness Spectrum Hypothesis (arXiv 2602.20273): the spectrum might be traversed layer-by-layer during the forward pass.

**Impact if confirmed**: Adds temporal/dynamic mechanistic depth. Shows not just where gating happens but WHEN the decision is made and why it's hard to reverse. The "point of no return" concept is vivid and memorable — good for NeurIPS presentation. Also prescriptive: interventions must target layers before the commitment point.

**Feasibility**: High. Logit lens already computed; this is mostly analysis on existing data. The intervention timing sweep requires 6 layers × 449 items × 4 conditions (i1a/i1c × real/control) = ~10,800 forward passes ≈ 15 minutes. Total: < 30 minutes compute + analysis.

---

## Experiment 9: Cross-Architecture Circuit Correspondence

**RQ**: Despite different steerability, do Qwen3-4B and GPT-oss-20B use *representationally similar* gating circuits that differ only in distribution width? Or are the circuits qualitatively different?

**Hypothesis**: The gating representations are linearly mappable between models (consistent with strong probe transfer AUROCs: 0.926 cross-condition). The difference is quantitative: Qwen concentrates gating in 2-3 layers (sharp circuit), GPT-oss distributes it across 10+ layers (diffuse circuit). But the underlying linear subspace is shared — a rotation matrix can map Qwen's L23 gating subspace to GPT-oss's L8-10 gating subspace. This explains why probes transfer but patching doesn't: the representation is the same, but the computational circuit that reads it differs in depth.

**Method**:
1. **Models**: Qwen3-4B (L23 activations) and GPT-oss-20B (L8 activations). Use the overlapping items (187 shared prior-wrong items).
2. **CKA alignment**: Compute centered kernel alignment (CKA) between Qwen L23 and GPT-oss L_k for each GPT-oss layer k. Identify which GPT-oss layer is most representationally similar to Qwen's gating layer.
3. **Procrustes alignment**: Find the optimal orthogonal transformation R mapping Qwen L23 activations to GPT-oss L_best activations (on shared items). Measure alignment quality (residual norm after rotation). If high alignment: the gating subspace is architecturally universal.
4. **Transferred patching**: Take Qwen's gating vector, rotate it through R into GPT-oss's activation space. Patch this rotated vector into GPT-oss. If representation is shared but circuit differs, this should: (a) improve probe accuracy on GPT-oss (confirm alignment), but (b) NOT improve patching (circuit reads representation differently).
5. **Circuit depth comparison**: Compute the "effective circuit depth" — number of layers where gating information changes by >1% (measured by patching each layer and measuring the marginal contribution). Plot this against model size for all 7 models. Hypothesis: effective depth scales with model size.

**Connection to literature**: Builds on representation similarity analysis (CKA, Kornblith et al. 2019) and the "Platonic Representation Hypothesis" (Huh et al. 2024). Tests whether the gating direction is a universal computation (model-independent) implemented in model-dependent circuits.

**Impact if confirmed**: The "same representation, different circuit" finding would be a clean, quotable result. It explains the decodable-not-steerable puzzle architecturally and predicts which future models will be steerable (those with concentrated circuits). Has implications for transfer of steering vectors across model families.

**Feasibility**: Medium. CKA and Procrustes are CPU-only on pre-extracted activations. The transferred patching requires ~187 forward passes on GPT-oss-20B ≈ 5 minutes. Circuit depth comparison requires existing patching data (already available for Qwen and GPT-oss; would need brief runs for other models). Total: ~1 hour.

---

## Experiment 10: Epistemic Observability Under Gating

**RQ**: When the gating mechanism prevents the correct answer (C4), does the model's internal confidence representation "know" it's wrong, or is the prevention total — extending to the model's own uncertainty estimates?

**Hypothesis**: Drawing on "Epistemic Observability" (arXiv 2603.20531), models report higher confidence when wrong under certain instructions. Under i1a, resisting items should show: (a) high reported confidence ("I'm sure it's X") even though the answer is wrong, (b) BUT the internal activation-level confidence (e.g., softmax entropy at intermediate layers) may still reflect uncertainty. The gating prevents the correct *answer* but not the *uncertainty signal* — the model "knows it doesn't know" internally even as it produces a confident wrong answer externally.

**Method**:
1. **Model**: Qwen3-4B-Instruct (Thinking variant for verbalized confidence).
2. **Internal vs external confidence**: For resisting items under i1a:
   - External confidence: Softmax probability of chosen answer at final layer.
   - Internal confidence: Entropy of the logit distribution at each intermediate layer (via logit lens). Specifically, at the gating layers (L22-24), measure whether entropy spikes even though the final answer is confidently wrong.
3. **Confidence calibration under gating**: Compare calibration curves (binned accuracy vs confidence) for gated items (resisting under i1a) vs non-gated items (correcting under i1c). Prediction: gated items are systematically overconfident (high confidence, low accuracy = classic miscalibration).
4. **Hidden uncertainty detection**: Train a probe on intermediate-layer activations to predict "model is wrong despite being confident." If the model internally represents its uncertainty, this probe should succeed. Layer sweep to find where the "I'm secretly uncertain" signal is strongest — predict it's BEFORE the gating layers (L18-22), getting suppressed by the gating.
5. **Verbalized vs actual confidence**: On the Thinking variant, compare verbalized confidence in chain-of-thought ("I'm fairly sure...") with actual softmax confidence. Does the thinking model verbalize the hidden uncertainty that the non-thinking model suppresses?

**Connection to literature**: Directly tests Epistemic Observability (arXiv 2603.20531) in our controlled setting. Their finding (models report higher confidence when wrong under instructions) aligns with our C1 (instructions make errors worse). We can test whether this is total suppression or just output-level masking.

**Impact if confirmed**: A "the model knows it's wrong" finding would be dramatic — it means the gating mechanism is an output-level policy, not a representational failure. This has major implications for AI safety (models that "lie" about their confidence) and for practical intervention (we can detect hidden uncertainty and override the gating). Highly citable.

**Feasibility**: High. Logit lens data already exists. Internal entropy computation is trivial. Probe training: same infrastructure as existing probes. Thinking model traces: 117 items already collected (Section 9). Total: < 45 minutes compute.

---

## Experiment 11: Persona × Gating Interaction Geometry

**RQ**: How does persona conditioning ("You are a cautious scientist" vs "You are a helpful assistant") interact with the correction-gating direction? Do personas shift along the gating direction, orthogonally to it, or do they change the gating direction itself?

**Hypothesis**: Personas operate in a low-dimensional subspace that intersects but does not subsume the gating direction. "Cautious" personas will shift representations toward the resist end of the gating direction (amplifying i1a's effect), "helpful" personas toward the correct end (amplifying i1c's effect). But the gating direction itself will be stable across personas — personas modulate WHERE on the gating axis the model starts, not the axis itself. This connects to Bigelow's prior-shifting interpretation: personas change the prior, instructions change the evidence weighting.

**Method**:
1. **Model**: Qwen3-4B-Instruct (persona sweep already running: 6 personas × 2 instructions).
2. **Data**: Awaiting persona sweep completion; design for post-hoc analysis.
3. **Persona axis extraction**: Compute mean activations at L23 (endorsement position) for each persona. Extract persona axes via PCA on the 6 persona centroids. Measure dimensionality of persona variation.
4. **Gating × persona geometry**: Project all activations onto 2D: (gating direction, top persona PC). Visualize. Measure: (a) cosine angle between gating direction and persona PCs, (b) whether gating direction is preserved within each persona (train probe within-persona, test cross-persona).
5. **Persona as prior shift**: In Bigelow's framework, personas should shift the prior but not the evidence weight. Test: (a) Does persona shift the N0 residual? (b) Does the C1-N0 margin (evidence effect) change across personas? If personas shift N0 but not (C1-N0), they're pure prior shifts.
6. **Interaction test**: Fit a linear model: margin ~ gating_score + persona_score + gating×persona. If the interaction term is significant, personas modulate gating (not just add to it).

**Connection to literature**: Connects to persona research broadly, but specifically tests whether persona effects are mechanistically independent of evidence-gating or interact with it. Bigelow's prior/evidence decomposition provides the theoretical framework.

**Impact if confirmed**: Resolves core open question Q6 (persona × gating interaction). If personas are pure prior shifts, it validates the Bayesian framework. If they interact with gating, it reveals a more complex picture where identity conditioning changes the evidence-integration circuit itself. Either result is interesting and publishable.

**Feasibility**: High once persona sweep completes. All analysis is on pre-extracted activations (CPU-only). Cross-persona probing reuses existing infrastructure. Total: < 30 minutes analysis.

---

## Experiment 12: The Multiplicative Gating Hypothesis

**RQ**: Is the gating mechanism truly multiplicative (as suggested by CAPPING > additive patching in GPT-oss), and can we test this by decomposing the gating into a binary gate (on/off) times a continuous magnitude (how much evidence to integrate)?

**Hypothesis**: The gating circuit implements something like: `output = gate(instruction) × evidence_signal`, where `gate()` is a soft binary function of the instruction (near-0 for i1a on resisting items, near-1 for i1c) and `evidence_signal` is the endorsement's strength. This multiplicative structure explains: (a) why CAPPING (which is scale-preserving) works better than additive patching (which disrupts the gate's scale), (b) why the gating is evidence-specific (the vector modulates evidence integration, not answer selection, per Section 4.4), and (c) why prevention ≠ suppression (gate=0 means the signal was never multiplied in, not that it was subtracted out).

**Method**:
1. **Model**: Qwen3-4B-Instruct.
2. **Gate extraction**: At L22-24, decompose the attention output into: (a) a component that depends on instruction (gate candidate), and (b) a component that depends on endorsement content (evidence candidate). Use the i1a/i1c contrast to isolate instruction-dependent variation, and the C0/C1/W0/W1 contrast to isolate endorsement-dependent variation.
3. **Multiplicativity test**: If the gate is multiplicative, then: instruction_effect × endorsement_effect should predict the full interaction. Concretely: (i1a_attn − i1c_attn) should scale proportionally with endorsement strength across items. Regress: actual_divergence ~ instruction_component × evidence_component. If R² > 0.8, the multiplicative model fits.
4. **Gate visualization**: Compute the effective gate value per item (ratio of attention output under i1a to attention output under i1c, projected onto the gating direction). Plot the distribution. Prediction: bimodal (gate ≈ 0 for resisting items, gate ≈ 1 for correcting items) under i1a.
5. **Capping replication**: Use the multiplicative decomposition to predict which items CAPPING should help most (those where the gate is intermediate, between 0 and 1). Verify on GPT-oss-20B data.

**Connection to literature**: Directly tests the activation scaling finding (EMNLP 2024) that multiplicative > additive. Provides a mechanistic explanation for WHY multiplicative works: the circuit IS multiplicative. Also connects to the CAPPING result on GPT-oss (CAPPING outperforms additive by +0.083 vs +0.066) and gives a principled reason.

**Impact if confirmed**: A clean "gating is multiplicative" finding unifies multiple observations (CAPPING > additive, evidence specificity, prevention not suppression) under a single computational motif. This is the kind of elegant mechanistic finding that NeurIPS loves — one principle explaining multiple phenomena. Also prescriptive: future steering should use multiplicative interventions.

**Feasibility**: High. Decomposition uses existing attention hook infrastructure. Requires ~449 × 4 conditions = ~1,800 forward passes with attention output caching ≈ 5 minutes on Qwen3-4B. Regression analysis is CPU-only. Total: < 30 minutes.

---

## Priority Ranking

For maximum NeurIPS impact, ordered by (impact × feasibility × novelty):

| Rank | Experiment | Hours (H100) | Key Contribution |
|------|-----------|-------------|-------------------|
| 1 | **Exp 1: Gating Head Surgery** | 1.0 | Circuit-level precision (heads, not just layers) |
| 2 | **Exp 8: Temporal Commitment** | 0.5 | "Point of no return" dynamics |
| 3 | **Exp 12: Multiplicative Gate** | 0.5 | Unifying computational motif |
| 4 | **Exp 4: N0 as Bayesian Prior** | 0.5 | Formal theoretical grounding |
| 5 | **Exp 10: Epistemic Observability** | 0.75 | "Model knows it's wrong" — safety implications |
| 6 | **Exp 3: SAE Decomposition** | 1.5 | Interpretable feature composition |
| 7 | **Exp 6: Instruction Token Tracing** | 0.75 | Lexical trigger for gating mode |
| 8 | **Exp 2: Manifold Curvature** | 2.0 | Geometric explanation for scale failure |
| 9 | **Exp 11: Persona × Gating** | 0.5 | Persona interaction geometry |
| 10 | **Exp 9: Cross-Architecture Correspondence** | 1.0 | "Same representation, different circuit" |
| 11 | **Exp 5: Dynamic Scaling** | 1.0 | Prescriptive (steering fix) |
| 12 | **Exp 7: Log-Belief Space** | 1.0 | Methodological contribution |

**Recommended package for paper**: Experiments 1, 8, 12, 4, and 10 form a coherent story: "Here is the circuit (1), here is when it commits (8), here is the computational motif (12), here is the theoretical framework (4), and here is the safety implication (10)." Total compute: ~3.25 hours on 1× H100.

---

## Implementation Notes

- **Shared infrastructure**: All experiments use the existing `ModelHookRunner` in `src/mechanism/hooks.py` for collection and intervention. Experiments 1 and 6 require per-head hooks (extend `collect()` to return per-head attention outputs). Experiments 3 and 7 require SAE training (use `sae-lens` or minimal custom implementation).
- **Data reuse**: Experiments 2, 4, 7, 8, 9, 11 can run entirely on pre-extracted activations with no new GPU forward passes.
- **Key model files**: `src/mechanism/attn_mlp_decomposition.py` (extend for per-head), `src/mechanism/logit_lens_analysis.py` (extend for entropy), `src/mechanism/run_correction_gating_patching.py` (extend for timing sweep).
