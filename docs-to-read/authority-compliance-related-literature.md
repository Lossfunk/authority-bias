# Authority-Compliance / Sycophancy / Deference Literature

Working list of papers/posts relevant to the current mechanism story:
- authority-conditioned compliance
- sycophancy / deference / conformity
- activation steering / mechanistic control
- persona / affect / eval-awareness confounds

Each item includes a one-line note on why it matters for our paper.

## Core directly relevant papers

1. **The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models**  
   Link: https://arxiv.org/pdf/2601.10387  
   Why relevant: Closest deconfound for the “this is just generic assistant/persona drift” alternative; directly motivates our assistant-axis overlap test.

2. **Emotion Concepts and their Function in a Large Language Model**  
   Link: https://transformer-circuits.pub/2026/emotions/index.html  
   Why relevant: Strong evidence that interpretable affect-like concepts exist and have behavioral effects; useful backdrop for asking whether our mechanism is reducible to emotional structure.

3. **NRC VAD Lexicon v2: Norms for Valence, Arousal, and Dominance for over 55k English Terms**  
   Link: https://arxiv.org/abs/2503.23547  
   Why relevant: Human-labeled lexical affect resource underlying the cheap VA deconfound we ran.

4. **Valence–Arousal Subspaces in LLMs**  
   Link: https://arxiv.org/abs/2604.03147  
   Why relevant: Most directly relevant prior on low-dimensional affective subspaces; provides the principled template for a cheap VA overlap check without full replication.

5. **Steering Language Models With Activation Engineering**  
   Link: https://arxiv.org/abs/2308.10248  
   Why relevant: Canonical activation-steering reference; useful for situating our prompt-time intervention methodology.

6. **Refusal in Language Models Is Mediated by a Single Direction**  
   Link: https://arxiv.org/abs/2406.11717  
   Why relevant: Strong precedent for a compact behavior-mediating direction and for intervention + erasure methodology; useful comparison point for our compliance direction.

7. **When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models**  
   Link: https://arxiv.org/abs/2508.02087  
   Why relevant: Probably the single most directly relevant mechanistic sycophancy paper; especially important because it studies internal overrides of truth, though it reportedly finds authority framing weak in their setup.

8. **There Is More to Refusal in Large Language Models than a Single Direction**  
   Link: https://arxiv.org/abs/2602.02132  
   Why relevant: Important counterweight to single-direction stories; useful when discussing whether our mechanism is truly 1D versus part of a richer subspace/state pattern.

9. **Towards Understanding Sycophancy in Language Models**  
   Link: https://www.anthropic.com/research/towards-understanding-sycophancy-in-language-models  
   Why relevant: Early foundational paper tying sycophancy to RLHF/preference data; important background for why these behaviors may arise at all.

## Social compliance / deference / authority hypotheses

10. **Human-like Social Compliance in Large Language Models: Unifying Sycophancy and Conformity through Signal Competition Dynamics**  
    Link: https://arxiv.org/abs/2601.11563  
    Why relevant: Best conceptual match to our current mechanism story; frames social cues as competing with factual/evidential signals.

11. **DialDefer: A Framework for Detecting and Mitigating LLM Dialogic Deference**  
    Link: https://arxiv.org/abs/2601.10896  
    Why relevant: Useful for framing our phenomenon as deference/compliance rather than only generic sycophancy.

12. **Status Hierarchies in Language Models**  
    Link: https://arxiv.org/abs/2601.17577  
    Why relevant: Relevant to the hypothesis that authority framing works by triggering internal status/deference priors.

13. **Verbalizing LLMs' assumptions to explain and control sycophancy**  
    Link: https://arxiv.org/abs/2604.03058  
    Why relevant: Suggests sycophancy can be mediated by latent assumptions about the user/speaker, which may connect to our authority-conditioned mechanism.

14. **How RLHF Amplifies Sycophancy**  
    Link: https://arxiv.org/abs/2602.01002  
    Why relevant: Useful training-side explanation for why deference/compliance tendencies emerge or are amplified.

15. **Who is In Charge? Dissecting Role Conflicts in Instruction Following**  
    Link: https://arxiv.org/abs/2510.01228  
    Why relevant: Directly relevant to conflicts between instruction hierarchy and socially salient cues such as authority, expertise, and role.

16. **How Reasoning Mitigates (Yet Masks) LLM Sycophancy**  
    Link: https://arxiv.org/abs/2603.16643  
    Why relevant: Relevant to how explicit reasoning interacts with sycophancy and whether CoT masks versus resolves underlying compliant behavior.

17. **SycEval: Evaluating LLM Sycophancy**  
    Link: https://arxiv.org/abs/2502.08177  
    Why relevant: One of the earlier focused evaluation frameworks for sycophancy; useful as a benchmark/measurement reference and for distinguishing progressive vs regressive sycophancy.

18. **ELEPHANT: Measuring and understanding social sycophancy in LLMs**  
    Link: https://arxiv.org/abs/2505.13995  
    Why relevant: Broadens sycophancy beyond factual agreement into social face-preservation, which is highly relevant when framing authority compliance as a richer social phenomenon.

19. **Measuring Sycophancy of Language Models in Multi-turn Dialogues (SYCON BENCH)**  
    Link: http://www.aclanthology.org/2025.findings-emnlp.121/  
    Why relevant: Strong benchmark precedent for multi-turn/free-form sycophancy measurement rather than only single-turn forced-choice settings.

20. **Conformity in Large Language Models**  
    Link: https://aclanthology.org/2025.acl-long.195/  
    Why relevant: Useful for tying our results to the broader conformity literature rather than only “agreement with the user”.

21. **Beacon: Single-Turn Diagnosis and Mitigation of Latent Sycophancy in Large Language Models**  
    Link: https://arxiv.org/abs/2510.16727  
    Why relevant: Useful benchmark/method paper that decomposes latent sycophancy into sub-biases and studies activation-level mitigation.

22. **Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs**  
    Link: https://arxiv.org/abs/2509.21305  
    Why relevant: Very relevant to our discussion of whether “compliance” is a single mechanism or one member of a family of separable sycophantic/deferential processes.

23. **PARROT: Persuasion and Agreement Robustness Rating of Output Truth -- A Sycophancy Robustness Benchmark for LLMs**  
    Link: https://arxiv.org/abs/2511.17220  
    Why relevant: Especially relevant because it explicitly studies authority/persuasion pressure and accuracy degradation under social pressure.

24. **Diagnosing and Mitigating Sycophancy and Skepticism in LLM Causal Judgment**  
    Link: https://arxiv.org/abs/2601.08258  
    Why relevant: Useful for showing that social pressure can cause models to abandon otherwise sound reasoning, which is conceptually close to our authority-induced factual override story.

25. **Ask don't tell: Reducing sycophancy in large language models**  
    Link: https://arxiv.org/abs/2602.23971  
    Why relevant: Strong framing paper on how input form, epistemic certainty, and first-person framing modulate sycophancy; useful for discussing why some authority framings may be especially potent.

26. **Not Your Typical Sycophant: The Elusive Nature of Sycophancy in Large Language Models**  
    Link: https://arxiv.org/abs/2601.15436  
    Why relevant: Useful as a caution that different prompt formats and social costs can substantially change apparent sycophancy, so the phenomenon is not exhausted by one benchmark setup.

## Affect / emotion / reasoning-adjacent papers

27. **Can Aha Moments Be Faked?**  
    Link: https://arxiv.org/pdf/2510.24941  
    Why relevant: Potentially useful when discussing whether verbalized reasoning about compliance is faithful or performative.

28. **Emotions, Where Art Thou?**  
    Link: https://openreview.net/pdf/d1281398c3aef599f43ff39077b01321779c6405.pdf  
    Why relevant: Another angle on whether emotion-like structure in LLMs is behaviorally real, representationally localized, or mostly surface-level.

## Eval-awareness / performative alignment / measurement-adjacent

29. **How eval awareness might emerge in training**  
    Link: https://www.lesswrong.com/posts/uRs5ebXKYLQyvJa2Q/how-eval-awareness-might-emerge-in-training-1  
    Why relevant: Good conceptual background on how evaluation-awareness-like capabilities may arise through training, even if not directly about authority.

30. **Sonnet 4.5's eval gaming seriously undermines alignment evals, and this seems caused by training on alignment evals**  
    Link: https://www.lesswrong.com/posts/qgehQxiTXj53X49mM/sonnet-4-5-s-eval-gaming-seriously-undermines-alignment  
    Why relevant: Important for distinguishing genuine alignment from performative “behave well when watched” effects; adjacent to our concern that compliance may be context-conditioned.

31. **Call for Science of Eval Awareness (+ Research Directions)**  
    Link: https://www.lesswrong.com/posts/tn8nKcNE4SDnDxLJj/call-for-science-of-eval-awareness-research-directions  
    Why relevant: Good survey of open research directions on evaluation awareness, measurement, feature suppression, and training-data attribution.

32. **Realistic Evaluations Will Not Prevent Evaluation Awareness**  
    Link: https://www.lesswrong.com/posts/7qBTcE3jqQFTuzssE/realistic-evaluations-will-not-prevent-evaluation-awareness  
    Why relevant: Helps frame why “realistic/free-generation” settings are useful but not sufficient, and why mechanism-level evidence matters.

## Emergent misalignment / latent behavioral structure

33. **Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs**  
    Link: https://arxiv.org/abs/2502.17424  
    Why relevant: Shows that narrow fine-tuning on one harmful behavior can cause broad misalignment elsewhere, suggesting models have latent behavioral structure that connects seemingly unrelated failure modes. Directly relevant to our finding that a compliance mechanism extracted from one task transfers to another: if authority-compliance is part of a broader latent behavioral manifold, narrow activation patching could shift the model along it, just as narrow fine-tuning does in this paper.

34. **Emergent Misalignment is Easy, Narrow Misalignment is Hard**  
    Link: https://arxiv.org/abs/2602.07852  
    Why relevant: Follow-up showing that emergent misalignment is reproducible and that narrowly targeted misalignment is harder to achieve than broad behavioral shifts. Supports the interpretation that our compliance mechanism may tap into a general deference/compliance mode rather than a task-specific artifact.

## Possibly relevant but currently unclear / lower priority

35. **Mitigating Goal Misgeneralization via Minimax Regret**  
    Link: https://arxiv.org/abs/2507.03068  
    Why relevant: Currently unclear for this project; included here because it was on our running list, but it does not appear directly about sycophancy/authority/deference.

## Additional papers surfaced in second-pass search

36. **Trust Me, I'm an Expert: Decoding and Steering Authority Bias in Large Language Models**  
    Link: https://arxiv.org/abs/2601.13433  
    Why relevant: Closest direct overlap risk discovered in the second-pass search; explicitly studies authority bias, varies source expertise, reports accuracy/confidence degradation under misleading expert endorsement, and claims a mechanistic steering result.

37. **Sycophancy Hides Linearly in the Attention Heads**  
    Link: https://aclanthology.org/2026.eacl-long.324/  
    Why relevant: Important mechanistic sycophancy paper showing linearly accessible signals in attention heads, transfer from TruthfulQA to other factual QA tasks, and limited overlap with “truthful” directions; highly relevant comparison point for our transfer and separability claims.

38. **Sycophantic Anchors: Localizing and Quantifying User Agreement in Reasoning Models**  
    Link: https://arxiv.org/abs/2601.21183  
    Why relevant: Strong support for the idea that sycophancy can build during the reasoning trajectory rather than being fixed entirely at the prompt; useful when discussing whether our effect is a dynamic evidence-weighting process versus a last-token flip.

39. **MONICA: Real-Time Monitoring and Calibration of Chain-of-Thought Sycophancy in Large Reasoning Models**  
    Link: https://arxiv.org/abs/2511.06419  
    Why relevant: Relevant follow-up on step-level monitoring and intervention during reasoning; useful for connecting our prompt-side mechanism story to later trajectory-level detection and calibration work.

40. **From Yes-Men to Truth-Tellers: Addressing Sycophancy in Large Language Models with Pinpoint Tuning**  
    Link: https://proceedings.mlr.press/v235/chen24u.html  
    Why relevant: Important earlier paper showing that a small subset of modules/heads disproportionately affects sycophancy; useful precedent that the behavior is mechanistically localized enough to target.

41. **Linear Probe Penalties Reduce LLM Sycophancy**  
    Link: https://arxiv.org/abs/2412.00967  
    Why relevant: Relevant reward-model-side mechanistic mitigation paper; useful for the broader story that sycophancy leaves detectable internal markers that can be penalized or steered against.

42. **Causally Motivated Sycophancy Mitigation for Large Language Models**  
    Link: https://openreview.net/forum?id=yRKelogz5i  
    Why relevant: Useful causal-framing paper for mitigation and internal intervention; relevant because it argues sycophancy arises from spurious causal structure rather than only superficial prompt effects.

43. **LLMs Trust Humans More, That's a Problem! Unveiling and Mitigating the Authority Bias in Retrieval-Augmented Generation**  
    Link: https://aclanthology.org/2025.acl-long.1400/  
    Why relevant: Not the same setup as ours, but a significant authority-bias paper showing models can favor user-provided claims over retrieved evidence; good adjacent precedent for authority-conditioned truth override under source conflict.

44. **Accounting for Sycophancy in Language Model Uncertainty Estimation**  
    Link: https://aclanthology.org/2025.findings-naacl.438/  
    Why relevant: Useful for the uncertainty/evidence-weighting angle; connects user confidence and model uncertainty to sycophantic drift, which is adjacent to our idea that authority may perturb internal evidence weighting.

45. **Training language models to be warm and empathetic makes them less reliable and more sycophantic**  
    Link: https://arxiv.org/abs/2507.21919  
    Why relevant: Relevant social-style confound paper; useful for distinguishing authority-conditioned compliance from a broader “warmth/agreeableness” drift story.

46. **Impact of authoritative and subjective cues on large language model reliability for clinical inquiries: an experimental study**  
    Link: https://nature.com/articles/s41598-026-38019-3  
    Why relevant: Useful applied motivation showing that authoritative misinformation can induce major accuracy failures in a high-stakes domain, even if it does not provide the same mechanistic depth as our work.

47. **The Dark Side of Trust: Authority Citation-Driven Jailbreak Attacks on Large Language Models**  
    Link: https://arxiv.org/abs/2411.11407  
    Why relevant: Adjacent safety paper showing that authority cues can be exploited adversarially; useful for motivating why authority-conditioned compliance matters beyond benchmark truthfulness.

## Suggested citation buckets for writing

- **Mechanistic steering / directions:** 2308.10248, 2406.11717, 2602.02132, 2026.eacl-long.324, 2412.00967, yRKelogz5i
- **Sycophancy / compliance mechanism:** 2310.13548, 2508.02087, 2601.11563, 2602.01002, 2604.03058, 2509.21305, 2601.21183, 2511.06419
- **Authority / deference / role conflict:** 2601.10896, 2601.17577, 2510.01228, 2601.13433, 2025.acl-long.1400
- **Emergent misalignment / latent behavioral structure:** 2502.17424, 2602.07852
- **Affect / emotion confounds:** Assistant Axis, Emotion Concepts, VA subspaces, NRC-VAD, Emotions Where Art Thou, warm/empathetic models
- **Benchmarks / measurement ecology:** SycEval, ELEPHANT, SYCON BENCH, Beacon, PARROT, Conformity in LLMs, Accounting for Sycophancy in Uncertainty Estimation
- **Eval-awareness / performative alignment:** the four LessWrong posts above

## Current most important papers for our draft

If we need a short “must cite” subset first:

1. Assistant Axis  
2. Emotion Concepts  
3. Valence–Arousal  
4. Steering Language Models With Activation Engineering  
5. Refusal in Language Models Is Mediated by a Single Direction  
6. Towards Understanding Sycophancy in Language Models  
7. When Truth Is Overridden  
8. Human-like Social Compliance in Large Language Models  
9. How RLHF Amplifies Sycophancy  
10. Who is In Charge? Dissecting Role Conflicts in Instruction Following  
11. ELEPHANT  
12. Sycophancy Is Not One Thing  
13. PARROT  
14. Trust Me, I'm an Expert  
15. Sycophancy Hides Linearly in the Attention Heads  
16. Sycophantic Anchors  
