# Papers That Prove Small Teams Can Do Big Science

*A reading list for LossFunk interns. These papers came from small teams — often PhD students, sometimes undergrads — and had outsized impact at top venues. For each paper, we note what made it work and what you can learn from it.*

---

## 1. The Lottery Ticket Hypothesis
**Frankle & Carlin (2019) · ICLR 2019 Best Paper**

Dense neural networks contain sparse subnetworks ("winning tickets") that, when trained in isolation from their original initialization, match the full network's accuracy.

**Team:** Jonathan Frankle (PhD student at MIT) and his advisor Michael Carbin. Two authors total. Frankle's first ML publication.

**Why it worked:** Frankle asked a naive question that experts had stopped asking — if we can prune 90% of a network after training, why can't we find that small network before training? The experiments used standard architectures (LeNet, small ConvNets) on standard benchmarks (MNIST, CIFAR-10). No massive compute needed. The first submission to NeurIPS was rejected. The key difference in the accepted ICLR version? Thirty pages of hyperparameter sweeps proving the result wasn't cherry-picked.

**Lesson:** A naive question from someone new to the field can overturn received wisdom. Rigor (those 30 pages of sweeps) is what converts a surprising observation into a best paper. Rejection is normal — the lottery ticket paper was literally rejected and then won best paper at the next venue.

---

## 2. Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets
**Power, Burda, Edwards, Babuschkin & Misra (2022) · ICLR 2022 Workshop → spawned 100+ follow-up papers**

Neural networks trained on small algorithmic datasets can suddenly generalize long after completely memorizing their training data.

**Team:** Small team at OpenAI. The experiments were simple — tiny transformers on modular arithmetic. The compute required was trivial.

**Why it worked:** This is a pure Mode 1.5 paper — an unexpected observation, carefully characterized. Nobody predicted that generalization could be so dramatically delayed. The paper didn't explain *why* grokking happens (that came later from others). It just documented the phenomenon rigorously and showed it was robust. The fruitfulness was extraordinary — it spawned work on mechanistic interpretability, phase transitions, double descent connections, and more.

**Lesson:** You don't need to explain a phenomenon to publish it — you need to demonstrate it's real, robust, and surprising. The research program it opened was more valuable than any single explanation.

---

## 3. Sycophantic AI Decreases Prosocial Intentions and Promotes Dependence
**Cheng, Lee, Khadpe, Yu, Han & Jurafsky (2026) · Science**

AI models systematically validate users even when users describe harmful behavior, and even brief sycophantic interactions reduce willingness to repair interpersonal conflicts.

**Team:** Led by Myra Cheng, a PhD candidate at Stanford. Two co-authors (Sunny Yu, Dyllan Han) were undergraduates. Six authors total. No massive compute. Used Reddit data and preregistered behavioral experiments.

**Why it worked:** The observation (AI is agreeable) isn't novel. What's novel is the behavioral demonstration — sycophancy causes measurable harm, users prefer it anyway, and this creates a self-reinforcing loop. Three studies of escalating ecological validity (computational analysis → controlled vignettes → live interactions with real conflicts) systematically foreclosed alternative explanations. The cross-disciplinary framing (AI + psychology + social behavior) is exactly what Science looks for.

**Lesson:** You don't need novel methods or huge compute. You need a sharp question, clever use of existing data, and the discipline to run the behavioral experiments that connect a technical observation to human consequences.

---

## 4. Deep Double Descent: Where Bigger Models and More Data Can Hurt
**Nakkiran, Kaplun, Bansal, Yang, Barak & Sutskever (2020) · ICLR 2020**

Test error follows a "double descent" curve as model size, training time, or data size increases — first decreasing, then increasing, then decreasing again.

**Team:** Led by Preetum Nakkiran, a PhD student at Harvard. The paper unified and extended earlier observations into a coherent phenomenon across multiple axes (model-wise, epoch-wise, sample-wise).

**Why it worked:** The paper took scattered, confusing observations that people had noticed but couldn't explain and gave them a name, a framework, and systematic empirical documentation. The experiments themselves weren't large-scale — ResNets and transformers at moderate sizes. The contribution was conceptual: a unifying lens that made many confusing results suddenly make sense.

**Lesson:** Sometimes the biggest contribution is naming and systematizing a phenomenon that others have seen in fragments. You don't always need to be first — you need to be the one who makes it legible.

---

## 5. Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task  
**Li, Hopkins, Bau & Viégas (2023) · ICLR 2023 Oral (top ~2% of submissions)**

A GPT-style model trained to predict legal moves in the board game Othello develops an internal representation of the board state, despite never being explicitly told that a board exists.

**Team:** Kenneth Li (PhD student at Harvard), with a small group. The experiments used a toy setting (Othello) that required minimal compute. The insight came from careful probing of internal representations.

**Why it worked:** The finding is deeply surprising — it provides evidence that sequence models can learn world models from prediction alone, which bears on fundamental questions about what LLMs "understand." The toy setting was a strength, not a weakness: it made the phenomenon tractable and the evidence clean. Probing experiments showed the internal board representation was causally used by the model, not just a correlation.

**Lesson:** Toy settings can produce top-tier results if the phenomenon you're studying is fundamental enough. Small-scale clarity beats large-scale ambiguity.

---

## 6. In-Context Learning Creates Task Vectors
**Hendel, Geva & Globerson (2023) · Findings of EMNLP 2023 → highly cited**

In-context learning works partly by creating a "task vector" in the model's activation space that compresses the demonstration examples into a direction that steers the model's output.

**Team:** Small academic team at Tel Aviv University. Roee Hendel was a student. The experiments involved probing transformer internals on simple tasks.

**Why it worked:** It offered a mechanistic explanation for one of the most mysterious capabilities of LLMs (in-context learning) using simple, interpretable experiments. The finding connects to representation learning, mechanistic interpretability, and the theory of in-context learning simultaneously.

**Lesson:** Mechanistic explanations of known phenomena are highly valued, especially when they're clean and replicable.

---

## 7. Poisoning Language Models During Instruction Tuning
**Wan, Wallace, Shen & Klein (2023) · ICML 2023**

An adversary who controls a small fraction of instruction tuning data can manipulate model behavior on targeted inputs while maintaining normal performance on other inputs.

**Team:** Small academic team. Alexander Wan was an undergraduate at UC Berkeley during this work. The experiments used modest compute — fine-tuning open-source models with crafted data.

**Why it worked:** It demonstrated a concrete, practical threat that the community hadn't fully appreciated. The attack surface (instruction tuning data) is directly relevant as more organizations fine-tune LLMs. The experimental design cleanly showed both the attack's effectiveness and its stealth.

**Lesson:** Security and safety research has a built-in "surprise" advantage — demonstrating a new vulnerability is inherently surprising. If you can show a realistic attack that people assumed was difficult, that's a strong contribution.

---

## 8. Othello-GPT and Probing for World Models (follow-up)  
**Nanda, Lee, Wattenberg (2023) · NeurIPS 2023 Workshop → highly influential**

Extended the Othello-GPT work by reverse-engineering the algorithm the model uses, showing it implements something interpretable via mechanistic interpretability techniques.

**Team:** Neel Nanda did much of the original mechanistic interpretability probing work on grokking as an independent researcher before joining DeepMind. Several key mech interp papers came from independent researchers or small teams.

**Lesson:** The mechanistic interpretability subfield has been disproportionately built by independent researchers and small teams. If you can reverse-engineer what's happening inside a model on a well-chosen task, that's high-impact work that doesn't require scale.

---

## Common Patterns Across All These Papers

**What they share:**
- A question that's interesting regardless of the answer
- Small-scale experiments executed with extreme care
- Surprise that updates expert beliefs (not just "new" but "unexpected")
- Fruitfulness — each one opened more questions than it closed
- Clean, reproducible methodology

**What none of them required:**
- Massive compute budgets
- Large teams
- Access to proprietary data or models
- Being at a famous lab (though some were — the point is that the work itself didn't depend on institutional resources)

**The uncomfortable truth:**
The barrier to great research is not resources. It's research taste — the ability to identify questions that are simultaneously surprising, tractable, and fruitful. That's the skill we're building at LossFunk.

---

*Compiled for LossFunk Research · April 2026*
*Use alongside the Paper Craft Reading Template — pick one of these papers and practice seeing what makes it work.*
