**One-line question:** *do some LLMs contain a distinct internal authority-conditioned compliance mechanism that can override truth, and is it causal, portable, and not reducible to persona or affect?*

### **Step 1 — Behavioral override (N0 / C1 / W1 trivia)**

* **What:** same trivia stem under no endorsement (N0), correct-answer endorsement (C1), and wrong-answer endorsement (W1) with an authority note.  
* **Why:** gate for H1. If no model flips under W1, there's nothing mechanistic to chase.  
* **Result:** W1 flip rates; GPT-OSS 65%, Gemma-4 63–68%, OLMo-2 83–89%, Qwen3.5 45%, OLMo-3.1 72%, Grok-4.20 88%. Gemini-3.1-Pro \~1% (outlier). Large, cross-model behavioral effect. 

### **Step 2 — Authority gradient (weak → authoritative)**

* **What:** same experiment, four endorsement styles per note.  
* **Why:** distinguish "authority" from "someone said so." If gradient is flat, mechanism claim weakens.  
* **Result:** monotone gradient within-model for the ones we checked all four on; assertive alone only flips OLMo-2 (76%), OLMo-3.1 (55%), Qwen (42%). GPT-OSS and Gemma jump much higher under `authoritative_verified`. Social framing is graded, not binary.

### **Step 3 — Activation extraction \+ compliance direction**

* **What:** collect N0/C1/W1 activations across many layer/position combos; fit a direction separating W1-complying from C1-complying states; sweep layersthree null controls (shuffled labels, random directions, C1-null template).  
* **Why:** if no direction exists, H2 is dead. Probe-level check before causal step.  
* **Result:** separable direction at a layer per model (GPT-OSS L18, OLMo-3.1 L15, OLMo-2 L22, Gemma-4 L16, Qwen3.5 L2). Shuffled-label AUROC \~0.50 vs real 0.81+. C1-null directions don't probe the same thing.

### **Step 4 — Forward-patch causal steering (trivia)**

* **What:** inject the extracted direction into a clean N0 run at varying α and layers; measure matched flip (fraction of baseline-correct items that flip wrong).  
* **Why:** probes are not mechanisms. Forward patching is the H2 test.  
* **Result (best config per model):**  
  * OLMo-3.1 L15, α=1.0 → **58.8%** matched flip  
  * GPT-OSS L16, α=1.0 → **32.7%**  
  * OLMo-2 L16, α=0.5 → **19.0%**  
  * Qwen3.5 L2, α=4.0 → **16.2%**  
  * Gemma-4 L15, α=1.0 → **10.6%**  
* Clean dose response on GPT-OSS and OLMo-3.1. 

**![][image1]**

### **Step 5 — PIQA out-of-domain transfer**

* **What:** take the trivia-fit direction with no refitting, apply to PIQA physical commonsense.  
* **Why:** H3. Separates "learned trivia pattern" from "general compliance knob."  
* **Result:** matched flip on PIQA; Qwen 20.6%, GPT-OSS 18.8%, OLMo-2 10.4%, Gemma-4 1.4%. Transfers for three of four. H3 supported (with Gemma caveat).

### **Step 6 — SYCON sycophancy-native test**

* **What:** SYCON false-presupposition benchmark, steering the trivia direction at α ∈ {0, 0.3, 0.5} across three layers, 100q × 5 rounds, graded by gemini-3.1-flash-lite LLM-as-a-judge.  
* **Why:** H3b. PIQA is OOD but not sycophancy-native. SYCON has explicit endorsement-vs-truth conflict.  
* Some interesting observations:   
  * Qwen v3 had thinking enabled and it performed *worse* compared to v4, where we shut it off.   
* **Result — ΔFlip (steering-induced collapse):**  
  * GPT-OSS L12 α=0.5: 0.39 → **0.91** (Δ \+0.53) ← cleanest collapse  
  * Qwen v4 L10 α=0.5: 0.45 → **0.63** (Δ \+0.19)  
  * OLMo-3.1 L22 α=0.5: 0.25 → 0.33 (Δ \+0.08)  
  * OLMo-2 L22 α=0.5: 0.46 → 0.53 (Δ \+0.07)  
  * Gemma-4 L24 α=0.5: 0.58 → 0.62 (Δ \+0.03)  
* **Result — mean\_traj (sustained resistance):**   
  * Qwen v4 L5 α=0.5 **3.41**  
  * OLMo-3.1 L18 α=0.5 **3.04**  
  * Gemma L22 α=0.3 2.96  
  * OLMo-2 L10 α=0.3 2.91.   
    * Interesting inversion: Qwen L5 and OLMo-3.1 L18 α=0.5 *reinforce* resistance, while L10 / L22 at the same α collapse it. Layer-phase structure, not a monotone knob. 

### **Step 7 — Assistant-axis deconfound**

* **What:** build a hardened default-assistant-vs-roles axis; measure cosine with authority direction at the mechanism layer.  
* **Why:** H4. If cosine ≈ 1, we've just rediscovered persona.  
* **Result:** cos(authority, assistant) ≈ \+0.01 (GPT-OSS), −0.04 (Gemma), −0.04 (OLMo-2). Essentially orthogonal. **Qwen check still pending**

### **Step 8 — Valence-arousal deconfound**

* **What:** project authority direction onto lexical valence/arousal axes.  
* **Why:** "verified / confirmed" vocab is affectively charged. Rule out affect.  
* **Result:** near-orthogonal. Authority direction is not a relabeling of lexical affect.

### 

### **Step 9 — Null controls (C1-null placebo, shuffled labels)**

* **What:** rerun extraction with semantically null notes (C1-null) of the same token shape; shuffled-label direction fit.  
* **Why:** isolate "authority content" from "note format" and "pipeline artifact."  
* **Result:** C1-null matched flip;  
  * GPT-OSS 2.9%  
  * Gemma 1.5%   
  * OLMo-2 0.0%  
  * Qwen 0.0%.   
  * Real authority direction is 10–30× stronger. Shuffled labels → chance AUROC. 