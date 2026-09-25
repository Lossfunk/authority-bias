# Tried Experiments and Main Results

This is a plain-language inventory of the main experiment buckets tried so far and what they found. It is meant as a crisp summary of the current repo state, not a source of new claims.

1. **Core behavioral experiments: instruction variants / evidence gating**
   - **What we tried:** Forced-choice factual QA with neutral, wrong-endorsement, and correct-endorsement conditions, plus instruction variants like `i1a`, `i1c`, and `i1d`, especially on prior-wrong items.
   - **Why we tried it:** To test whether "be accurate" style instructions make models more truth-tracking or instead change how they gate incoming evidence.
   - **Main result:** On Qwen3-4B, `i1a` can backfire on prior-wrong items under low-authority framing (`-0.68`), while `i1c` often helps (`+0.42`); `i1d` lands near neutral (`-0.05`).
   - **Bottom-line interpretation:** Instruction wording matters a lot. Some "accuracy" prompts behave like blanket skepticism rather than selective truth-tracking.

2. **Cross-model replication**
   - **What we tried:** Replicated the behavioral setup across Qwen3-4B, Llama-3.1-8B, GPT-oss-20B, Qwen3-30B-A3B, Qwen3.5-27B, OLMo-2-32B, and Gemma-4-26B, with PIQA and DebateQA extensions where available.
   - **Why we tried it:** To see whether the effect is a one-model quirk or a broader phenomenon.
   - **Main result:** The basic behavior replicates, but unevenly: Qwen3-4B shows the clearest sign flip, Llama shows attenuation / partial replication, Qwen3-30B still shows entrenchment at 30B, GPT-oss reveals the effect after prior-state conditioning, while Qwen3.5-27B and OLMo-2-32B have too few resisting items for strong causal conclusions. Gemma-4-26B is qualitatively different and shows near-zero strict entrenchment.
   - **Bottom-line interpretation:** The phenomenon is not unique to one model, but the strength and mechanism vary a lot by family and training recipe.

3. **Probe / patching / capping studies**
   - **What we tried:** Linear probes, cross-condition transfer, causal activation patching, and capping-style interventions on the putative correction-gating direction.
   - **Why we tried it:** To test whether there is a real internal direction tied to correcting vs resisting, and whether moving that direction changes behavior.
   - **Main result:** In Qwen3-4B, the endorsement-position signal is strong and causal (probe accuracy up to `81%`; layer-23 patching shifts margins by `+1.93` and flips `8.6%` of resisting items with `0%` harm in the main test). GPT-oss has decodable structure but much weaker steerability, improving mostly with multi-layer rather than single-layer patching.
   - **Bottom-line interpretation:** Qwen has a clean, intervention-sensitive gating signal; larger or different architectures can keep the representation while making it much harder to steer.

4. **Base vs instruct / RLHF comparison**
   - **What we tried:** Compared Qwen3-4B-Base vs Qwen3-4B-Instruct with logit-lens analysis and cross-model activation patching (CMAP).
   - **Why we tried it:** To ask whether post-training created the gating mechanism from scratch or sharpened something already latent.
   - **Main result:** The base model does not show the instruct model's strong prevention behavior on resisting items, but CMAP shows base-model downstream layers can use instruct-model gating activations if they are injected. Replacing instruct L22-L24 with base activations destroys almost all gating.
   - **Bottom-line interpretation:** Post-training appears to have strongly sharpened the mid-layer gating signal rather than building the whole downstream receiver from nothing.

5. **Attention/MLP decomposition**
   - **What we tried:** Decomposed the instruction-sensitive effect into attention-output and MLP-output contributions across layers in Qwen3-4B.
   - **Why we tried it:** To locate what kind of computation carries the gating signal.
   - **Main result:** Around the key gating layers, attention dominates (for example, at L24 on resisting items, attention divergence is about `8.6x` the MLP divergence), while later MLP layers amplify the consequences of whatever attention let through.
   - **Bottom-line interpretation:** The critical branching seems to happen in attention; later MLPs mostly amplify the branch that attention selected.

6. **Head surgery**
   - **What we tried:** Head-level attribution and ablation around L22-L24 to look for a small gating circuit.
   - **Why we tried it:** To see whether a few heads are doing most of the work.
   - **Main result:** Some heads are interpretable (`L24.H8` as an endorsement-reader, `L23.H7` as an instruction-reader), but the effect is not concentrated in a tiny circuit: the top-5 head cascade gives only about `2%` flips versus `8.6%` for full-layer patching. `L22.H3` is also causally important despite low divergence ranking.
   - **Bottom-line interpretation:** There are identifiable specialist heads, but the full gating effect is distributed and coordinated, not reducible to a tiny set of "magic heads."

7. **Temporal commitment**
   - **What we tried:** Patched the correction-gating direction at different layers to see when the model becomes committed to its answer.
   - **Why we tried it:** To distinguish early, still-editable computation from late, already-fixed computation.
   - **Main result:** There is a narrow vulnerability window around `L18`; by about `L24`, interventions stop flipping answers. Resisting items keep higher entropy than correcting items but still stay committed to the wrong answer.
   - **Bottom-line interpretation:** The model seems to decide surprisingly early. After that, later computation mostly carries the committed state forward.

8. **Multiplicative gating test**
   - **What we tried:** Tested whether the behavior is well explained by a simple multiplicative `gate × evidence` model.
   - **Why we tried it:** To see if one compact computational story explains the correction/resistance split.
   - **Main result:** The simple multiplicative regression fails badly (`R² < 0.3%`), so the clean item-level story is not supported. But gate values still separate groups, and the correction instruction shifts those gate values even for items that still resist.
   - **Bottom-line interpretation:** The model is processing the correction instruction, but that gate change alone is not enough to produce correction. The mechanism is more complicated than a simple multiplicative switch.

9. **Persona sweep**
   - **What we tried:** Ran six personas (`default`, `scientist`, `skeptic`, `judge`, `empath`, `therapist`) across instruction variants and speaker tags.
   - **Why we tried it:** To test whether system-prompt persona changes evidence gating in a substantial way.
   - **Main result:** Effects are mostly near-null. The only notable exception is a small empath effect under the Note tag (for example, `3.1%` correction under `i1a`), which looks more like margin compression than better evidence discrimination.
   - **Bottom-line interpretation:** Evidence gating mostly does not look like a persona-level phenomenon; persona changes nudge confidence more than they change the core gating computation.

10. **Gemma-4 expanded extraction**
    - **What we tried:** Because Gemma-4-26B has almost no strict resisting items, extracted an expanded label set based on susceptibility to wrong endorsement (`W1`) rather than strict `C1` resistance.
    - **Why we tried it:** To get something analyzable from a model where the headline entrenchment effect is nearly absent.
    - **Main result:** Strict resistance is almost nonexistent (`1/887`), but the expanded extraction yields `409` expanded-resisting and `478` expanded-correcting items. Confidence explains most of the behavior, and the confidence-independent probe signal largely collapses.
    - **Bottom-line interpretation:** Gemma-4 is not a clean instance of the main gating story. It is better described as a confidence-dominated susceptibility system than an instruction-dependent entrenchment system.

## What seems robust

- Instruction wording can flip the effect from entrenching to helping.
- Prior-state conditioning is essential; aggregate averages hide the failure mode.
- Qwen3-4B has a real, causal correction-gating signal at the endorsement position.
- The key branching looks attention-led, with later layers amplifying the outcome.
- The basic behavioral story replicates across multiple models and domains, even though strength varies.

## What still seems uncertain

- How far the Qwen-style mechanism generalizes to larger or different architectures.
- Whether GPT-oss, Qwen3.5-27B, and OLMo-2-32B share the same causal story or only a related representation.
- Whether there is a compact mechanistic description simpler than the current distributed account.
- How much to make of small persona effects, since most are near-null.
- How much Gemma-4 should be treated as part of the same phenomenon versus a different confidence-driven regime.
