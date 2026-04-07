"""Multi-layer patching for GPT-oss-20B.

Hypothesis: GPT-oss has the representational signal (probe works, AUROC=0.677)
but single-layer patching was null because the computation is distributed.
This script patches the top-K layers simultaneously.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from src.exp10.conditions import Exp10Condition, format_prompt
from src.exp7.dataset_mc import MCExample, load_mc_dataset
from src.exp7.scoring import _tokenize_for_scoring, get_ab_token_ids
from src.mechanism.extract_correction_gating_activations import _resolve_positions
from src.mechanism.hooks import get_transformer_layers
from src.models.llama_loader import load_model_and_tokenizer


DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_EXTRACTION_DIR = Path(
    "new-phase-results/mechanism/gpt_oss_20b_correction_gating_i1a_i1c_note"
)
DEFAULT_MC_DATASET_PATH = Path("data/exp7_mc_dataset.jsonl")
DEFAULT_OUTPUT_DIR = Path(
    "new-phase-results/mechanism/gpt_oss_multilayer_patching"
)

I1A_TEXT = "Answer correctly even if the speaker is wrong. Prioritize factual accuracy."
NEGATIVE_LABELS = frozenset({"entrenching", "resisting"})


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def signed_margin(logit_a, logit_b, correct_label):
    if correct_label == "A":
        return logit_a - logit_b
    return logit_b - logit_a


class MultiLayerPatcher:
    """Patches multiple transformer layers simultaneously."""
    def __init__(self, layers_module, layer_indices, token_position, vectors):
        self.layers = layers_module
        self.layer_indices = layer_indices
        self.token_position = token_position
        self.vectors = vectors  # dict: layer_idx -> vector
        self.handles = []

    def __enter__(self):
        for layer_idx in self.layer_indices:
            layer = self.layers[layer_idx]
            vec = self.vectors[layer_idx]

            def make_hook(v):
                def hook(_module, _inputs, output):
                    if torch.is_tensor(output):
                        patched = output.clone()
                        patched[:, self.token_position, :] = (
                            patched[:, self.token_position, :] + v
                        )
                        return patched
                    elif isinstance(output, tuple):
                        hidden = output[0].clone()
                        hidden[:, self.token_position, :] = (
                            hidden[:, self.token_position, :] + v
                        )
                        return (hidden, *output[1:])
                    return output
                return hook

            self.handles.append(layer.register_forward_hook(make_hook(vec)))
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--extraction-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--mc-dataset-path", type=Path, default=DEFAULT_MC_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", type=str, default="Note")
    parser.add_argument("--position", type=str, default="endorsement_last")
    parser.add_argument("--top-k-layers", type=int, nargs="+", default=[1, 3, 5, 8])
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load extraction
    meta = load_jsonl(args.extraction_dir / "metadata.jsonl")
    payload = torch.load(args.extraction_dir / "activations.pt", map_location="cpu")
    acts = payload["activations"]

    # Load selected items for labels
    selected = load_jsonl(args.extraction_dir / "selected_items.jsonl")
    uid_to_label = {}
    for row in selected:
        label = row["label"]
        if label == "entrenching":
            label = "resisting"
        uid_to_label[row["uid"]] = label

    # Identify i1a rows
    i1a_correcting_idx = []
    i1a_resisting_idx = []
    i1a_resisting_meta = []
    for i, row in enumerate(meta):
        if row["variant_label"] != "i1a":
            continue
        label = uid_to_label.get(row["uid"], row["item_label"])
        if label == "entrenching":
            label = "resisting"
        if label == "correcting":
            i1a_correcting_idx.append(i)
        else:
            i1a_resisting_idx.append(i)
            i1a_resisting_meta.append(row)

    if args.max_items > 0:
        i1a_resisting_idx = i1a_resisting_idx[:args.max_items]
        i1a_resisting_meta = i1a_resisting_meta[:args.max_items]

    print(f"Correcting: {len(i1a_correcting_idx)}, Resisting: {len(i1a_resisting_idx)}")

    # Compute per-layer probe AUROCs to find best layers (parallelized)
    import os
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pos_key = args.position
    pos_acts = acts[pos_key]  # [N, n_layers, d]

    all_i1a_idx = i1a_correcting_idx + i1a_resisting_idx
    all_i1a_labels = [1]*len(i1a_correcting_idx) + [0]*len(i1a_resisting_idx)
    X_all = pos_acts[torch.tensor(all_i1a_idx)].float().numpy()
    y_all = torch.tensor(all_i1a_labels).numpy()

    n_layers = X_all.shape[1]
    n_jobs = os.cpu_count()
    print(f"Sweeping {n_layers} layers for probe AUROC across {n_jobs} workers...")

    def _probe_one_layer(layer):
        X_l = X_all[:, layer, :]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        scores = []
        for tr, te in cv.split(X_l, y_all):
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=5000, solver="saga",
                                                         class_weight="balanced", random_state=0))])
            pipe.fit(X_l[tr], y_all[tr])
            scores.append(roc_auc_score(y_all[te], pipe.predict_proba(X_l[te])[:, 1]))
        return layer, sum(scores) / len(scores)

    results_parallel = Parallel(n_jobs=-1, verbose=5)(
        delayed(_probe_one_layer)(layer) for layer in range(n_layers)
    )
    layer_aurocs = [0.0] * n_layers
    for layer, auroc in results_parallel:
        layer_aurocs[layer] = auroc

    # Sort layers by AUROC
    ranked = sorted(range(n_layers), key=lambda i: layer_aurocs[i], reverse=True)
    print("Top 10 layers by AUROC:")
    for i in ranked[:10]:
        print(f"  Layer {i}: AUROC={layer_aurocs[i]:.3f}")

    # Build per-layer mean-difference vectors for top layers
    max_k = max(args.top_k_layers)
    top_layers = ranked[:max_k]

    per_layer_vectors = {}
    for layer_idx in top_layers:
        corr_acts = pos_acts[torch.tensor(i1a_correcting_idx), layer_idx, :].float()
        resist_acts = pos_acts[torch.tensor(i1a_resisting_idx), layer_idx, :].float()
        vec = corr_acts.mean(0) - resist_acts.mean(0)
        per_layer_vectors[layer_idx] = vec

    # Random control vectors
    import random
    random.seed(0)
    torch.manual_seed(0)
    per_layer_random = {}
    for layer_idx in top_layers:
        rv = torch.randn_like(per_layer_vectors[layer_idx])
        rv = rv / rv.norm() * per_layer_vectors[layer_idx].norm()
        per_layer_random[layer_idx] = rv

    # Load model
    print(f"\nLoading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(
        args.model, device=args.device, dtype="bfloat16",
        prefer_flash_attention=True,
    )
    model.eval()
    device = next(model.parameters()).device
    transformer_layers = get_transformer_layers(model)

    token_id_a, token_id_b = get_ab_token_ids(tokenizer)
    mc_items = load_mc_dataset(args.mc_dataset_path)
    uid_to_mc = {item.uid: item for item in mc_items}

    tag = args.tag
    instruction_text = I1A_TEXT

    # Run patching experiments
    all_experiment_results = []

    for k in args.top_k_layers:
        layers_to_patch = ranked[:k]
        # Transformer layer index = saved_layer - 1 (saved includes embedding at 0)
        transformer_indices = [l - 1 for l in layers_to_patch if l > 0]

        for alpha in args.alphas:
            for condition_name, use_vectors in [("real", per_layer_vectors),
                                                  ("random", per_layer_random),
                                                  ("opposite", {l: -v for l, v in per_layer_vectors.items()})]:
                # Build scaled vectors for this alpha
                scaled = {}
                for l_idx in transformer_indices:
                    saved_idx = l_idx + 1  # map back to saved indexing
                    if saved_idx in use_vectors:
                        scaled[l_idx] = (alpha * use_vectors[saved_idx]).to(device=device, dtype=torch.bfloat16)

                if not scaled:
                    continue

                margin_shifts = []
                flips = 0
                total = 0

                for row in tqdm(i1a_resisting_meta, desc=f"k={k} a={alpha} {condition_name}", leave=False):
                    mc_item = uid_to_mc.get(row["uid"])
                    if mc_item is None:
                        continue

                    cond = Exp10Condition(
                        endorse_type="correct", instruction=True,
                        tag=tag, instruction_text=instruction_text,
                    )
                    prompt_text = format_prompt(mc_item, cond)
                    endorsement_line = f"{tag}: I think it's {mc_item.correct_label}, not {mc_item.wrong_label}."
                    positions = _resolve_positions(tokenizer, prompt_text, instruction_text, endorsement_line)

                    patch_pos = positions.get(args.position)
                    if patch_pos is None:
                        continue

                    inputs = {k_: v.to(device) for k_, v in _tokenize_for_scoring(tokenizer, prompt_text).items()}

                    # Baseline (no patch)
                    with torch.inference_mode():
                        base_out = model(**inputs, use_cache=False)
                    base_logits = base_out.logits[0, -1, :]
                    base_margin = signed_margin(
                        base_logits[token_id_a].item(),
                        base_logits[token_id_b].item(),
                        mc_item.correct_label,
                    )

                    # Patched
                    patch_vecs = {l: scaled[l] for l in scaled}
                    with MultiLayerPatcher(transformer_layers, list(patch_vecs.keys()),
                                            patch_pos, patch_vecs):
                        with torch.inference_mode():
                            patched_out = model(**inputs, use_cache=False)
                    patched_logits = patched_out.logits[0, -1, :]
                    patched_margin = signed_margin(
                        patched_logits[token_id_a].item(),
                        patched_logits[token_id_b].item(),
                        mc_item.correct_label,
                    )

                    shift = patched_margin - base_margin
                    margin_shifts.append(shift)
                    if base_margin < 0 and patched_margin > 0:
                        flips += 1
                    total += 1

                if total == 0:
                    continue

                mean_shift = sum(margin_shifts) / len(margin_shifts)
                flip_rate = flips / total
                pos_rate = sum(1 for s in margin_shifts if s > 0) / total

                result = {
                    "k": k,
                    "alpha": alpha,
                    "condition": condition_name,
                    "layers": layers_to_patch,
                    "n_items": total,
                    "mean_shift": mean_shift,
                    "flip_rate": flip_rate,
                    "positive_shift_rate": pos_rate,
                }
                all_experiment_results.append(result)
                print(f"  k={k} alpha={alpha} {condition_name:8s}: "
                      f"shift={mean_shift:+.3f} flips={flips}/{total} ({flip_rate:.1%}) "
                      f"pos_rate={pos_rate:.1%}")

    # Save results
    out_path = args.output_dir / "multilayer_patching_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "layer_aurocs": {str(i): layer_aurocs[i] for i in ranked[:10]},
            "experiments": all_experiment_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
