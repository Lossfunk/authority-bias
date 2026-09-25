from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.decomposition import NMF
from sklearn.metrics import roc_auc_score

from src.authority_steering.run_steering_test import _load_extraction_store, _load_masks, _load_uids_from_masks
from src.direction_controls.common import normalize, parse_int_csv, save_direction_payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Discover SNMF-inspired authority features per layer.")
    p.add_argument("--extraction-dir", type=Path, required=True)
    p.add_argument("--masks-path", type=Path, required=True)
    p.add_argument("--layers", type=str, required=True)
    p.add_argument("--position", type=str, default="endorsement_mean")
    p.add_argument("--uids-subset", choices=("w1", "c1", "all"), default="w1")
    p.add_argument("--style", type=str, default="authoritative_verified")
    p.add_argument("--components", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=800)
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs across layers.")
    p.add_argument("--parallel-backend", choices=("threads", "processes"), default="threads")
    p.add_argument("--torch-threads", type=int, default=0, help="If >0, set torch intra-op threads.")
    p.add_argument("--resume", action="store_true", help="Skip layers with existing output files.")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if output exists.")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _oriented_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    auc = float(roc_auc_score(labels, scores))
    if auc >= 0.5:
        return auc, +1
    return 1.0 - auc, -1


def _prepare_matrix(
    store,
    *,
    layer: int,
    position: str,
    uids: List[str],
    style: str,
) -> tuple[np.ndarray, np.ndarray]:
    c1 = store.matrix(position=position, layer_value=layer, uids=uids, style=style, condition_code="C1_note")
    w1 = store.matrix(position=position, layer_value=layer, uids=uids, style=style, condition_code="W1_note")
    if c1.shape[0] == 0 or w1.shape[0] == 0:
        raise ValueError(f"No C1/W1 samples at layer={layer}, position={position}, style={style}.")
    x = torch.cat([c1, w1], dim=0).float().numpy()
    labels = np.concatenate([np.zeros(c1.shape[0], dtype=np.int64), np.ones(w1.shape[0], dtype=np.int64)], axis=0)
    return x, labels


def _fit_nmf_direction(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    n_components: int,
    seed: int,
    max_iter: int,
) -> Dict[str, object]:
    x_min = float(x.min())
    x_shift = x - x_min + 1e-6
    n_components = int(max(2, min(n_components, x_shift.shape[0] - 1, x_shift.shape[1] - 1)))
    nmf = NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=seed,
        max_iter=max_iter,
        solver="cd",
        beta_loss="frobenius",
    )
    w = nmf.fit_transform(x_shift)
    h = nmf.components_

    best_idx = 0
    best_auc = -1.0
    best_sign = +1
    per_comp: List[Dict[str, float]] = []
    for k in range(h.shape[0]):
        auc, sign = _oriented_auc(w[:, k], labels)
        per_comp.append({"component": int(k), "oriented_auc": float(auc), "sign": int(sign)})
        if auc > best_auc:
            best_auc = auc
            best_idx = k
            best_sign = sign

    vec = torch.tensor(h[best_idx], dtype=torch.float32)
    if best_sign < 0:
        vec = -vec
    return {
        "vector": normalize(vec),
        "best_component": int(best_idx),
        "best_oriented_auc": float(best_auc),
        "component_scores": per_comp,
        "x_min": x_min,
        "n_components": int(n_components),
        "reconstruction_err": float(nmf.reconstruction_err_),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    store = _load_extraction_store(args.extraction_dir)
    masks = _load_masks(args.masks_path)
    uids = _load_uids_from_masks(masks, args.uids_subset)
    layers = parse_int_csv(args.layers)

    if args.position not in store.position_names:
        raise SystemExit(f"Position '{args.position}' not in extraction positions {store.position_names}")

    def _fit_layer(layer: int) -> Dict[str, object]:
        if layer not in store.layer_to_index:
            raise ValueError(f"Layer {layer} not in extraction target layers {store.target_layers}")
        out_path = args.output_dir / f"layer_{layer}_snmf_feature_direction.pt"
        if args.resume and not args.overwrite and out_path.exists():
            print(f"[resume] skip layer={layer} ({out_path})")
            return {
                "layer": layer,
                "path": str(out_path),
                "best_component": None,
                "best_oriented_auc": None,
                "n_components": None,
                "reconstruction_err": None,
                "skipped": True,
            }

        x, labels = _prepare_matrix(
            store,
            layer=layer,
            position=args.position,
            uids=uids,
            style=args.style,
        )
        fitted = _fit_nmf_direction(
            x,
            labels,
            n_components=args.components,
            seed=args.seed,
            max_iter=args.max_iter,
        )
        save_direction_payload(
            path=out_path,
            vector=fitted["vector"],  # type: ignore[arg-type]
            layer=layer,
            position=args.position,
            direction_name="snmf_feature_direction",
            extra={
                "uids_subset": args.uids_subset,
                "style": args.style,
                "best_component": fitted["best_component"],
                "best_oriented_auc": fitted["best_oriented_auc"],
                "n_components": fitted["n_components"],
                "reconstruction_err": fitted["reconstruction_err"],
                "component_scores": fitted["component_scores"],
            },
        )
        print(f"[ok] layer={layer} auc={fitted['best_oriented_auc']:.3f} -> {out_path}")
        return {
            "layer": layer,
            "path": str(out_path),
            "best_component": fitted["best_component"],
            "best_oriented_auc": fitted["best_oriented_auc"],
            "n_components": fitted["n_components"],
            "reconstruction_err": fitted["reconstruction_err"],
        }

    if args.n_jobs == 1:
        summary = [_fit_layer(layer) for layer in layers]
    else:
        prefer = "processes" if args.parallel_backend == "processes" else "threads"
        summary = Parallel(n_jobs=args.n_jobs, prefer=prefer)(delayed(_fit_layer)(layer) for layer in layers)
    summary = sorted(summary, key=lambda x: int(x["layer"]))

    (args.output_dir / "snmf_direction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.output_dir / 'snmf_direction_summary.json'}")


if __name__ == "__main__":
    main()
