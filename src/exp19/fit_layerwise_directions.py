from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from joblib import Parallel, delayed

from src.exp16.run_steering_test import ExtractionStore, _load_extraction_store, _load_masks, _load_uids_from_masks
from src.exp19.common import normalize, parse_int_csv, save_direction_payload


STYLE_ORDER = ["weak", "uncertain", "assertive", "authoritative_verified"]
STYLE_TO_LEVEL = {name: float(i) for i, name in enumerate(STYLE_ORDER)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit per-layer authority directions from extraction activations.")
    p.add_argument("--extraction-dir", type=Path, required=True)
    p.add_argument("--masks-path", type=Path, required=True)
    p.add_argument("--layers", type=str, required=True, help="Comma-separated layer ids.")
    p.add_argument("--position", type=str, default="endorsement_mean")
    p.add_argument("--uids-subset", choices=("w1", "c1", "all"), default="w1")
    p.add_argument(
        "--direction-kind",
        choices=("w1_minus_c1", "c1_minus_w1", "authority_given_wrong", "authority_given_correct", "shared_within_label"),
        default="w1_minus_c1",
    )
    p.add_argument("--residualize-style", action="store_true")
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs across layers (CPU).")
    p.add_argument("--torch-threads", type=int, default=0, help="If >0, set torch intra-op threads.")
    p.add_argument("--resume", action="store_true", help="Skip layers with existing output files.")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if output exists.")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _stack_samples(
    store: ExtractionStore,
    *,
    layer: int,
    position: str,
    uids: Sequence[str],
    styles: Sequence[str],
    conditions: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mats: List[torch.Tensor] = []
    style_levels: List[torch.Tensor] = []
    cond_flags: List[torch.Tensor] = []
    for style in styles:
        for condition in conditions:
            m = store.matrix(position=position, layer_value=layer, uids=uids, style=style, condition_code=condition)
            if m.shape[0] == 0:
                continue
            mats.append(m)
            style_levels.append(torch.full((m.shape[0],), STYLE_TO_LEVEL[style], dtype=torch.float32))
            is_w1 = 1.0 if condition == "W1_note" else 0.0
            cond_flags.append(torch.full((m.shape[0],), is_w1, dtype=torch.float32))
    if not mats:
        return torch.zeros((0, 0)), torch.zeros((0,)), torch.zeros((0,))
    return torch.cat(mats, dim=0), torch.cat(style_levels, dim=0), torch.cat(cond_flags, dim=0)


def _style_slope(x: torch.Tensor, style_level: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    s = style_level.float()
    s_center = s - s.mean()
    var = torch.mean(s_center**2)
    if float(var.item()) == 0.0:
        return torch.zeros((x.shape[1],), dtype=torch.float32)
    x_center = x - x.mean(dim=0, keepdim=True)
    cov = torch.mean(s_center.unsqueeze(1) * x_center, dim=0)
    return cov / var


def _compute_direction(
    store: ExtractionStore,
    *,
    layer: int,
    position: str,
    uids: Sequence[str],
    kind: str,
    residualize_style: bool,
) -> Dict[str, torch.Tensor]:
    styles = STYLE_ORDER
    conditions = ["C1_note", "W1_note"]
    x, style_levels, cond_flags = _stack_samples(
        store,
        layer=layer,
        position=position,
        uids=uids,
        styles=styles,
        conditions=conditions,
    )
    if x.numel() == 0:
        raise ValueError(f"No samples for layer={layer}, position={position}.")
    slope = _style_slope(x, style_levels)
    if residualize_style and slope.numel() > 0:
        centered = style_levels - style_levels.mean()
        x = x - centered.unsqueeze(1) * slope.unsqueeze(0)

    is_w1 = cond_flags > 0.5
    is_c1 = ~is_w1
    if int(is_w1.sum().item()) == 0 or int(is_c1.sum().item()) == 0:
        raise ValueError(f"Need both C1/W1 samples at layer={layer}.")

    mean_w1 = x[is_w1].mean(dim=0)
    mean_c1 = x[is_c1].mean(dim=0)
    pooled_authority = 0.5 * ((mean_w1 - x.mean(dim=0)) + (mean_c1 - x.mean(dim=0)))
    dmap = {
        "w1_minus_c1": mean_w1 - mean_c1,
        "c1_minus_w1": mean_c1 - mean_w1,
        "authority_given_wrong": mean_w1 - x.mean(dim=0),
        "authority_given_correct": mean_c1 - x.mean(dim=0),
        "shared_within_label": mean_w1 - mean_c1,
        "pooled_authority": pooled_authority,
        "style_slope": slope,
    }
    return dmap


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
        raise SystemExit(f"Position '{args.position}' not in {store.position_names}")

    def _run_layer(layer: int) -> Dict[str, object]:
        if layer not in store.layer_to_index:
            raise ValueError(f"Layer {layer} not in extraction target layers {store.target_layers}")
        dmap = _compute_direction(
            store,
            layer=layer,
            position=args.position,
            uids=uids,
            kind=args.direction_kind,
            residualize_style=args.residualize_style,
        )
        vec = normalize(dmap[args.direction_kind])
        mode_tag = "style_residualized" if args.residualize_style else "raw"
        out_path = args.output_dir / f"layer_{layer}_{args.direction_kind}_{mode_tag}.pt"
        if args.resume and not args.overwrite and out_path.exists():
            print(f"[resume] skip layer={layer} ({out_path})")
            return {
                "layer": layer,
                "path": str(out_path),
                "direction_kind": args.direction_kind,
                "vector_norm": None,
                "style_slope_norm": None,
                "skipped": True,
            }
        save_direction_payload(
            path=out_path,
            vector=vec,
            layer=layer,
            position=args.position,
            direction_name=f"{args.direction_kind}_{mode_tag}",
            extra={
                "uids_subset": args.uids_subset,
                "residualize_style": bool(args.residualize_style),
                "style_order": STYLE_ORDER,
                "style_slope_norm": float(dmap["style_slope"].norm().item()) if dmap["style_slope"].numel() else 0.0,
            },
        )
        print(f"[ok] layer={layer} -> {out_path}")
        return {
            "layer": layer,
            "path": str(out_path),
            "direction_kind": args.direction_kind,
            "vector_norm": float(vec.norm().item()),
            "style_slope_norm": float(dmap["style_slope"].norm().item()) if dmap["style_slope"].numel() else 0.0,
        }

    if args.n_jobs == 1:
        summary = [_run_layer(layer) for layer in layers]
    else:
        summary = Parallel(n_jobs=args.n_jobs, prefer="threads")(delayed(_run_layer)(layer) for layer in layers)
    summary = sorted(summary, key=lambda x: int(x["layer"]))

    (args.output_dir / "layerwise_direction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.output_dir / 'layerwise_direction_summary.json'}")


if __name__ == "__main__":
    main()
