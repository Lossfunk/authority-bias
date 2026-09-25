from __future__ import annotations

import argparse
import os
from pathlib import Path

from .caa import CAASelection
from .model import QwenAdapter
from .pipeline import (
    analyze_run,
    bundle_run,
    fit_and_tune_caa,
    fit_paper_mitigation_vectors,
    fit_paper_reference_directions,
    fit_source_user_directions,
    fit_span_directions,
    h200_smoke,
    preflight,
    prepare_run,
    run_caa_fresh_margins,
    run_caa_only_margins,
    run_caa_reviewer_margins,
    run_generation_experiments,
    run_head_to_head_generation,
    run_margin_experiments,
    run_reviewer_generation_experiments,
    run_reviewer_margin_experiments,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean Qwen rebuttal pipeline")
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "preflight",
            "smoke",
            "reviewer-run",
            "head-to-head",
            "caa-only",
            "run",
            "analyze",
            "bundle",
            "all",
        ],
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("rebuttal/configs/qwen_h200.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("REBUTTAL_OUTPUT_ROOT", "/home/rebuttal-results")),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo_root = args.repo_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    output_root = args.output_root.resolve()
    config, splits, manifest, paths = prepare_run(
        repo_root=repo_root,
        config_path=config_path,
        output_root=output_root,
        require_clean=args.command != "prepare",
    )
    if args.command == "prepare":
        print(paths.run_dir)
        return
    if args.command == "bundle":
        archive, checksum = bundle_run(paths)
        print(f"{archive}\nsha256={checksum}")
        return
    if args.command == "analyze":
        selection_path = paths.directions / "caa_selection.json"
        if not selection_path.exists():
            raise RuntimeError("CAA selection is missing; run experiments before analysis")
        import json

        caa_selection = CAASelection(**json.loads(selection_path.read_text()))
        analyze_run(
            config=config,
            splits=splits,
            paths=paths,
            caa_selection=caa_selection,
        )
        print(paths.run_dir)
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required; anonymous model fallback is not permitted")
    needs_model = {
        "preflight",
        "smoke",
        "reviewer-run",
        "head-to-head",
        "caa-only",
        "run",
        "all",
    }
    if args.command in needs_model:
        preflight(config=config, output_root=output_root, token=token, paths=paths)
    if args.command == "preflight":
        print(paths.run_dir)
        return
    adapter = QwenAdapter.load(config.model, token)
    if args.command in {"smoke", "all"}:
        h200_smoke(
            adapter=adapter,
            config=config,
            splits=splits,
            repo_root=repo_root,
            manifest=manifest,
            paths=paths,
            token=token,
        )
        if args.command == "smoke":
            print(paths.run_dir)
            return
        adapter = QwenAdapter.load(config.model, token)
    if args.command == "caa-only":
        caa_selection, caa_direction = fit_and_tune_caa(
            adapter=adapter,
            config=config,
            manifest=manifest,
            paths=paths,
        )
        run_caa_only_margins(
            adapter=adapter,
            config=config,
            splits=splits,
            manifest=manifest,
            paths=paths,
            repo_root=repo_root,
            selection=caa_selection,
            direction=caa_direction,
        )
        print(paths.run_dir)
        return
    if args.command == "head-to-head":
        # The matched mitigation comparison. Both arms are configured before any
        # evaluation item is seen: authority removal takes the submitted paper's
        # declared layer and alpha and gets no search at all, while CAA takes
        # whatever its own layer-by-multiplier search picked on CAA's own
        # held-out data. The search budget is asymmetric in CAA's favour.
        vectors = fit_paper_mitigation_vectors(
            adapter=adapter,
            config=config,
            splits=splits,
            repo_root=repo_root,
            paths=paths,
        )
        caa_selection, caa_direction = fit_and_tune_caa(
            adapter=adapter,
            config=config,
            manifest=manifest,
            paths=paths,
        )
        run_head_to_head_generation(
            adapter=adapter,
            config=config,
            manifest=manifest,
            paths=paths,
            repo_root=repo_root,
            vectors=vectors,
            selection=caa_selection,
            caa_direction=caa_direction,
            uids=splits.fresh,
            split_name="fresh",
        )
        print(paths.run_dir)
        return
    fitted, shuffled, random = fit_source_user_directions(
        adapter=adapter,
        config=config,
        splits=splits,
        repo_root=repo_root,
        paths=paths,
    )
    if args.command == "reviewer-run":
        span_fitted = fit_span_directions(
            adapter=adapter,
            config=config,
            splits=splits,
            repo_root=repo_root,
            paths=paths,
        )
        paper_source = fit_paper_reference_directions(
            adapter=adapter,
            config=config,
            splits=splits,
            repo_root=repo_root,
            paths=paths,
        )
        run_reviewer_margin_experiments(
            adapter=adapter,
            config=config,
            splits=splits,
            manifest=manifest,
            paths=paths,
            repo_root=repo_root,
            fitted=fitted,
            span_fitted=span_fitted,
            paper_source=paper_source,
            shuffled=shuffled,
            random=random,
        )
        caa_selection, caa_direction = fit_and_tune_caa(
            adapter=adapter,
            config=config,
            manifest=manifest,
            paths=paths,
        )
        run_caa_reviewer_margins(
            adapter=adapter,
            config=config,
            splits=splits,
            manifest=manifest,
            paths=paths,
            repo_root=repo_root,
            selection=caa_selection,
            direction=caa_direction,
        )
        run_reviewer_generation_experiments(
            adapter=adapter,
            config=config,
            splits=splits,
            manifest=manifest,
            paths=paths,
            repo_root=repo_root,
            fitted=fitted[config.experiment.primary_layer],
            span_fitted=span_fitted[config.experiment.primary_layer],
        )
        print(paths.run_dir)
        return
    run_margin_experiments(
        adapter=adapter,
        config=config,
        splits=splits,
        manifest=manifest,
        paths=paths,
        repo_root=repo_root,
        fitted=fitted,
        shuffled=shuffled,
        random=random,
    )
    caa_selection, caa_direction = fit_and_tune_caa(
        adapter=adapter,
        config=config,
        manifest=manifest,
        paths=paths,
    )
    run_caa_fresh_margins(
        adapter=adapter,
        config=config,
        splits=splits,
        manifest=manifest,
        paths=paths,
        repo_root=repo_root,
        selection=caa_selection,
        direction=caa_direction,
    )
    run_generation_experiments(
        adapter=adapter,
        config=config,
        splits=splits,
        manifest=manifest,
        paths=paths,
        repo_root=repo_root,
        fitted=fitted[config.experiment.primary_layer],
        caa_selection=caa_selection,
        caa_direction=caa_direction,
    )
    if args.command in {"run"}:
        print(paths.run_dir)
        return
    analyze_run(
        config=config,
        splits=splits,
        paths=paths,
        caa_selection=caa_selection,
    )
    archive, checksum = bundle_run(paths)
    print(f"{archive}\nsha256={checksum}")
