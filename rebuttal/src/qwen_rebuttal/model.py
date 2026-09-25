from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from packaging.version import Version

from .types import ModelSpec

REQUIRED_DISTRIBUTIONS = {
    "torch": "2.9.1",
    "flash-attn": "2.8.3",
    "flash-linear-attention": "0.5.0",
    "causal-conv1d": "1.6.1",
}
TRANSFORMERS_COMMIT = "83c3672b706b47061c0fb56a10ae8a451ae35506"


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"Required distribution {name} is not installed") from error


def dependency_manifest() -> dict[str, str]:
    versions = {name: _distribution_version(name) for name in REQUIRED_DISTRIBUTIONS}
    for name, expected in REQUIRED_DISTRIBUTIONS.items():
        if Version(versions[name]).base_version != expected:
            raise RuntimeError(f"{name} must be {expected}, found {versions[name]}")
    distribution = importlib.metadata.distribution("transformers")
    direct_url_files = [
        item
        for item in distribution.files or ()
        if str(item).endswith(".dist-info/direct_url.json")
    ]
    if len(direct_url_files) != 1:
        raise RuntimeError("Cannot uniquely locate the Transformers direct_url.json")
    direct_url = Path(distribution.locate_file(direct_url_files[0]))
    if not direct_url.exists():
        raise RuntimeError("Cannot prove the installed Transformers Git revision")
    source = json.loads(direct_url.read_text())
    commit = source.get("vcs_info", {}).get("commit_id")
    if commit != TRANSFORMERS_COMMIT:
        raise RuntimeError(f"Transformers commit must be {TRANSFORMERS_COMMIT}, found {commit}")
    versions["transformers_commit"] = commit
    return versions


def assert_h200_name(name: str) -> None:
    if "H200" not in name.upper():
        raise RuntimeError(f"An H200 is required, found {name}")


def hardware_manifest(output_root: Path, minimum_free_gib: int = 180) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback is permitted")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Exactly one visible GPU is required, found {torch.cuda.device_count()}"
        )
    properties = torch.cuda.get_device_properties(0)
    name = properties.name
    assert_h200_name(name)
    free, total = torch.cuda.mem_get_info(0)
    disk = shutil.disk_usage(output_root.parent if not output_root.exists() else output_root)
    free_gib = disk.free / 2**30
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"At least {minimum_free_gib} GiB free disk is required, found {free_gib:.1f}"
        )
    smi = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,name,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    return {
        "gpu_name": name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "gpu_memory_total_bytes": total,
        "gpu_memory_free_bytes": free,
        "cuda_runtime": torch.version.cuda,
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "nvidia_smi": smi,
        "disk_free_bytes": disk.free,
    }


def assert_model_access(spec: ModelSpec, token: str | None) -> dict[str, str]:
    from huggingface_hub import model_info

    info = model_info(spec.model_id, revision=spec.revision, token=token)
    if info.sha != spec.revision:
        raise RuntimeError(f"Resolved model revision {info.sha}, expected {spec.revision}")
    return {"model_id": spec.model_id, "resolved_revision": info.sha}


@dataclass
class QwenAdapter:
    model: Any
    tokenizer: Any
    layers: list[Any]
    spec: ModelSpec
    reference_linear_attention: bool = False
    _scoring_head_weight: torch.Tensor | None = field(
        default=None, init=False, repr=False
    )

    @classmethod
    def load(
        cls,
        spec: ModelSpec,
        token: str | None,
        *,
        reference_linear_attention: bool = False,
    ) -> QwenAdapter:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if spec.dtype != "bfloat16":
            raise RuntimeError("This pipeline only permits BF16")
        if spec.attention_backend not in {"flash_attention_2", "eager"}:
            raise RuntimeError(f"Unsupported explicit attention backend: {spec.attention_backend}")
        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            token=token,
            trust_remote_code=False,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            token=token,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation=spec.attention_backend,
            device_map={"": 0},
        )
        model.eval()
        layers = cls._resolve_layers(model)
        adapter = cls(
            model=model,
            tokenizer=tokenizer,
            layers=layers,
            spec=spec,
            reference_linear_attention=reference_linear_attention,
        )
        if reference_linear_attention:
            adapter._force_reference_linear_attention()
        adapter.assert_architecture()
        return adapter

    @staticmethod
    def _resolve_layers(model: Any) -> list[Any]:
        candidates = (
            ("model", "language_model", "layers"),
            ("language_model", "layers"),
            ("model", "layers"),
        )
        for path in candidates:
            value = model
            try:
                for attribute in path:
                    value = getattr(value, attribute)
            except AttributeError:
                continue
            return list(value)
        raise RuntimeError("Unsupported Qwen layout: language-model layers not found")

    def assert_architecture(self) -> None:
        config = getattr(self.model.config, "text_config", self.model.config)
        layers = int(config.num_hidden_layers)
        hidden = int(config.hidden_size)
        if layers != self.spec.expected_layers or len(self.layers) != layers:
            raise RuntimeError(
                f"Expected {self.spec.expected_layers} Qwen layers, "
                f"found {layers}/{len(self.layers)}"
            )
        if hidden != self.spec.expected_hidden_size:
            raise RuntimeError(
                f"Expected hidden size {self.spec.expected_hidden_size}, found {hidden}"
            )
        implementations = {
            getattr(getattr(layer, "self_attn", None), "_attn_implementation", None)
            for layer in self.layers
        }
        configured = getattr(config, "_attn_implementation", None)
        if (
            configured != self.spec.attention_backend
            and self.spec.attention_backend not in implementations
        ):
            raise RuntimeError(
                f"FlashAttention was requested but model reports {configured}/{implementations}"
            )
        if next(self.model.parameters()).dtype != torch.bfloat16:
            raise RuntimeError("Model did not load in BF16")
        if next(self.model.parameters()).device.type != "cuda":
            raise RuntimeError("Model did not load on CUDA")
        linear_modules = [
            layer.linear_attn for layer in self.layers if hasattr(layer, "linear_attn")
        ]
        expected_linear = self.spec.expected_linear_attention_layers
        if len(linear_modules) != expected_linear:
            raise RuntimeError(
                f"Expected {expected_linear} linear-attention layers, "
                f"found {len(linear_modules)}"
            )
        if not linear_modules:
            # Dense families have no hybrid blocks, so the kernel-binding checks
            # below do not apply.
            return
        chunk_modules = {
            module.chunk_gated_delta_rule.__module__ for module in linear_modules
        }
        if self.reference_linear_attention:
            if any(name.startswith("fla.") for name in chunk_modules):
                raise RuntimeError("Reference run still has optimized linear-attention kernels")
        else:
            if not all(name.startswith("fla.") for name in chunk_modules):
                raise RuntimeError(
                    f"Optimized run did not bind flash-linear-attention: {chunk_modules}"
                )

    def _force_reference_linear_attention(self) -> None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_chunk_gated_delta_rule,
            torch_recurrent_gated_delta_rule,
        )

        for layer in self.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
                layer.linear_attn.recurrent_gated_delta_rule = (
                    torch_recurrent_gated_delta_rule
                )

    def render_chat(self, text: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def scoring_logits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        if logits_to_keep <= 0:
            raise ValueError("Scoring requires a positive logits_to_keep")
        base_model = getattr(self.model, "model", None)
        output_head = self.model.get_output_embeddings()
        if base_model is None or output_head is None:
            raise RuntimeError("Qwen scoring modules could not be resolved")
        if output_head.bias is not None:
            raise RuntimeError("Qwen output head unexpectedly has a bias")
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state[:, -logits_to_keep:, :]
        if self._scoring_head_weight is None:
            self._scoring_head_weight = output_head.weight.detach().to(torch.float32)
        return F.linear(hidden_states.float(), self._scoring_head_weight)

    def release(self) -> None:
        self.layers.clear()
        self._scoring_head_weight = None
        del self.model
        torch.cuda.empty_cache()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device
