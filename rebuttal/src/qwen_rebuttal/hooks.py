from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

HookMode = Literal["add", "project_out", "collect"]
HookPhase = Literal["prefill", "answer", "all"]


@dataclass
class HookAudit:
    layer: int
    calls: int = 0
    changed_calls: int = 0
    changed_rows: list[int] = field(default_factory=list)
    changed_positions: list[int] = field(default_factory=list)
    observed_sequence_lengths: list[int] = field(default_factory=list)


class ResidualHook(AbstractContextManager["ResidualHook"]):
    """One explicit residual-stream hook with a machine-checkable audit trail."""

    def __init__(
        self,
        *,
        module: Any,
        layer: int,
        mode: HookMode,
        vector: torch.Tensor | None = None,
        coefficient: float = 1.0,
        phase: HookPhase = "prefill",
        token_positions: list[int | list[int]] | None = None,
        row_mask: list[bool] | None = None,
    ) -> None:
        self.module = module
        self.layer = layer
        self.mode = mode
        self.vector = vector
        self.coefficient = coefficient
        self.phase = phase
        self.token_positions = token_positions
        self.row_mask = row_mask
        self.audit = HookAudit(layer=layer)
        self.collected: list[torch.Tensor] = []
        self._handle: Any = None

    @staticmethod
    def _hidden(output: Any) -> tuple[torch.Tensor, Any]:
        if isinstance(output, tuple):
            return output[0], lambda updated: (updated, *output[1:])
        if torch.is_tensor(output):
            return output, lambda updated: updated
        raise TypeError(f"Unsupported transformer block output: {type(output)}")

    def _active(self, sequence_length: int) -> bool:
        if self.phase == "all":
            return True
        if self.phase == "prefill":
            return sequence_length > 1
        return sequence_length == 1

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        hidden, rebuild = self._hidden(output)
        self.audit.calls += 1
        self.audit.observed_sequence_lengths.append(hidden.shape[1])
        if not self._active(hidden.shape[1]):
            return output
        if self.mode == "collect":
            if self.token_positions is None:
                raise RuntimeError("Collection requires one token position per batch row")
            if len(self.token_positions) != hidden.shape[0]:
                raise RuntimeError("Hook batch selectors do not match the hidden-state batch")
            values: list[torch.Tensor] = []
            for row, row_positions in enumerate(self.token_positions):
                # A list of positions collects their mean, which is how span-fitted
                # directions are built; a bare int collects that one position.
                if isinstance(row_positions, int):
                    values.append(hidden[row, row_positions].detach().float().cpu())
                    continue
                if not row_positions:
                    raise RuntimeError("Span collection needs at least one position per row")
                span = torch.stack(
                    [hidden[row, position] for position in row_positions]
                )
                values.append(span.detach().float().mean(dim=0).cpu())
            self.collected.extend(values)
            return output
        if self.vector is None:
            raise RuntimeError(f"{self.mode} hook requires a vector")
        positions = self.token_positions
        if positions is None:
            positions = [hidden.shape[1] - 1] * hidden.shape[0]
        rows = self.row_mask if self.row_mask is not None else [True] * hidden.shape[0]
        if len(positions) != hidden.shape[0] or len(rows) != hidden.shape[0]:
            raise RuntimeError("Hook batch selectors do not match the hidden-state batch")
        updated = hidden.clone()
        unit = self.vector.to(device=hidden.device, dtype=hidden.dtype)
        if self.mode == "project_out":
            norm = torch.linalg.vector_norm(unit.float())
            if not torch.isfinite(norm) or norm <= 0:
                raise RuntimeError("Cannot project out a zero or non-finite vector")
            unit = unit / norm.to(unit.dtype)
        for row, (enabled, row_positions) in enumerate(zip(rows, positions, strict=True)):
            if not enabled:
                continue
            if isinstance(row_positions, int):
                row_positions = [row_positions]
            for position in row_positions:
                if not -hidden.shape[1] <= position < hidden.shape[1]:
                    raise IndexError(f"Token position {position} is outside sequence")
                before = updated[row, position].clone()
                if self.mode == "add":
                    updated[row, position] = before + self.coefficient * unit
                elif self.mode == "project_out":
                    projection = torch.dot(before.float(), unit.float()).to(before.dtype)
                    updated[row, position] = before - self.coefficient * projection * unit
                else:
                    raise RuntimeError(f"Unknown hook mode {self.mode}")
                if not torch.equal(before, updated[row, position]):
                    self.audit.changed_calls += 1
                    self.audit.changed_rows.append(row)
                    self.audit.changed_positions.append(position)
        return rebuild(updated)

    def __enter__(self) -> ResidualHook:
        if self._handle is not None:
            raise RuntimeError("Hook is already registered")
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class MultiResidualHooks(AbstractContextManager["MultiResidualHooks"]):
    def __init__(self, hooks: list[ResidualHook]) -> None:
        self.hooks = hooks

    def __enter__(self) -> MultiResidualHooks:
        for hook in self.hooks:
            hook.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        for hook in reversed(self.hooks):
            hook.__exit__(*exc)
