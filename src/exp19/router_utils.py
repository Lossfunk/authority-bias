from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from src.exp16.run_dissociation_test import _format_chat_prompt
from src.mechanism.hooks import _extract_hidden, _replace_hidden


def batch_by_tokens(items: List[Dict[str, Any]], max_batch_size: int, max_batch_tokens: int) -> List[List[Dict[str, Any]]]:
    out: List[List[Dict[str, Any]]] = []
    batch: List[Dict[str, Any]] = []
    max_len = 0
    for it in items:
        n_tok = int(it["n_tokens"])
        if not batch:
            batch = [it]
            max_len = n_tok
            continue
        cand_max = max(max_len, n_tok)
        cand_bs = len(batch) + 1
        if cand_bs > max_batch_size or cand_max * cand_bs > max_batch_tokens:
            out.append(batch)
            batch = [it]
            max_len = n_tok
        else:
            batch.append(it)
            max_len = cand_max
    if batch:
        out.append(batch)
    return out


def prepare_model_prompts(prompts: List[str], tokenizer, plain_prompt: bool, n_jobs: int) -> List[Dict[str, Any]]:
    def _fmt(text: str) -> Dict[str, Any]:
        mp = _format_chat_prompt(tokenizer, text, plain_prompt)
        n_tok = len(tokenizer(mp, add_special_tokens=False).input_ids)
        return {"model_prompt": mp, "n_tokens": n_tok}

    if n_jobs == 1:
        rows = [_fmt(p) for p in prompts]
    else:
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_fmt)(p) for p in prompts)
    rows.sort(key=lambda x: int(x["n_tokens"]))
    return rows


def router_hooks(model, patterns: List[str], stats: Dict[str, Dict[str, torch.Tensor | int]]):
    handles = []
    for name, module in model.named_modules():
        lname = name.lower()
        if not any(p in lname for p in patterns):
            continue

        def _hook(_m, _inp, out, mname=name):
            x = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(x) or x.ndim != 3 or x.shape[-1] < 2 or x.shape[-1] > 256:
                return
            probs = torch.softmax(x, dim=-1)
            top = probs.argmax(dim=-1).reshape(-1)
            n_experts = probs.shape[-1]
            hist = torch.bincount(top, minlength=n_experts).float().cpu()
            sum_probs = probs.sum(dim=(0, 1)).float().cpu()
            n_tokens = int(top.numel())
            if mname not in stats:
                stats[mname] = {"hist": hist, "sum_probs": sum_probs, "n_tokens": n_tokens}
            else:
                stats[mname]["hist"] = stats[mname]["hist"] + hist  # type: ignore[operator]
                stats[mname]["sum_probs"] = stats[mname]["sum_probs"] + sum_probs  # type: ignore[operator]
                stats[mname]["n_tokens"] = int(stats[mname]["n_tokens"]) + n_tokens  # type: ignore[arg-type]

        handles.append(module.register_forward_hook(_hook))
    return handles


def hidden_patch_hook(direction: torch.Tensor, alpha: float, position: str):
    def hook(_module, _inputs, out):
        hidden = _extract_hidden(out)
        d = direction.to(hidden.device, dtype=hidden.dtype)
        upd = hidden.clone()
        if position == "last_prompt_token":
            upd[:, -1, :] = hidden[:, -1, :] + alpha * d
        else:
            upd = hidden + alpha * d.view(1, 1, -1)
        return _replace_hidden(out, upd)

    return hook


def router_bias_hook(bias: torch.Tensor, alpha: float):
    def hook(_module, _inputs, out):
        x = out[0] if isinstance(out, tuple) else out
        if not torch.is_tensor(x) or x.ndim != 3:
            return out
        b = bias.to(x.device, dtype=x.dtype)
        y = x + alpha * b.view(1, 1, -1)
        if isinstance(out, tuple):
            return (y, *out[1:])
        return y

    return hook


def collect_router_stats(stats_in: Dict[str, Dict[str, torch.Tensor | int]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for name, rec in stats_in.items():
        n_tokens = int(rec["n_tokens"])
        if n_tokens <= 0:
            continue
        hist = rec["hist"].float()  # type: ignore[union-attr]
        sum_probs = rec["sum_probs"].float()  # type: ignore[union-attr]
        hist = hist / max(float(hist.sum().item()), 1.0)
        mean_probs = sum_probs / max(float(n_tokens), 1.0)
        stats[name] = {
            "n_experts": int(hist.numel()),
            "n_tokens": n_tokens,
            "top1_hist": hist.tolist(),
            "mean_probs": mean_probs.tolist(),
        }
    return stats


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p / max(float(p.sum().item()), 1e-8)
    q = q / max(float(q.sum().item()), 1e-8)
    m = 0.5 * (p + q)
    eps = 1e-8
    kl_pm = torch.sum(p * torch.log((p + eps) / (m + eps)))
    kl_qm = torch.sum(q * torch.log((q + eps) / (m + eps)))
    return float((0.5 * (kl_pm + kl_qm)).item())


def diff_stats(base: Dict[str, Dict[str, Any]], other: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name in sorted(set(base) & set(other)):
        p = torch.tensor(base[name]["top1_hist"], dtype=torch.float32)
        q = torch.tensor(other[name]["top1_hist"], dtype=torch.float32)
        mp = torch.tensor(base[name]["mean_probs"], dtype=torch.float32)
        mq = torch.tensor(other[name]["mean_probs"], dtype=torch.float32)
        out[name] = {
            "js_top1_hist": js_divergence(p, q),
            "l1_top1_hist": float(torch.sum(torch.abs(p - q)).item()),
            "l1_mean_probs": float(torch.sum(torch.abs(mp - mq)).item()),
        }
    return out


def learn_router_biases(base_c1: Dict[str, Dict[str, Any]], base_w1: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name in sorted(set(base_c1) & set(base_w1)):
        p1 = torch.tensor(base_c1[name]["mean_probs"], dtype=torch.float32)
        p2 = torch.tensor(base_w1[name]["mean_probs"], dtype=torch.float32)
        delta = p2 - p1
        n = float(delta.norm().item())
        out[name] = delta / n if n > 0 else delta
    return out


def run_forward(
    *,
    model,
    tokenizer,
    prepared_prompts: List[Dict[str, Any]],
    batch_size: int,
    max_batch_tokens: int,
    router_patterns: List[str],
    hidden_patch: Tuple[Any, torch.Tensor, float, str] | None = None,
    router_biases: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, torch.Tensor | int]] = {}
    handles = router_hooks(model, router_patterns, stats)
    if hidden_patch is not None:
        module, vec, alpha, pos = hidden_patch
        handles.append(module.register_forward_hook(hidden_patch_hook(vec, alpha, pos)))
    if router_biases:
        for name, module in model.named_modules():
            if name in router_biases:
                handles.append(module.register_forward_hook(router_bias_hook(router_biases[name], 1.0)))
    try:
        batches = batch_by_tokens(prepared_prompts, batch_size, max_batch_tokens)
        for batch in tqdm(batches, desc="forward"):
            model_prompts = [str(x["model_prompt"]) for x in batch]
            inputs = tokenizer(model_prompts, padding=True, truncation=False, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                model(**inputs, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return collect_router_stats(stats)
