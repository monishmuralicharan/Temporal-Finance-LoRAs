"""Lightweight LoRA utilities for Kronos predictor modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class LoRAInjectionReport:
    target_names: List[str]
    trainable_parameter_count: int
    total_parameter_count: int

    @property
    def target_count(self) -> int:
        return len(self.target_names)


class LoRALinear(nn.Module):
    """Wrap a frozen Linear layer with trainable low-rank adapters."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False)
        self.reset_lora_parameters()

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def inject_lora_adapters(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_prefixes: Sequence[str] = ("transformer", "dep_layer"),
    exclude_keywords: Sequence[str] = ("embedding", "time_emb", "head", "tokenizer"),
) -> LoRAInjectionReport:
    """Freeze base weights and wrap selected Linear modules with LoRA adapters."""
    freeze_base_model(model)
    selected = []
    for name, module in list(model.named_modules()):
        if not name:
            continue
        if not isinstance(module, nn.Linear):
            continue
        if _is_lora_target(name, target_prefixes, exclude_keywords):
            selected.append((name, module))
    if not selected:
        discovered = ", ".join(name for name, _ in list(model.named_modules())[:200])
        raise RuntimeError(
            "No Linear modules matched Temporal LoRA target prefixes. "
            "First discovered modules: {0}".format(discovered)
        )
    for name, module in selected:
        parent, child_name = _parent_module(model, name)
        setattr(
            parent,
            child_name,
            LoRALinear(
                base=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )
    return LoRAInjectionReport(
        target_names=[name for name, _ in selected],
        trainable_parameter_count=count_trainable_parameters(model),
        total_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def freeze_base_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def mark_only_lora_trainable(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_a." in name or "lora_b." in name


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def reset_lora_parameters(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.reset_lora_parameters()


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if ".lora_a." in name or ".lora_b." in name
    }


def save_lora_state(model: nn.Module, path: str | Path, metadata: Dict[str, object] | None = None) -> None:
    payload = {
        "lora_state_dict": lora_state_dict(model),
        "metadata": metadata or {},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_lora_state(
    model: nn.Module,
    path_or_state: str | Path | Dict[str, torch.Tensor],
    strict: bool = True,
) -> Dict[str, object]:
    if isinstance(path_or_state, (str, Path)):
        payload = torch.load(path_or_state, map_location="cpu")
        state = payload.get("lora_state_dict", payload)
        metadata = payload.get("metadata", {})
    else:
        state = path_or_state
        metadata = {}
    model_state = model.state_dict()
    missing = sorted(set(state) - set(model_state))
    if strict and missing:
        raise RuntimeError(
            "LoRA state contains keys not present in model: {0}".format(
                ", ".join(missing[:20])
            )
        )
    compatible = {
        key: value.to(model_state[key].device)
        for key, value in state.items()
        if key in model_state
    }
    model_state.update(compatible)
    model.load_state_dict(model_state, strict=False)
    return metadata


def discover_lora_target_names(
    model: nn.Module,
    target_prefixes: Sequence[str] = ("transformer", "dep_layer"),
    exclude_keywords: Sequence[str] = ("embedding", "time_emb", "head", "tokenizer"),
) -> List[str]:
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and _is_lora_target(name, target_prefixes, exclude_keywords)
    ]


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for name, parameter in model.named_parameters():
        if "lora_a." in name or "lora_b." in name:
            yield parameter


def _is_lora_target(
    name: str,
    target_prefixes: Sequence[str],
    exclude_keywords: Sequence[str],
) -> bool:
    normalized = name.replace(".base", "")
    if any(keyword in normalized for keyword in exclude_keywords):
        return False
    return any(
        normalized == prefix or normalized.startswith(prefix + ".")
        for prefix in target_prefixes
    )


def _parent_module(model: nn.Module, qualified_name: str):
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]
