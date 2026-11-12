"""按配置组装训练需要的对象：设备、数据、模型、优化器、autocast 上下文。"""

from __future__ import annotations

import inspect

import torch

from .configs import Config, OptimConfig
from .protocols import BatchSource, TrainableModel
from .registry import build_dataset, build_model
from .utils import amp_dtype, pick_device


def build_all(cfg: Config) -> tuple[torch.device, BatchSource, TrainableModel]:
    """train / sample / bench 共用的三件套。"""
    device, _ = pick_device(cfg.run.device)
    dataset = build_dataset(cfg.data)
    model = build_model(cfg.model, dataset.vocab_size).to(device)
    return device, dataset, model


def build_optimizer(model: TrainableModel, cfg: OptimConfig, device_type: str) -> torch.optim.AdamW:
    """2 维参数做权重衰减，其余不衰减（沿用 model.py 的做法，适用于任意模型）。"""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]  # type: ignore[attr-defined]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]  # type: ignore[attr-defined]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, **({"fused": True} if fused else {}))


def autocast_context(amp: str, device_type: str) -> torch.autocast:
    return torch.autocast(device_type=device_type, dtype=amp_dtype(amp), enabled=amp != "none")
