"""ckpt.pt 的读写：原子写入、按 [run] resume 定位。"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .configs import Config


def save_ckpt(
    path: Path,
    model,
    optimizer,
    scaler,
    step: int,
    best_val_loss: float,
    cfg: Config,
    vocab_size: int,
) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "step": step,
        "best_val_loss": float(best_val_loss),
        "config": asdict(cfg),
        "dataset": {"name": cfg.data.name, "reader": cfg.data.reader, "vocab_size": vocab_size},
    }
    tmp = path.with_suffix(".tmp")  # 先写临时文件再替换，避免中断留下半个存档
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_ckpt(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def resolve_resume(cfg: Config, out_dir: Path) -> Path | None:
    """auto：有存档就接续；none：从零；其它值当路径。"""
    if cfg.run.resume == "none":
        return None
    path = out_dir / "ckpt.pt" if cfg.run.resume == "auto" else Path(cfg.run.resume)
    return path if path.exists() else None
