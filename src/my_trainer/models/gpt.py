"""GPT 适配器：把根目录 model.py 的 GPT 包成模板契约，不复制实现。"""

from __future__ import annotations

import importlib

import torch.nn as nn
from torch import Tensor

from ..configs import ModelConfig
from ..registry import register_model
from ..utils import ConfigError


def _load_gpt():
    try:
        module = importlib.import_module("model")
    except ImportError as exc:
        raise ConfigError(
            "找不到根目录 model.py：请在仓库根目录运行 python -m my_trainer.train ..."
        ) from exc
    return module.GPT, module.GPTConfig


class GPTModel(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        GPT, GPTConfig = _load_gpt()
        self.block_size = cfg.block_size
        self.vocab_size = vocab_size
        self.inner = GPT(
            GPTConfig(
                block_size=cfg.block_size,
                vocab_size=vocab_size,
                n_layer=cfg.n_layer,
                n_head=cfg.n_head,
                n_embd=cfg.n_embd,
                dropout=cfg.dropout,
                bias=cfg.bias,
            )
        )
        self.config = self.inner.config

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        return self.inner(idx, targets)

    def generate(
        self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None
    ) -> Tensor:
        return self.inner.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@register_model("gpt")
def build_gpt(cfg: ModelConfig, vocab_size: int) -> GPTModel:
    return GPTModel(cfg, vocab_size)
