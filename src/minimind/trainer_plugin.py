"""MiniModel 的 my_trainer 适配器：主干 + lm_head，包成 TrainableModel 契约。

配置里这样用（仓库根目录运行）：

    [model]
    name = "minimind"
    location = "src/minimind/trainer_plugin.py"

NOTE: my_trainer.registry.load_source 用 spec_from_file_location 按顶层模块导入本文件，
  所以这里只能用绝对 import，不能写相对 import。
"""

from __future__ import annotations

from dataclasses import fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from minimind.LLMConfig import LLMConfig
from minimind.model import MiniModel
from my_trainer.configs import ModelConfig
from my_trainer.registry import register_model

# NOTE: [model.extras] 与 LLMConfig 字段取交集，写错名字直接报错而不是被静默忽略。
_LLM_FIELDS = {f.name for f in fields(LLMConfig)}


class TrainerMiniMind(nn.Module):
    """MiniModel（主干）+ 独立 lm_head，与官方 MiniMindForCausalLM 的分层一致。"""

    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        extras = dict(cfg.extras or {})
        n_kv_head = int(extras.pop("n_kv_head", max(cfg.n_head // 2, 1)))
        # NOTE: 官方配置的 RoPE 表建到 32768 而训练长度只有 340，这里允许显式覆盖。
        max_pos = int(extras.pop("max_position_embeddings", cfg.block_size))
        # NOTE: 官方 MiniMindForCausalLM 把 lm_head 与 embedding 绑定（_tied_weights_keys），复刻时才省钱。
        tie_embeddings = bool(extras.pop("tie_embeddings", False))
        unknown = set(extras) - _LLM_FIELDS
        if unknown:
            raise ValueError(
                f"[model.extras] 里有 LLMConfig 不认识的字段 {sorted(unknown)}；"
                f"可用字段：{sorted(_LLM_FIELDS)}"
            )

        self.block_size = cfg.block_size
        self.vocab_size = vocab_size
        self.config = LLMConfig(
            vocab_size=vocab_size,
            hidden_size=cfg.n_embd,
            n_layers=cfg.n_layer,
            n_attn_head=cfg.n_head,
            n_kv_head=n_kv_head,
            dropout=cfg.dropout,
            # NOTE: 默认按 block_size 建 RoPE 查找表；官方配置用 extras 覆盖成 32768。
            max_position_embeddings=max_pos,
            **{k: v for k, v in extras.items() if k not in ("vocab_size", "hidden_size", "n_layers", "n_attn_head", "n_kv_head", "dropout", "max_position_embeddings")},
        )
        self.inner = MiniModel(self.config)
        # NOTE: 不与 embedding 共享权重（官方 MiniMind 是 tie 的）；本适配器不追求与官方权重互通。
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.inner.embedd_tokens.weight

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        hidden, _ = self.inner(idx)
        logits = self.lm_head(hidden)
        if targets is None:  # 契约：不传 targets 时只返回最后一步
            return logits[:, [-1], :], None
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None
    ) -> Tensor:
        # NOTE: 不用 KV cache：每步重算整个（截断到 block_size 的）前缀，采样速度够用且少一层出错面。
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size :])
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    k = min(int(top_k), logits.size(-1))
                    thresh = torch.topk(logits, k).values[:, [-1]]
                    logits = logits.masked_fill(logits < thresh, float("-inf"))
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@register_model("minimind")
def build_minimind(cfg: ModelConfig, vocab_size: int) -> TrainerMiniMind:
    return TrainerMiniMind(cfg, vocab_size)
