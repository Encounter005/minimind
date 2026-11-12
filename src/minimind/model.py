#!/usr/bin/env python3
from minimind.RMSNorm import RMSNorm
from minimind.block import MiniBlock
from minimind.RoPE import RoPE
from minimind.YaRN import YaRN
from .LLMConfig import LLMConfig
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniModel(nn.Module):
    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.cfg = cfg
        self.vocab_size, self.num_hidden_layers = (
            cfg.vocab_size,
            cfg.n_layers,
        )

        self.embedd_tokens = nn.Embedding(self.vocab_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        # 全模型共用一份 RoPE 查找表；配置了 rope_scaling 就换成 YaRN 那张频率表
        self.rope = (
            YaRN(
                cfg.head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                cfg.rope_scaling,
            )
            if cfg.rope_scaling
            else RoPE(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)
        )
        self.layers = nn.ModuleList(
            [MiniBlock(l, cfg, self.rope) for l in range(self.num_hidden_layers)]
        )

        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        hidden_states = self.dropout(self.embedd_tokens(input_ids))
        # 位置信息只在这里算一次：带 KV cache 时新 token 的绝对位置从 start_pos 开始
        positions = torch.arange(
            start_pos, start_pos + seq_length, device=input_ids.device
        )
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                positions,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
        return self.norm(hidden_states), presents
