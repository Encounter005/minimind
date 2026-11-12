#!/usr/bin/env python3
from minimind.FeedForward import FeedForward
from minimind.RMSNorm import RMSNorm
from minimind.attention import Attention
from minimind.RoPE import RoPE
from typing import Optional
from .LLMConfig import LLMConfig
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniBlock(nn.Module):
    def __init__(self, layer_id: int, cfg: LLMConfig, rope: RoPE):
        super().__init__()
        self.self_attn = Attention(cfg, rope)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = FeedForward(cfg)

    def forward(
        self,
        hidden_states,
        positions,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            positions,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )

        return hidden_states, present_key_value
