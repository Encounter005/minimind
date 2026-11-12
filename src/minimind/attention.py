#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .RMSNorm import RMSNorm
from .LLMConfig import LLMConfig
from torch import Tensor
from .RoPE import RoPE


class Attention(nn.Module):
    def __init__(self, cfg: LLMConfig, rope: RoPE):
        super().__init__()
        # 共用上层传入的 RoPE/YaRN，每层不再各建一份 cos/sin 查找表
        self.rope = rope
        self.n_local_kv_heads = cfg.n_kv_head
        self.n_local_heads = cfg.n_attn_head
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = cfg.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(
            cfg.hidden_size, cfg.n_attn_head * cfg.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            cfg.hidden_size, cfg.n_kv_head * cfg.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            cfg.hidden_size, cfg.n_kv_head * cfg.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            cfg.n_attn_head * cfg.head_dim, cfg.hidden_size, bias=False
        )
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout
        self.flash = torch.nn.functional.scaled_dot_product_attention

    @staticmethod
    def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
        batch_size, seq_len, num_key_value_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, num_key_value_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, num_key_value_heads * n_rep, head_dim)
        )

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = self.q_norm(xq), self.k_norm(xk)
        xq, xk = self.rope(xq, xk, positions)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            self._repeat_kv(xk, self.n_rep).transpose(1, 2),
            self._repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        if (
            (seq_len > 1)
            and (not self.is_causal or past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )

        else:
            scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
            if self.is_causal:
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device
                ).triu(1)

            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores, dim=-1).type_as(xq)) @ xv

        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
