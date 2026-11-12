#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class LLMConfig:
    # NOTE: 字段顺序即 __init__ 位置参数顺序，调用方尽量用关键字传参。
    vocab_size: int = 6400
    hidden_size: int = 768
    n_layers: int = 8
    n_attn_head: int = 8
    n_kv_head: int = 4
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768
    rope_theta: float = 1e6
    rope_scaling: Optional[dict] = None
    # NOTE: 0 = 未指定，由 __post_init__ 按 hidden_size / n_attn_head 推导。
    #   不能在类定义处直接写表达式：dataclass 默认值只在类创建时求值一次，
    #   LLMConfig(hidden_size=128, n_attn_head=4) 会拿到 768//8 的 96。
    head_dim: int = 0
    intermediate_size: int = 0
    hidden_act: str = "silu"  # NOTE: 目前未被使用，FeedForward 固定 F.silu

    def __post_init__(self):
        if self.hidden_size % self.n_attn_head:
            raise ValueError(
                f"hidden_size({self.hidden_size}) 必须能被 n_attn_head({self.n_attn_head}) 整除"
            )
        if self.n_attn_head % self.n_kv_head:
            raise ValueError(
                f"n_attn_head({self.n_attn_head}) 必须是 n_kv_head({self.n_kv_head}) 的整数倍（GQA）"
            )
        if not self.head_dim:
            self.head_dim = self.hidden_size // self.n_attn_head
        if self.head_dim % 2:
            raise ValueError(f"head_dim({self.head_dim}) 必须为偶数，RoPE 按前后两半旋转")
        if not self.intermediate_size:
            self.intermediate_size = math.ceil(self.hidden_size * math.pi / 64) * 64
