#!/usr/bin/env python3
from typing import Optional
from .LLMConfig import LLMConfig
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    # NOTE: SwiGLU 门控前馈层（Llama / Mistral 同款），结构为
    #   down( silu(gate(x)) * up(x) )，不是经典 Transformer 的 Linear->ReLU->Linear。

    def __init__(self, cfg: LLMConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        # NOTE: 允许外部显式指定中间维度（MoE 专家、缩小 width 的实验等），默认读 cfg。
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else cfg.intermediate_size
        )
        # NOTE: 三条投影一律 bias=False，与 Llama 实现一致：FFN 里的 bias 收益很小，
        #   还会破坏权重形状的整齐（融合/切分、量化都更麻烦）。
        self.gate_proj = nn.Linear(cfg.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, cfg.hidden_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, self.intermediate_size, bias=False)
        # NOTE: SiLU(x) = x * sigmoid(x)，平滑、非单调、负值区仍有梯度，比 ReLU 好训。
        self.act_fn = F.silu

    def forward(self, x: Tensor):
        # NOTE: 两次投影各自输出完整的 [B, T, intermediate_size]，逐元素相乘即门控：
        #   silu(gate(x)) 充当一个「由数据决定」的逐通道增益，把 up(x) 的通道放大或压到近 0。
        #   相比 ReLU-FFN，多出一条通路可以整条屏蔽某个通道，相当于逐样本的可学习稀疏/裁剪。
        # NOTE: 常见实现把 gate_proj 与 up_proj 拼成一个 [2*intermediate, hidden] 的矩阵做一次
        #   matmul，再 .chunk(2, dim=-1) 切成两半——那只是融合写法，与本文件两个独立 Linear
        #   数学上完全等价，好处仅少一次 kernel 启动和一次输出分配。本文件并未切分任何张量。
        # NOTE: 形状: [B, T, hidden] -> gate/up [B, T, I] -> 相乘 [B, T, I] -> 输出 [B, T, hidden]。
        # NOTE: 本仓库 I = ceil(H*π/64)*64 ≈ 3.17H，比标准 SwiGLU 的 8/3*H 略大，
        #   因此参数略多于经典 4H-FFN（代价是质量/参数比的提升）。
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
