#!/usr/bin/env python3
from typing import Optional
import torch
import math
from torch import Tensor

# RoPE
def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (
        (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    ).to(q.type)

    k_embed = (
        (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    ).to(k.type)

    return q_embed, k_embed


# YaRN
def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, step=2)[: (dim // 2)].float() / dim)),
        1.0,
    )
    if rope_scaling is not None:
        # f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )
        # NOTE:推断长度大于训练长度，使用缩放
        if end / orig_max > 1.0:
            # NOTE: 波长b到i的映射
            inv_dim = lambda b: (
                (dim * math.log(orig_max / (b * 2 * math.pi)))
                / (2 * math.log(rope_base))
            )

            # NOTE: 划分高低维度
            # low: 不需要缩放的低频部分
            # high: 需要缩放的高频部分
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # NOTE: 计算缩放因子
            # low之前，ramp为0，在high之后ramp为1
            # torch.clamp() 将input的值限制在[min, max]之间
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


def main():
    # x = torch.tensor([1, 2, 3, 4, 5])
    # y = torch.tensor([10, 20, 30, 40, 50])
    # condition = x > 3
    # result = torch.where(condition, x, y)
    # print(result)

    # v1 = torch.tensor([1, 2, 3])
    # v2 = torch.tensor([4, 5, 6])
    # print(torch.outer(v1, v2))  # 外积
    t = torch.arange(9).reshape(3, 3)
    b = torch.clamp(t, 3, 6)
    print(t)
    print(b)


if __name__ == "__main__":
    main()
