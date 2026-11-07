#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch import Tensor


def get_rotary_frequencies(dim: int, theta: float = 1e6):
    """
    生成rope旋转频率
    args:
        dim: 嵌入维度（必须是偶数）
        theta: 基础频率参数

    return:
        freqs: shape[dim // 2] 每个维度对的频率
    """

    # theta_i = 10000 ^(-2i / d), i = 0, 1, ...., d / 2 - 1
    i = torch.arange(0, dim // 2, dtype=torch.float32)
    return theta ** (-2 * i / dim)


def build_angles(freqs: Tensor, seq_len: int):
    """
    每个位置的角度 = position * frequency
    args:
        freqs: shape[dim // 2]
        seq_len: 序列长度

    return:
        angles: shape[seq_len, dim // 2]
    """

    # 生成位置索引
    positions = torch.arange(seq_len, dtype=torch.float32)

    # 计算每个位置的角度: position * frequency
    return torch.outer(positions, freqs)  # [seq_len, dim // 2]


def build_cos_sin(angles: Tensor, attention_factor: float = 1.0):
    """
    由角度表生成 cos / sin 查找表
    args:
        angles: shape[seq_len, dim // 2]
        attention_factor: 注意力修正因子
            普通 RoPE 恒为 1.0；YaRN 用它修正缩放后的注意力分数分布

    return:
        cos: shape[seq_len, dim]
        sin: shape[seq_len, dim]
    """

    cos = torch.cos(angles) * attention_factor
    sin = torch.sin(angles) * attention_factor

    # 将[seq_len, dim // 2]维度扩展成[seq_len, dim]
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

    return cos, sin


class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 1e6):
        '''
        args:
            dim: 每个注意力头的维度
            max_seq_len: 最大序列长度
            theta: 基础频率参数

        '''
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 预计算cos sin的值并缓存
        # 频率表由 _build_angles() 决定，子类覆写它就能换用别的频率表（如 YaRN）
        angles, attention_factor = self._build_angles()
        cos, sin = build_cos_sin(angles, attention_factor)
        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)

    def _build_angles(self):
        """
        模板方法：返回角度表和注意力修正因子
        子类覆写此方法即可改变频率表，cos/sin 构造与旋转逻辑由基类统一处理

        return:
            angles: shape[max_seq_len, dim // 2]
            attention_factor: 注意力修正因子，普通 RoPE 恒为 1.0
        """

        freqs = get_rotary_frequencies(self.dim, self.theta)
        return build_angles(freqs, self.max_seq_len), 1.0

    def forward(self, q:Tensor, k:Tensor, positions: Tensor = None):
        """
        对 Query 和 Key 应用 RoPE

        Args:
            q: Query，shape (batch, seq_len, num_heads, head_dim)
            k: Key，shape (batch, seq_len, num_heads, head_dim)
            positions: 位置索引，默认为 [0, 1, 2, ..., seq_len-1]

        Returns:
            q_rot, k_rot: 旋转后的 Query 和 Key
        """
        seq_len = q.shape[1]

        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        q_rot, k_rot = self._apply_rotary_pos_emb(q, k, cos, sin)

        return q_rot, k_rot



    def _apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
        """
        RoPE旋转
        旋转公式：
            q' = q * cos + rotate_half(q) * sin
            k' = k * cos + rotate_half(k) * sin
        args:
            q: Query，shape (batch, seq_len, num_heads, head_dim)
            k: Key，shape (batch, seq_len, num_heads, head_dim)
            cos: shape (seq_len, head_dim)
            sin: shape (seq_len, head_dim)

        return:
            q_rot, k_rot: 旋转后的 Query 和 Key
        """

        def rotate_half(x: Tensor):
            """
            将向量的前半部分和后半部分交换，并对后半部分取负
            [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]
            """

            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        # 调整cos sin维度方便广播[seq_len, dim] -> [1, seq_len, 1, dim]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        # apply rotate
        q_embed: Tensor = (q * cos) + rotate_half(q) * sin
        k_embed: Tensor = (k * cos) + rotate_half(k) * sin

        return q_embed, k_embed


def main():
    rope = RoPE(dim=64, max_seq_len=4096)
    batch_size = 2
    seq_len = 128
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    q_rot, k_rot = rope(q, k)
    print(f'Q_rot shape is: {q_rot.shape}')
    print(f'K_rot shape is: {k_rot.shape}')

if __name__ == "__main__":
    main()
