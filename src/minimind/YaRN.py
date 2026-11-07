#!/usr/bin/env python3
import math
from typing import Optional

import torch
from torch import Tensor

try:  # 作为包导入：from src.minimind.YaRN import YaRN
    from .RoPE import RoPE, build_angles, get_rotary_frequencies
except ImportError:  # 直接运行脚本：python YaRN.py
    from RoPE import RoPE, build_angles, get_rotary_frequencies


class YaRN(RoPE):
    """
    带 YaRN 长度外推的旋转位置编码：本质是「换了一张频率表的 RoPE」

    继承 RoPE 后只需要覆写 _build_angles() 这一个模板方法，
    cos/sin 查找表的构造、按 seq_len 切片和旋转计算全部复用基类，
    所以 forward / _apply_rotary_pos_emb 在这里不再重复实现。
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 32768,
        theta: float = 1e6,
        rope_scaling: Optional[dict] = None,
    ):
        """
        带 YaRN 长度外推的旋转位置编码（RoPE 的扩展）

        args:
            dim: 每个注意力头的维度
            max_seq_len: 最大序列长度（超出训练长度时自动启用 YaRN 缩放）
            theta: 基础频率参数
            rope_scaling: YaRN 缩放配置，例如:
                {
                    "original_max_position_embeddings": 2048,  # 模型训练时的最大长度
                    "factor": 16,                              # 上下文扩展倍数
                    "beta_fast": 32.0,                         # 低频分界（波长基准）
                    "beta_slow": 1.0,                          # 高频分界（波长基准）
                    "attention_factor": 1.0,                   # 注意力分数修正因子
                }
                为 None 时退化为普通 RoPE
        """
        # 必须在 super().__init__() 之前赋值：基类构造时会调用 _build_angles() 读取它
        # （普通属性在 Module.__init__ 之前赋值是安全的，只有 Parameter / buffer 不行）
        self.rope_scaling = rope_scaling
        super().__init__(dim, max_seq_len, theta)

    def _build_angles(self):
        """
        覆写基类的模板方法：返回 YaRN 缩放后的角度表和注意力修正因子

        return:
            angles: shape [max_seq_len, dim // 2]
            attention_factor: 注意力分数修正因子
        """
        angles = self._get_yarn_frequencies(
            self.dim, self.max_seq_len, self.theta, self.rope_scaling
        )
        return angles, self._get_attention_factor(self.rope_scaling)

    def _get_attention_factor(self, rope_scaling: Optional[dict] = None) -> float:
        """
        YaRN 的注意力修正因子

        缩放频率后注意力分数的分布会改变，attention_factor 用于修正：
        cos/sin 查找表整体乘以该因子，从而修正注意力分数
        未启用 YaRN 时为 1.0，即不修正

        args:
            rope_scaling: YaRN 缩放配置

        return:
            attention_factor: 注意力修正因子
        """
        if rope_scaling is None:
            return 1.0
        return rope_scaling.get("attention_factor", 1.0)

    @staticmethod
    def _compute_yarn_ramp(
        dim: int,
        theta: float,
        original_max_position_embeddings: int = 2048,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        """
        计算 YaRN 的线性插值系数 ramp

        YaRN 的核心思想：按波长把维度对分成三组
            - 高频维度（索引小于 low）：波长很短，不缩放，保留局部位置信息
            - 低频维度（索引大于 high）：波长很长，完全缩放（频率除以 factor，波长放大 factor 倍），
            使模型能外推到更长的上下文
            - 中间维度：在 [low, high] 之间线性插值，平滑过渡

        args:
            dim: 嵌入维度
            theta: 基础频率参数
            original_max_position_embeddings: 模型训练时的最大长度
            beta_fast: 低频分界（波长基准）
            beta_slow: 高频分界（波长基准）

        return:
            ramp: shape [dim // 2]，每个维度对的插值系数，范围 [0, 1]
                ramp = 0 表示频率不变，ramp = 1 表示完全缩放
        """

        # 给定波长边界 b，反解出对应的维度索引（波长越大，索引越大）
        def wavelength_to_dim(wavelength: float) -> float:
            return (
                dim
                * math.log(
                    original_max_position_embeddings / (wavelength * 2 * math.pi)
                )
            ) / (2 * math.log(theta))

        # 划分高低维度：
        # 索引小于 low 的维度对（高频段）不缩放，索引大于 high 的维度对（低频段）完全缩放
        low = max(math.floor(wavelength_to_dim(beta_fast)), 0)
        high = min(math.ceil(wavelength_to_dim(beta_slow)), dim // 2 - 1)

        # 在 [low, high] 之间线性插值，并裁剪到 [0, 1]
        ramp = torch.clamp(
            (torch.arange(dim // 2, dtype=torch.float32) - low)
            / max(high - low, 0.001),
            0,
            1,
        )
        return ramp

    def _apply_yarn_scaling(
        self,
        freqs: Tensor,
        dim: int,
        theta: float,
        seq_len: int,
        rope_scaling: Optional[dict] = None,
    ):
        """
        对频率应用 YaRN 缩放
        f'(i) = f(i) * ((1 - ramp) + ramp / factor)

        注意：只有当 seq_len 超过训练长度 original_max_position_embeddings 时才缩放，
        训练长度内保持原始 RoPE 频率不变

        args:
            freqs: 基础频率，shape [dim // 2]
            dim: 嵌入维度
            theta: 基础频率参数
            seq_len: 序列长度
            rope_scaling: YaRN 缩放配置，None 时不缩放

        return:
            freqs: 缩放后的频率，shape [dim // 2]
        """
        if rope_scaling is None:
            return freqs

        original_max_position_embeddings = rope_scaling.get(
            "original_max_position_embeddings", 2048
        )
        factor = rope_scaling.get("factor", 16)

        # 推断长度不大于训练长度时不需要缩放
        if seq_len / original_max_position_embeddings <= 1.0:
            return freqs

        ramp = self._compute_yarn_ramp(
            dim,
            theta,
            original_max_position_embeddings,
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
        )
        return freqs * (1 - ramp + ramp / factor)

    def _get_yarn_frequencies(
        self,
        dim: int,
        seq_len: int,
        theta: float = 1e6,
        rope_scaling: Optional[dict] = None,
    ):
        """
        生成 YaRN 缩放后的旋转频率角度
        angle = position * freq'，其中 freq' 是缩放后的频率

        args:
            dim: 嵌入维度（必须是偶数）
            seq_len: 序列长度
            theta: 基础频率参数
            rope_scaling: YaRN 缩放配置，None 时等价于普通 RoPE

        return:
            angles: shape [seq_len, dim // 2] 每个位置每个维度对的旋转角度
        """
        # 1. RoPE 基础频率 theta^(-2i/dim)
        freqs = get_rotary_frequencies(dim, theta)

        # 2. 按 YaRN 规则缩放频率（超过训练长度时才生效）
        freqs = self._apply_yarn_scaling(freqs, dim, theta, seq_len, rope_scaling)

        # 3. 计算每个位置的旋转角度 position * frequency
        return build_angles(freqs, seq_len)


def main():
    dim, max_seq_len, num_heads = 64, 4096, 8

    # YaRN 配置：训练长度 2048，扩展 2 倍
    rope_scaling = {
        "original_max_position_embeddings": 2048,
        "factor": 2,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    }

    yarn = YaRN(dim=dim, max_seq_len=max_seq_len, theta=1e6, rope_scaling=rope_scaling)
    rope = RoPE(dim=dim, max_seq_len=max_seq_len, theta=1e6)

    # 继承的效果：YaRN 只是换了一张频率表，旋转与 forward 全部来自 RoPE
    assert isinstance(yarn, RoPE), "YaRN 应继承自 RoPE"
    # rope_scaling=None 时必须与 RoPE 完全一致（退化检查）
    yarn_plain = YaRN(dim=dim, max_seq_len=max_seq_len, theta=1e6)
    assert torch.equal(yarn_plain.cos_cached, rope.cos_cached), (
        "rope_scaling=None 时应等同 RoPE"
    )

    # 查看 YaRN 的插值系数：索引小的维度对不缩放，索引大的维度对完全缩放，中间线性过渡
    ramp = YaRN._compute_yarn_ramp(dim, 1e6, 2048, 32.0, 1.0)
    print(f"ramp: {[round(r, 2) for r in ramp.tolist()]}")

    batch_size, seq_len = 2, 128
    q = torch.randn(batch_size, seq_len, num_heads, dim)
    k = torch.randn(batch_size, seq_len, num_heads, dim)

    q_yarn, k_yarn = yarn(q, k)
    q_rope, _ = rope(q, k)
    print(f"Q_rot shape is: {q_yarn.shape}")
    print(f"K_rot shape is: {k_yarn.shape}")
    # 位置 0 的旋转角度恒为 0，两种编码在该位置完全一致
    print(f"位置 0 相同: {torch.allclose(q_yarn[:, :1], q_rope[:, :1])}")
    # 其余位置因频率缩放产生差异
    print(f"其余位置有差异: {not torch.allclose(q_yarn[:, 1:], q_rope[:, 1:])}")


if __name__ == "__main__":
    main()
