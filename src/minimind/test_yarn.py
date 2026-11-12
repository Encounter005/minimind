#!/usr/bin/env python3
import math
import torch
from torch import Tensor
from typing import Optional
from .RoPE import RoPE, get_rotary_frequencies


class YaRN(RoPE):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 32768,
        theta: float = 1e6,
        rope_scaling: Optional[dict] = None,
    ):
        self.rope_scalling = rope_scaling
        super().__init__(dim, max_seq_len, theta)

    def _get_yarn_frequencies(
        self,
        dim: int,
        seq_len: int,
        theta: float = 1e6,
        rope_scaling: Optional[dict] = None,
    ):
        freqs = get_rotary_frequencies(dim, theta)

    def _get_attn_factor(self, rope_scaling: Optional[dict] = None) -> float:
        if rope_scaling is not None:
            return rope_scaling.get("attention_factor", 1.0)
        return 1.0

    def _apply_yarn_scaling(
        self,
        freqs: Tensor,
        dim: int,
        theta: float,
        seq_len: int,
        rope_scaling: Optional[dict] = None,
    ):
        if rope_scaling is None:
            return freqs

        original_max_position_embeddings = rope_scaling.get(
            "original_max_position_embeddings", 2048
        )
        factor = rope_scaling.get("factor", 16)

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

    @staticmethod
    def _compute_yarn_ramp(
        dim: int,
        theta: float,
        original_max_position_embddings: int = 2048,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        def wavelength_to_dim(wavelength: float) -> float:
            return (
                dim
                * math.log(original_max_position_embddings / (wavelength * 2 * math.pi))
                / (2 * math.log(theta))
            )

        low = max(math.floor(wavelength_to_dim(beta_fast)), 0)
        high = min(math.ceil(wavelength_to_dim(beta_slow)), dim // 2 - 1)

        ramp = torch.clamp(
            (torch.arange(dim // 2, dtype=torch.float32) - low)
            / max(high - low, 0.001),
            0,
            1,
        )

        return ramp

    def _build_angles(self):
        angles = self._get_yarn_frequencies(
            self.dim, self.max_seq_len, self.theta, self.rope_scaling
        )

        return angles, self._get_attn_factor(self.rope_scaling)
