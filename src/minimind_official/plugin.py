"""官方 MiniMind（model/model_minimind.py 的 MiniMindForCausalLM）的 my_trainer 适配器。

配置里这样用（仓库根目录运行，cwd 要在仓库根，否则 import 不到 model 包）：

    [model]
    name = "minimind_official"
    location = "src/minimind_official/plugin.py"

官方模型本身就是契约的形状，要垫的只有三处差异：
  1. forward 返回 MoeCausalLMOutputWithPast，且 loss 要传 labels 才在内部算；
     而训练器的 targets 已经右移过一位，所以这里直接拿全序列 logits 自己算 CE。
  2. 不传 targets 时用 logits_to_keep=1，只算最后一步（否则采样要把整段前缀都过一次 lm_head）。
  3. 官方 generate 的第二个位置参数是 attention_mask，不是 max_new_tokens，必须包一层转发。

权重初始化、tie_word_embeddings、RoPE、MoE 全是官方代码（PreTrainedModel.post_init → initializer_range=0.02），
本文件不接管，所以 tie 生效后参数量与官方发布权重一致。

NOTE: my_trainer.registry.load_source 按源码路径把本文件当顶层模块导入，这里只能用绝对 import。
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from my_trainer.configs import ModelConfig
from my_trainer.registry import register_model

# MiniMindConfig 真正会读的字段（除 hidden_size / num_hidden_layers / dropout 这些 [model] 主字段外）。
# 白名单的意义：写错名字直接报错，而不是被 PretrainedConfig 静默挂成无用属性。
_EXTRA_FIELDS = {
    "bos_token_id", "eos_token_id", "flash_attn", "head_dim", "hidden_act",
    "intermediate_size", "inference_rope_scaling", "max_position_embeddings",
    "moe_intermediate_size", "norm_topk_prob", "num_experts", "num_experts_per_tok",
    "num_key_value_heads", "rms_norm_eps", "rope_theta", "router_aux_loss_coef",
    "tie_word_embeddings",
}


class TrainerMiniMindOfficial(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        extras = dict(cfg.extras or {})
        use_moe = bool(extras.pop("use_moe", False))  # MiniMindConfig 的显式参数，不在 kwargs 里
        unknown = set(extras) - _EXTRA_FIELDS
        if unknown:
            raise ValueError(
                f"[model.extras] 里有 MiniMindConfig 不认识的字段 {sorted(unknown)}；"
                f"可用字段：{sorted(_EXTRA_FIELDS | {'use_moe'})}"
            )
        self.block_size = cfg.block_size
        self.vocab_size = vocab_size
        self.config = MiniMindConfig(
            hidden_size=cfg.n_embd,
            num_hidden_layers=cfg.n_layer,
            use_moe=use_moe,
            vocab_size=vocab_size,
            num_attention_heads=cfg.n_head,
            dropout=cfg.dropout,
            **extras,
        )
        self.inner = MiniMindForCausalLM(self.config)  # 构造末尾的 post_init() 负责 0.02 初始化与 tie_weights

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        out = self.inner(idx, logits_to_keep=1 if targets is None else 0)
        if targets is None:  # 契约：不传 targets 时只返回最后一步
            return out.logits, None
        logits = out.logits
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        # MoE 的路由负载均衡项；dense 模型这个字段恒为 0，加上无副作用。
        return logits, loss + out.aux_loss

    def generate(
        self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None
    ) -> Tensor:
        # 走官方 generate（自带 KV cache / 遇 eos 提前结束）；关掉 top_p，采样只由 temperature 与 top_k 决定。
        return self.inner.generate(
            inputs=idx,
            max_new_tokens=max_new_tokens,
            temperature=max(float(temperature), 1e-6),
            top_k=int(top_k or 0),
            top_p=1.0,
            do_sample=temperature > 0,
        )

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@register_model("minimind_official")
def build_minimind_official(cfg: ModelConfig, vocab_size: int) -> TrainerMiniMindOfficial:
    return TrainerMiniMindOfficial(cfg, vocab_size)
