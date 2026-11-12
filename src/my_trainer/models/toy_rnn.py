"""教学示例：Embedding + GRU + Linear。抄这个文件就能接一个新架构。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..configs import ModelConfig
from ..registry import register_model


class ToyRNN(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        self.block_size = cfg.block_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, cfg.n_embd)
        self.rnn = nn.GRU(cfg.n_embd, cfg.n_embd, batch_first=True)
        self.head = nn.Linear(cfg.n_embd, vocab_size)

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        hidden = self.rnn(self.embed(idx))[0]
        if targets is None:
            return self.head(hidden[:, [-1], :]), None  # 推理只算最后一步，和 GPT 的约定一致
        logits = self.head(hidden)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None
    ) -> Tensor:
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.block_size :])[0][:, -1, :] / max(temperature, 1e-6)
            if top_k:
                top, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top[:, [-1]]] = -float("inf")
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
        return idx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@register_model("toy_rnn")
def build_toy_rnn(cfg: ModelConfig, vocab_size: int) -> ToyRNN:
    return ToyRNN(cfg, vocab_size)
