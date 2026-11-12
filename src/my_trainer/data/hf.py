"""HuggingFace datasets 数据源：整段文本编码进内存，适合小语料（大语料请用 memmap）。"""

from __future__ import annotations

import numpy as np
import torch

from ..configs import DataConfig
from ..registry import register_dataset
from ..utils import DataError


class HFDataset:
    def __init__(self, cfg: DataConfig):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise DataError("读取 HF 数据集需要 datasets 包：uv add datasets") from exc
        import tiktoken

        # shortcut: 字段名写死 text、只用 train split、末尾 5% 当验证集，够教学用；需要别的字段再扩展配置
        try:
            ds = load_dataset(cfg.name, split="train")
        except Exception as exc:  # noqa: BLE001 - 数据集名/网络问题，给中文提示即可
            raise DataError(f"加载 HF 数据集 {cfg.name!r} 失败：{exc}") from exc
        text = "\n".join(t for t in ds["text"] if t)
        self._enc = tiktoken.get_encoding("gpt2")
        tokens = np.array(self._enc.encode_ordinary(text), dtype=np.uint16)
        cut = int(len(tokens) * 0.95)
        self.train, self.val = tokens[:cut], tokens[cut:]
        self.vocab_size = self._enc.n_vocab

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = {"train": self.train, "val": self.val}.get(split)
        if data is None:
            raise DataError(f"未知的 split：{split!r}（只能是 train / val）")
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)


@register_dataset("hf")
def build_hf(cfg: DataConfig) -> HFDataset:
    return HFDataset(cfg)
