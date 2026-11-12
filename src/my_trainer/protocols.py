"""四个契约：模型、数据、日志、分词器。新增实现只需满足这里的方法签名。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class TrainableModel(Protocol):
    """训练器眼里的模型：给 targets 返回全序列 logits + loss，不给只返回最后一步。"""

    block_size: int
    vocab_size: int

    def __call__(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]: ...

    def generate(
        self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None
    ) -> Tensor: ...

    def num_params(self) -> int: ...


class BatchSource(Protocol):
    """训练器眼里的数据：自己负责右移一位得到目标。"""

    vocab_size: int

    def get_batch(
        self, split: str, batch_size: int, block_size: int, device: str
    ) -> tuple[Tensor, Tensor]: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class Tokenizer(Protocol):
    """分词器契约：数据集只需要词表大小 + 编解码。自己训练的分词器满足这三样就能挂上去。"""

    vocab_size: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class Logger(Protocol):
    """训练器眼里的日志：一次调用带一组指标，写到哪儿由实现决定。"""

    def log(self, step: int, metrics: dict[str, float], split: str = "train") -> None: ...

    def close(self) -> None: ...
