"""分词器适配：把不同来源的 encode/decode 统一成 BatchSource 需要的样子。

[data] tokenizer 的写法是 "<类型>:<参数>"，参数原样交给该类型的构建函数：
    ""                        自动：有 meta.pkl 用字符级字表，否则 GPT-2 BPE（保持旧行为）
    "tiktoken:p50k_base"      tiktoken 内置编码（gpt2 / r50k_base / p50k_base；词表须 ≤ 65535）
    "hf:bert-base-uncased"    HuggingFace tokenizer（仓库 id 或本地目录；首次可能需要网络）

自己训练的分词器：写一个类满足 Tokenizer（vocab_size + encode/decode），用
@register_tokenizer("类型名") 注册 "参数 -> Tokenizer" 的构建函数，配置里就能写 "类型名:参数"。
实现写在哪个文件都行，把路径填进配置的 [data] tokenizer_location 即可，不用改这里。

注意：分词器必须和数据准备时用的一致，否则 id 能对上、文本解出来是乱码。
"""

from __future__ import annotations

from ..protocols import Tokenizer
from ..registry import TOKENIZERS, register_tokenizer
from ..utils import DataError, suggest


class _TiktokenTokenizer:
    def __init__(self, name: str):
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - tiktoken 是直接依赖
            raise DataError("使用 tiktoken 分词器需要 tiktoken（uv add tiktoken）") from exc
        try:
            self._enc = tiktoken.get_encoding(name)
        except ValueError as exc:
            available = ", ".join(tiktoken.list_encoding_names())
            raise DataError(f"未知的 tiktoken 编码 {name!r}；可用：{available}") from exc
        self.vocab_size = self._enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)


class _HFTokenizer:
    def __init__(self, repo_id: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - transformers 是直接依赖
            raise DataError("使用 HF 分词器需要 transformers（uv add transformers）") from exc
        try:
            self._tok = AutoTokenizer.from_pretrained(repo_id)
        except Exception as exc:  # noqa: BLE001 - 网络/仓库名/缓存问题，统一转中文提示
            raise DataError(f"加载 HF 分词器 {repo_id!r} 失败（需要网络或本地缓存）：{exc}") from exc
        self.vocab_size = len(self._tok)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


@register_tokenizer("tiktoken")
def _build_tiktoken(name: str) -> Tokenizer:
    return _TiktokenTokenizer(name)


@register_tokenizer("hf")
def _build_hf(repo_id: str) -> Tokenizer:
    return _HFTokenizer(repo_id)


def build_tokenizer(spec: str) -> Tokenizer:
    """把 "类型:参数" 交给注册的构建函数；空串由调用方决定。"""
    kind, _, param = spec.partition(":")
    kind, param = kind.strip(), param.strip()
    if not param:
        raise DataError(f"[data] tokenizer 写法应为 '类型:参数'，例如 'tiktoken:gpt2'，收到 {spec!r}")
    if kind not in TOKENIZERS:
        raise DataError(f"未注册的分词器类型 {kind!r}{suggest(kind, TOKENIZERS)}")
    return TOKENIZERS[kind](param)
