"""注册表：名字 → 构建函数。models/ 与 data/ 在 import 时自我注册。"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable

from .configs import DataConfig, ModelConfig
from .protocols import BatchSource, Tokenizer, TrainableModel
from .utils import ConfigError, suggest

MODELS: dict[str, Callable[[ModelConfig, int], TrainableModel]] = {}
DATASETS: dict[str, Callable[[DataConfig], BatchSource]] = {}
TOKENIZERS: dict[str, Callable[[str], Tokenizer]] = {}


def register_model(name: str) -> Callable:
    """@register_model("gpt") 标在构建函数上，import 该模块即生效。"""

    def deco(fn: Callable[[ModelConfig, int], TrainableModel]) -> Callable:
        MODELS[name] = fn
        return fn

    return deco


def register_dataset(name: str) -> Callable:
    def deco(fn: Callable[[DataConfig], BatchSource]) -> Callable:
        DATASETS[name] = fn
        return fn

    return deco


def register_tokenizer(kind: str) -> Callable:
    """@register_tokenizer("mine") 标在 "参数 -> Tokenizer" 的构建函数上，配置里可写 "mine:参数"。"""

    def deco(fn: Callable[[str], Tokenizer]) -> Callable[[str], Tokenizer]:
        TOKENIZERS[kind] = fn
        return fn

    return deco


def load_source(location: str, where: str) -> None:
    """import 一个 .py 源码文件，让里面的 @register_* 生效。

    路径来自 [model] location 或 [data] tokenizer_location，所以扩展不用改 my_trainer 源码。
    """
    name = f"_trainer_source_{Path(location).stem}"
    if name in sys.modules:  # 两个字段指向同一个文件时只执行一次
        return
    try:
        loader = importlib.util.spec_from_file_location(name, location)
        module = importlib.util.module_from_spec(loader)  # type: ignore[arg-type]
        sys.modules[name] = module
        loader.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001 - 源码文件自身的报错，统一转中文提示
        sys.modules.pop(name, None)
        raise ConfigError(f"{where} 加载 {location!r} 失败：{exc}") from exc


def build_model(cfg: ModelConfig, vocab_size: int) -> TrainableModel:
    from . import models  # noqa: F401  import 即注册

    if cfg.location:
        load_source(cfg.location, "[model] location")
    if cfg.name not in MODELS:
        raise ConfigError(f"未注册的模型 {cfg.name!r}{suggest(cfg.name, MODELS)}")
    return MODELS[cfg.name](cfg, vocab_size)


def build_dataset(cfg: DataConfig) -> BatchSource:
    from . import data  # noqa: F401  import 即注册

    if cfg.tokenizer_location:
        load_source(cfg.tokenizer_location, "[data] tokenizer_location")
    if cfg.reader not in DATASETS:
        raise ConfigError(f"未注册的数据集实现 {cfg.reader!r}{suggest(cfg.reader, DATASETS)}")
    return DATASETS[cfg.reader](cfg)
