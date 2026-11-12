"""小工具：错误类型、入口参数、种子、设备与精度。"""

from __future__ import annotations

import difflib
import random
import sys
import traceback
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch


class ConfigError(ValueError):
    """配置写错：键名、类型或取值不合法。"""


class DataError(RuntimeError):
    """数据缺失或格式不对。"""


def config_path_from_argv(default: str = "configs/debug_cpu.toml") -> Path:
    """入口约定：argv[1] 是配置文件，多传参数直接报错。"""
    if len(sys.argv) > 2:
        raise ConfigError(f"只接受一个配置文件路径，收到多余参数：{sys.argv[2:]}")
    return Path(sys.argv[1]) if len(sys.argv) == 2 else Path(default)


def suggest(name: str, candidates: Iterable[str]) -> str:
    """拼错时的中文提示：给最接近的名字，或列出全部可用项。"""
    candidates = list(candidates)
    hit = difflib.get_close_matches(name, candidates, n=1, cutoff=0.5)
    return f"，是不是想写 {hit[0]!r}？" if hit else f"，可用项：{sorted(candidates)}"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> tuple[torch.device, str]:
    """auto 自动选卡；返回 (device, device_type)，后者给 autocast / fused 用。"""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ConfigError("配置要求 cuda，但当前环境没有可用 GPU；把 [run] device 改成 cpu 或 auto")
    return torch.device(name), name.split(":")[0]


def amp_dtype(amp: str) -> torch.dtype | None:
    return {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[amp]


def run_entry(fn: Callable[[], object]) -> None:
    """四个入口统一的错误处理：配置/数据错误给中文提示，未预期异常保留 traceback。"""
    try:
        fn()
    except (ConfigError, DataError) as exc:
        print(f"配置/数据错误：{exc}")
        sys.exit(1)
    except Exception:  # noqa: BLE001 - 入口需要兜底，否则用户只看到半截堆栈
        traceback.print_exc()
        sys.exit(2)
