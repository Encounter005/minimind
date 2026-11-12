"""数据集实现：import 本包即完成注册。"""

from . import hf, memmap  # noqa: F401

__all__ = ["hf", "memmap"]
