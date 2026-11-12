"""内置模型：import 本包即完成注册。

其他模型（仓库根 src/ 下的、自己写的）不在这里 import：
配置里用 [model] location = "src/xxx.py" 指定源码，import 时注册。
"""

from . import gpt, toy_rnn  # noqa: F401

__all__ = ["gpt", "toy_rnn"]
