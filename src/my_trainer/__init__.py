"""my_trainer：可插拔、单一配置来源的最小训练模板。

入口（都只接受一个可选的配置文件路径）：
    python -m my_trainer.train  [configs/debug_cpu.toml]
    python -m my_trainer.sample [configs/debug_cpu.toml]
    python -m my_trainer.bench  [configs/debug_cpu.toml]
    python -m my_trainer.sweep  [configs/sweep_lr.toml]
"""

__version__ = "0.1.0"
