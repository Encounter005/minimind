"""配置：dataclass 树 + 一份 TOML。文件是唯一来源，命令行不覆盖任何字段。"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from .utils import ConfigError, suggest


@dataclass(frozen=True)
class DataConfig:
    name: str = "shakespeare_char"  # 数据集目录名，位于 root 下
    root: str = "data"
    reader: str = "memmap"  # 数据集实现（注册名）
    tokenizer: str = ""  # 空=有 meta.pkl 用字符级、否则 GPT-2；或 "类型:参数"，如 "tiktoken:p50k_base" / "hf:仓库id"
    tokenizer_location: str = ""  # 分词器实现源码（.py 路径），import 即注册自定义类型
    batch_size: int = 12
    block_size: int = 256


@dataclass(frozen=True)
class ModelConfig:
    name: str = "gpt"
    location: str = ""  # 模型实现源码（.py 路径），import 即完成 @register_model
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    extras: dict[str, Any] = field(default_factory=dict)  # 交给具体模型自解释的额外超参


@dataclass(frozen=True)
class OptimConfig:
    lr: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_iters: int = 50
    lr_decay_iters: int = 500


@dataclass(frozen=True)
class TrainConfig:
    max_iters: int = 500
    grad_accum: int = 1
    amp: str = "none"  # none | bf16 | fp16
    compile: bool = False  # torch.compile 训练前向；采样仍走未编译模型
    eval_interval: int = 100
    eval_iters: int = 20
    log_interval: int = 10
    seed: int = 1337


@dataclass(frozen=True)
class RunConfig:
    out_root: str = "runs"
    name: str = "debug"
    device: str = "auto"  # auto | cpu | cuda
    resume: str = "auto"  # auto | none | 存档路径
    save_interval: int = 100
    sample_after_train: bool = True


@dataclass(frozen=True)
class LogConfig:
    console: bool = True
    progress: bool = True  # 用 tqdm 进度条代替逐步打印（VAL 结果进 postfix）
    file: bool = True
    tensorboard: bool = True
    graph: bool = False  # 把模型计算图写进 TensorBoard（trace 一次前向）
    wandb: bool = False
    wandb_project: str = "my_trainer-template"


@dataclass(frozen=True)
class SampleConfig:
    ckpt: str = ""  # 空 = runs/<run.name>/ckpt.pt
    prompt: str = "\n"
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: int = 0  # 0 = 不截断


@dataclass(frozen=True)
class BenchConfig:
    iters: int = 10
    warmup: int = 3


@dataclass(frozen=True)
class SweepConfig:
    name: str = "sweep"
    mode: str = "grid"  # grid | random
    space: dict[str, list] = field(default_factory=dict)
    num_samples: int = 8
    seed: int = 0


# 小节名字 → 配置类，顺序即 TOML 输出顺序
SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "model": ModelConfig,
    "optim": OptimConfig,
    "train": TrainConfig,
    "run": RunConfig,
    "log": LogConfig,
    "sample": SampleConfig,
    "bench": BenchConfig,
    "sweep": SweepConfig,
}


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run: RunConfig = field(default_factory=RunConfig)
    log: LogConfig = field(default_factory=LogConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"找不到配置文件：{path}")
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path} 不是合法 TOML：{exc}") from exc
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        if not isinstance(raw, dict):
            raise ConfigError("配置顶层必须是表（[section]）")
        unknown = [key for key in raw if key not in SECTIONS]
        if unknown:
            raise ConfigError(f"未知配置小节 {unknown[0]!r}{suggest(unknown[0], SECTIONS)}")
        cfg = cls(**{name: _section(cls_, raw.get(name, {}), name) for name, cls_ in SECTIONS.items()})
        cfg.validate()
        return cfg

    def validate(self) -> None:
        m, d, o, t, r, s = self.model, self.data, self.optim, self.train, self.run, self.sweep
        if m.n_embd % m.n_head != 0:
            raise ConfigError(f"[model] n_embd={m.n_embd} 必须能被 n_head={m.n_head} 整除")
        if d.block_size > m.block_size:
            raise ConfigError(f"[data] block_size={d.block_size} 超过 [model] block_size={m.block_size}")
        if d.tokenizer:
            kind, _, param = d.tokenizer.partition(":")
            if not (kind.strip() and param.strip()):
                raise ConfigError(
                    f"[data] tokenizer 需要写成 \"<类型>:<参数>\"，例如 \"tiktoken:gpt2\" / \"hf:仓库 id\"；"
                    f"自定义类型用 [data] tokenizer_location 指定实现源码，收到 {d.tokenizer!r}"
                )
        for where, loc in (("[model] location", m.location), ("[data] tokenizer_location", d.tokenizer_location)):
            if loc and not (loc.endswith(".py") and Path(loc).is_file()):
                raise ConfigError(f"{where} 需要一个存在的 .py 路径（相对仓库根），收到 {loc!r}")
        if t.amp not in ("none", "bf16", "fp16"):
            raise ConfigError(f"[train] amp 只能是 none / bf16 / fp16，收到 {t.amp!r}")
        if t.amp == "fp16" and r.device == "cpu":
            raise ConfigError("[train] amp=\"fp16\" 需要 GPU；CPU 上请用 bf16 或 none")
        if o.min_lr > o.lr:
            raise ConfigError(f"[optim] min_lr={o.min_lr} 不能大于 lr={o.lr}")
        if t.max_iters < 1 or t.grad_accum < 1 or t.eval_interval < 1 or t.log_interval < 1:
            raise ConfigError("[train] max_iters / grad_accum / eval_interval / log_interval 必须 >= 1")
        if r.resume not in ("auto", "none") and not Path(r.resume).exists():
            raise ConfigError(f"[run] resume 指向的存档不存在：{r.resume}")
        if s.mode not in ("grid", "random"):
            raise ConfigError(f"[sweep] mode 只能是 grid / random，收到 {s.mode!r}")
        for key, values in s.space.items():
            self._check_space_key(key)
            if not isinstance(values, list) or not values:
                raise ConfigError(f"[sweep] space[{key!r}] 需要一个非空列表，例如 [1e-3, 3e-4]")

    def _check_space_key(self, dotted: str) -> None:
        parts = dotted.split(".")
        if len(parts) != 2:
            raise ConfigError(f"[sweep] space 的键只能形如 'optim.lr'，收到 {dotted!r}")
        section = SECTIONS.get(parts[0])
        if section is None:
            raise ConfigError(f"[sweep] 未知小节 {parts[0]!r}{suggest(parts[0], SECTIONS)}")
        hints = get_type_hints(section)
        if parts[1] not in hints:
            raise ConfigError(f"[sweep] {parts[0]} 里没有 {parts[1]!r}{suggest(parts[1], hints)}")

    def to_toml(self) -> str:
        """配置快照：写进 run 目录，复制这一份就能复现整次实验。"""
        lines = ["# my_trainer 配置快照：这一份文件就是本次实验的完整定义", ""]
        for name, cls in SECTIONS.items():
            lines.append(f"[{name}]")
            for f in fields(cls):
                lines.append(f"{f.name} = {_toml_value(getattr(getattr(self, name), f.name))}")
            lines.append("")
        return "\n".join(lines)


def _section(cls: type, raw: Any, name: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"[{name}] 必须是表")
    hints = get_type_hints(cls)
    unknown = [key for key in raw if key not in hints]
    if unknown:
        raise ConfigError(f"[{name}] 里出现了未知配置项 {unknown[0]!r}{suggest(unknown[0], hints)}")
    return cls(**{key: _coerce(value, hints[key], f"[{name}] {key}") for key, value in raw.items()})


def _coerce(value: Any, tp: Any, where: str) -> Any:
    """按 dataclass 声明的类型校验并转换（TOML 没有元组，列表要转成元组）。"""
    origin = get_origin(tp)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where} 需要列表，例如 [0.9, 0.95]，收到 {value!r}")
        args = get_args(tp)
        return tuple(
            _coerce(v, args[i] if i < len(args) else args[-1], f"{where}[{i}]") for i, v in enumerate(value)
        )
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where} 需要表，收到 {value!r}")
        key_t, val_t = get_args(tp) or (str, Any)
        out = {}
        for k, v in value.items():
            out[_coerce(k, key_t, f"{where} 的键")] = v if val_t is Any else _coerce(v, val_t, f"{where}[{k!r}]")
        return out
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where} 需要列表，收到 {value!r}")
        item_t = (get_args(tp) or (Any,))[0]
        return [_coerce(v, item_t, f"{where}[{i}]") for i, v in enumerate(value)]
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where} 需要 true/false，收到 {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} 需要整数，收到 {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where} 需要数字，收到 {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where} 需要字符串，收到 {value!r}")
        return value
    return value


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        body = ", ".join(f"{json.dumps(str(k), ensure_ascii=False)} = {_toml_value(v)}" for k, v in value.items())
        return "{ " + body + " }"
    raise ConfigError(f"无法写入 TOML 的值：{value!r}")
