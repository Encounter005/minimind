"""性能基准入口：python -m my_trainer.bench [config.toml]"""

from __future__ import annotations

import time

import torch

from .build import autocast_context, build_all
from .configs import Config
from .utils import config_path_from_argv, run_entry


def main() -> None:
    cfg = Config.load(config_path_from_argv())
    device, dataset, model = build_all(cfg)
    device_type = device.type
    ctx = autocast_context(cfg.train.amp, device_type)
    model.train()

    def one_step() -> None:
        x, y = dataset.get_batch("train", cfg.data.batch_size, cfg.data.block_size, device_type)
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)

    for _ in range(cfg.bench.warmup):
        one_step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(cfg.bench.iters):
        one_step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    per_step = (time.perf_counter() - t0) / cfg.bench.iters
    tokens_per_sec = cfg.data.batch_size * cfg.data.block_size / per_step
    print(
        f"参数量 {model.num_params() / 1e6:.2f}M | {cfg.bench.iters} 步平均 {per_step * 1000:.1f} ms/step "
        f"| {tokens_per_sec:,.0f} tokens/s"
    )
    print("（只做前向+反向，不写 checkpoint，不出日志）")


if __name__ == "__main__":
    run_entry(main)
