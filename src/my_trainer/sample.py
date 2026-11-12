"""采样入口：python -m my_trainer.sample [config.toml]"""

from __future__ import annotations

from pathlib import Path

import torch

from .build import build_all
from .checkpoint import load_ckpt
from .configs import Config
from .utils import ConfigError, config_path_from_argv, run_entry


def generate_text(cfg: Config, model, dataset, device: torch.device) -> str:
    """训练结束时也会调用这里，采样逻辑只写一份。"""
    ids = dataset.encode(cfg.sample.prompt or "\n")[-cfg.data.block_size :] or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model.generate(
            idx, cfg.sample.max_new_tokens, temperature=cfg.sample.temperature, top_k=cfg.sample.top_k or None
        )
    if was_training:
        model.train()
    return dataset.decode(out[0].tolist())


def main() -> None:
    cfg = Config.load(config_path_from_argv())
    device, dataset, model = build_all(cfg)
    ckpt = Path(cfg.sample.ckpt) if cfg.sample.ckpt else Path(cfg.run.out_root) / cfg.run.name / "ckpt.pt"
    if not ckpt.exists():
        raise ConfigError(f"找不到存档 {ckpt}：先跑 python -m my_trainer.train")
    model.load_state_dict(load_ckpt(ckpt, device)["model"])
    text = generate_text(cfg, model, dataset, device)
    print(text)
    out_file = ckpt.parent / "samples.txt"
    with out_file.open("a", encoding="utf-8") as f:
        f.write(text + "\n" + "-" * 60 + "\n")
    print(f"\n（已追加到 {out_file}）")


if __name__ == "__main__":
    run_entry(main)
