"""训练入口：python -m my_trainer.train [config.toml]"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .build import autocast_context, build_all, build_optimizer
from .checkpoint import load_ckpt, resolve_resume, save_ckpt
from .configs import Config, OptimConfig
from .loggers import build_loggers
from .sample import generate_text
from .utils import ConfigError, config_path_from_argv, run_entry, seed_all


def lr_at(step: int, cfg: OptimConfig) -> float:
    """warmup 线性升到 lr，随后 cosine 降到 min_lr。"""
    if step < cfg.warmup_iters:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_iters)
    if step > cfg.lr_decay_iters:
        return cfg.min_lr
    ratio = (step - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def evaluate(model, dataset, cfg: Config, device_type: str, ctx) -> dict[str, float]:
    model.eval()
    total_loss = total_acc = 0.0
    for _ in range(cfg.train.eval_iters):
        x, y = dataset.get_batch("val", cfg.data.batch_size, cfg.data.block_size, device_type)
        with ctx:
            logits, loss = model(x, y)
        total_loss += loss.item()
        total_acc += (logits.argmax(-1) == y).float().mean().item()
    model.train()
    loss = total_loss / cfg.train.eval_iters
    return {"loss": loss, "perplexity": math.exp(loss), "accuracy": total_acc / cfg.train.eval_iters}


def train(cfg: Config) -> Path:
    seed_all(cfg.train.seed)
    device, dataset, model = build_all(cfg)
    device_type = device.type
    raw_model = model  # 未包装的模型：存档、加载、采样都用它（torch.compile 会给键加 _orig_mod. 前缀）
    if cfg.train.compile:
        print("编译模型（torch.compile）：首次前向会慢几十秒……")
        model = torch.compile(model)
        raw_model = getattr(model, "_orig_mod", model)
    out_dir = Path(cfg.run.out_root) / cfg.run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    loggers = build_loggers(cfg.log, out_dir, cfg.run.name, {"config_toml": cfg.to_toml()})
    if cfg.log.graph:  
        sample = torch.randint(dataset.vocab_size, (1, cfg.data.block_size), device=device)
        for logger in loggers:
            add_graph = getattr(logger, "add_graph", None)
            if add_graph is not None:
                add_graph(raw_model, sample)

    optimizer = build_optimizer(raw_model, cfg.optim, device_type)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp == "fp16" and device_type == "cuda")
    ctx = autocast_context(cfg.train.amp, device_type)

    start_step, best_val = 0, float("inf")
    resume_path = resolve_resume(cfg, out_dir)
    if resume_path is not None:
        state = load_ckpt(resume_path, device)
        try:
            raw_model.load_state_dict(state["model"])
        except RuntimeError as exc:
            raise ConfigError(
                f"存档 {resume_path} 与当前配置不匹配（结构或参数量不同）：换回对应配置，或把 [run] resume 改成 none"
            ) from exc
        optimizer.load_state_dict(state["optimizer"])
        if scaler.is_enabled() and state["scaler"] is not None:
            scaler.load_state_dict(state["scaler"])
        start_step, best_val = state["step"] + 1, state["best_val_loss"]
        print(f"接续 {resume_path}：从第 {start_step} 步继续，历史最好 val loss {best_val:.4f}")

    # 配置快照＝复现入口；放在续训校验之后，避免失败的 resume 覆盖上一次的记录
    (out_dir / "config.toml").write_text(cfg.to_toml(), encoding="utf-8")
    model.train()
    ckpt_path = out_dir / "ckpt.pt"
    t0, tokens = time.perf_counter(), 0
    use_bar = cfg.log.progress and cfg.log.console and cfg.train.max_iters > start_step
    steps = tqdm(
        range(start_step, cfg.train.max_iters), desc=cfg.run.name, unit="step", dynamic_ncols=True
    ) if use_bar else range(start_step, cfg.train.max_iters)
    for step in steps:
        lr = lr_at(step, cfg.optim)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_sum = acc_sum = 0.0
        # 单卡训练：要上多卡时在反向前对梯度做 all-reduce，并把日志/存档限制在 rank 0
        for _ in range(cfg.train.grad_accum):
            x, y = dataset.get_batch("train", cfg.data.batch_size, cfg.data.block_size, device_type)
            with ctx:
                logits, loss = model(x, y)
            scaler.scale(loss / cfg.train.grad_accum).backward()
            loss_sum += loss.item() / cfg.train.grad_accum
            acc_sum += (logits.argmax(-1) == y).float().mean().item() / cfg.train.grad_accum
            tokens += y.numel()

        if cfg.optim.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip).item()
        else:
            grad_norm = 0.0
        scaler.step(optimizer)
        scaler.update()

        last = step == cfg.train.max_iters - 1
        if step % cfg.train.log_interval == 0 or last:
            dt = time.perf_counter() - t0
            metrics = {
                "loss": loss_sum,
                "accuracy": acc_sum,
                "lr": lr,
                "grad_norm": grad_norm,
                "tokens_per_sec": tokens / max(dt, 1e-9),
            }
            for logger in loggers:
                logger.log(step, metrics, "train")
            if use_bar:
                steps.set_postfix(loss=f"{loss_sum:.4f}", lr=f"{lr:.2g}")
            t0, tokens = time.perf_counter(), 0

        eval_now = step % cfg.train.eval_interval == 0 or last
        if eval_now:
            val = evaluate(model, dataset, cfg, device_type, ctx)
            for logger in loggers:
                logger.log(step, val, "val")
            best_val = min(best_val, val["loss"])
            if use_bar:  # val 结果贴在进度条右边，不打断进度条
                steps.set_postfix(loss=f"{loss_sum:.4f}", lr=f"{lr:.2g}", val=f"{val['loss']:.4f}", ppl=f"{val['perplexity']:.1f}")
        if eval_now or step % cfg.run.save_interval == 0 or last:
            save_ckpt(ckpt_path, raw_model, optimizer, scaler, step, best_val, cfg, dataset.vocab_size)

    last_step = max(start_step - 1, cfg.train.max_iters - 1)
    if use_bar:
        steps.close()
    print(f"\n训练结束：已到第 {last_step} 步，最好 val loss {best_val:.4f}，存档 {ckpt_path}")
    if cfg.run.sample_after_train:
        text = generate_text(cfg, raw_model, dataset, device)
        (out_dir / "samples.txt").write_text(text + "\n", encoding="utf-8")
        print(f"采样已写入 {out_dir / 'samples.txt'}：\n{text}\n")
    for logger in loggers:
        logger.close()
    return out_dir


if __name__ == "__main__":
    run_entry(lambda: train(Config.load(config_path_from_argv())))
