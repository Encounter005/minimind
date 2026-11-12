"""超参搜索入口：python -m my_trainer.sweep [config.toml]

从 [sweep].space 展开组合，为每组生成 runs/<sweep>/<run_id>/config.toml，
再用子进程 python -m my_trainer.train 跑一次（隔离显存与全局状态），最后汇总成表。
"""

from __future__ import annotations

import copy
import csv
import itertools
import json
import random
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from .configs import Config
from .utils import ConfigError, config_path_from_argv, run_entry


def expand(space: dict[str, list], mode: str, num_samples: int, seed: int) -> list[dict[str, Any]]:
    keys = list(space)
    if mode == "grid":
        return [dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))]
    rng = random.Random(seed)
    seen: set[str] = set()
    groups: list[dict[str, Any]] = []
    for _ in range(num_samples * 20):  # 组合空间可能比 num_samples 小，用尝试次数兜底
        group = {k: rng.choice(space[k]) for k in keys}
        sig = json.dumps(group, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        groups.append(group)
        if len(groups) == num_samples:
            break
    return groups


def run_id(group: dict[str, Any]) -> str:
    """目录名即超参，例如 lr=0.0003__n_layer=4。"""
    parts = [f"{key.split('.')[-1]}={value}" for key, value in sorted(group.items())]
    return "__".join(parts).replace("/", "-").replace(" ", "")


def patch(raw: dict[str, Any], group: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(raw)
    for dotted, value in group.items():
        section, field = dotted.split(".")
        out.setdefault(section, {})[field] = value
    return out


def last_val(metrics_path: Path) -> tuple[float, float]:
    loss = acc = float("nan")
    if not metrics_path.exists():
        return loss, acc
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("split") == "val":
            loss, acc = row["loss"], row.get("accuracy", float("nan"))
    return loss, acc


def write_summary(sweep_root: Path, rows: list[dict[str, Any]], space_keys: list[str]) -> None:
    def rank(row: dict[str, Any]) -> tuple[bool, float]:
        loss = row["val_loss"]
        return row["status"] != "ok", float("inf") if loss != loss else loss

    rows = sorted(rows, key=rank)
    fields = ["run_id", *space_keys, "val_loss", "val_accuracy", "status"]
    with (sweep_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k) for k in fields} for row in rows])
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        cells = [f"{row.get(k):.6g}" if isinstance(row.get(k), float) else str(row.get(k)) for k in fields]
        lines.append("| " + " | ".join(cells) + " |")
    (sweep_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    path = config_path_from_argv()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    base = Config.from_dict(raw)  # 先整份校验，含 space 里的键名
    if not base.sweep.space:
        raise ConfigError('[sweep] space 为空：例如 space = { "optim.lr" = [1e-3, 3e-4] }')

    groups = expand(base.sweep.space, base.sweep.mode, base.sweep.num_samples, base.sweep.seed)
    sweep_root = Path(base.run.out_root) / base.sweep.name
    sweep_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for group in groups:
        rid = run_id(group)
        run_dir = sweep_root / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        patched = patch(raw, group)
        patched.setdefault("run", {})["out_root"] = str(sweep_root)
        patched["run"]["name"] = rid
        patched["run"]["resume"] = "none"  # 每组独立目录，从头跑
        cfg_path = run_dir / "config.toml"
        cfg_path.write_text(Config.from_dict(patched).to_toml(), encoding="utf-8")
        print(f"\n=== {rid} ===")
        proc = subprocess.run([sys.executable, "-m", "my_trainer.train", str(cfg_path)], capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr.strip()[-1500:])
            rows.append(
                {**group, "run_id": rid, "val_loss": float("inf"), "val_accuracy": float("nan"), "status": "error"}
            )
            continue
        loss, acc = last_val(run_dir / "metrics.jsonl")
        rows.append({**group, "run_id": rid, "val_loss": loss, "val_accuracy": acc, "status": "ok"})

    write_summary(sweep_root, rows, list(base.sweep.space))
    print(f"\n汇总：{sweep_root / 'summary.csv'} 与 {sweep_root / 'summary.md'}")


if __name__ == "__main__":
    run_entry(main)
