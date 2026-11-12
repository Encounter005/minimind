"""把 dataset/ 下的 JSONL 预训练语料打成 nanoGPT 风格的 uint16 .bin。

    python -m my_trainer.data.prepare_minimind \
        --jsonl dataset/pretrain_t2t_mini.jsonl \
        --out dataset/minimind_smoke --max-lines 5000

分词规则与仓库 dataset/lm_dataset.py 的 PretrainDataset 一致：
[tokenizer(text, add_special_tokens=False)] 前后各补一个 bos / eos。
不写 meta.pkl —— 配置里用 [data] tokenizer = "hf:model" 显式指定分词器。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="JSONL 预训练语料 -> train.bin / val.bin")
    parser.add_argument("--jsonl", required=True, help="输入 JSONL，每行含 text 字段")
    parser.add_argument("--out", required=True, help="输出目录，对应 [data] root/name")
    parser.add_argument("--tokenizer", default="model", help="HF 分词器目录或仓库 id")
    parser.add_argument("--max-lines", type=int, default=0, help="只取前 N 行，0 = 全量")
    parser.add_argument("--max-length", type=int, default=256, help="单条截断长度（含 bos/eos）")
    parser.add_argument("--val-ratio", type=float, default=0.01, help="验证集占比")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = len(tokenizer)
    if vocab_size > 65535:
        raise SystemExit(f"词表 {vocab_size} 超过 65535，uint16 的 .bin 存不下")
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    if bos_id is None or eos_id is None:
        raise SystemExit(f"{args.tokenizer} 缺少 bos/eos token，无法按 PretrainDataset 的规则拼接")

    ids: list[int] = []
    n_lines = 0
    with Path(args.jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if args.max_lines and n_lines >= args.max_lines:
                break
            line = line.strip()
            if not line:
                continue
            text = str(json.loads(line).get("text", ""))
            if not text:
                continue
            n_lines += 1
            piece = tokenizer(
                text,
                add_special_tokens=False,
                max_length=args.max_length - 2,
                truncation=True,
            ).input_ids
            ids.append(bos_id)
            ids.extend(piece)
            ids.append(eos_id)

    if not ids:
        raise SystemExit("没有解析到任何文本，检查 --jsonl 路径与 text 字段")
    arr = np.asarray(ids, dtype=np.uint16)

    n_val = max(int(len(arr) * args.val_ratio), args.max_length + 1)
    if n_val >= len(arr):
        raise SystemExit(f"语料太小：{len(arr)} token 不够切出 {n_val} 的验证集")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arr[:-n_val].tofile(out / "train.bin")
    arr[-n_val:].tofile(out / "val.bin")
    print(
        f"行数 {n_lines:,} | 词表 {vocab_size} | token 总数 {len(arr):,} "
        f"| train {len(arr) - n_val:,} | val {n_val:,}"
    )
    print(f"已写入 {out}/train.bin, {out}/val.bin")


if __name__ == "__main__":
    main()
