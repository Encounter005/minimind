# AGENTS.md

MiniMind: a small language model built from scratch in PyTorch — pretrain, full SFT, LoRA, distillation, DPO, PPO, GRPO and agent RL. The repo is a set of standalone scripts sharing three top-level modules, not an installed library.

Authoritative docs: `README.md` (Chinese, current) and `README_en.md` (English mirror). Use them for datasets, weights, recipes and hardware notes instead of restating them here.

## Layout and package boundaries

- `model/model_minimind.py` — architecture + `MiniMindConfig`; `model/model_lora.py` — hand-written LoRA (no PEFT dependency).
- `dataset/lm_dataset.py` — JSONL datasets (`PretrainDataset`, `SFTDataset`, `DPODataset`, `RLAIFDataset`, `AgentRLDataset`).
- `trainer/` — one script per training stage plus `trainer_utils.py` (checkpointing, LR schedule, `init_model`, DDP init, logging).
- `scripts/` — `web_demo.py` (Streamlit), `serve_openai_api.py` (FastAPI, `POST /v1/chat/completions` on port 8998), `eval_toolcall.py`, `chat_api.py`, `convert_model.py` (torch ↔ transformers).
- `eval_llm.py` — repo-root CLI chat/eval entry point.

Imports are absolute from the repo root (`from model.model_minimind import ...`); every entry script prepends the repo root to `sys.path`. Run scripts as files from the directories listed below — `python -m trainer.train_pretrain`, or moving modules into packages, breaks both the imports and the relative paths.

`pyproject.toml`, `uv.lock`, `.python-version` and `src/` are untracked uv scaffolding; the `minimind` console script only prints a greeting. The editable install puts `src/` on `sys.path` (`.venv/.../minimind.pth`), and a regular package under `src/` (such as `src/trainer/`) shadows the repo's same-named top-level directory, because a package with `__init__.py` wins over a namespace directory. That is why `model/`, `dataset/` and `trainer/` each carry an empty `__init__.py` — keep them. Do not put project code under `src/`.

## Working-directory contract

Defaults are relative to each entry point's expected cwd, so the same word means different directories:

| entry point | run from | default paths |
| --- | --- | --- |
| `trainer/train_*.py` | `cd trainer` | `../dataset/*.jsonl`, `../model`, `../out`, `../checkpoints` |
| `scripts/*.py` | `cd scripts` | `../model`, `../out`; `web_demo.py` scans `scripts/` for model dirs |
| `eval_llm.py` | repo root | `./model`, `./out` |

Launching a trainer from the repo root resolves `../dataset` outside the repository; it will fail or read the wrong files. Pass absolute paths via `--data_path`/`--save_dir` if you need another cwd.

## Environment

- Python 3.13 (`.python-version`).
- `torch` is **not** installed by `requirements.txt` (it is commented out there) and is not a direct dependency in `pyproject.toml`; install the PyTorch build matching the host first, as in README step 0 / section Ⅱ. `uv sync` would pull torch 2.14.1 transitively through `sentence-transformers`, but that is not the documented setup.
- `.venv/` exists but is unsynced (site-packages empty). The documented install is `pip install -r requirements.txt`.
- DDP starts only when `RANK` is set (i.e. under `torchrun`); plain `python train_x.py` trains on one device.

## Commands

Training stages (each must be preceded by `cd trainer`):

```bash
torchrun --nproc_per_node N train_pretrain.py      # required first; `python train_pretrain.py` for 1 GPU
torchrun --nproc_per_node N train_full_sft.py      # required second
torchrun --nproc_per_node N train_lora.py
torchrun --nproc_per_node N train_distillation.py
torchrun --nproc_per_node N train_dpo.py
torchrun --nproc_per_node N train_ppo.py
torchrun --nproc_per_node N train_grpo.py
torchrun --nproc_per_node N train_agent.py
```

Inference and tooling:

```bash
python eval_llm.py --load_from ./out --weight full_sft   # native torch weights in ./out (interactive: uses input(); pipe stdin in non-tty runs)
python eval_llm.py --load_from ./minimind-3              # transformers-format model dir
cd scripts && python serve_openai_api.py                 # OpenAI-compatible API
cd scripts && streamlit run web_demo.py                  # needs a transformers model dir copied into scripts/
cd scripts && python convert_model.py                    # edits paths in __main__, see below
```

## Data, weights and generated artifacts

- `dataset/` contains loaders only. Download the `.jsonl` files from the ModelScope dataset link in README into that directory; defaults expect `pretrain_t2t_mini.jsonl` and `sft_t2t_mini.jsonl`.
- `model/` ships the tokenizer only. `.pth` weights live in `out/` (gitignored) or are downloaded from ModelScope/HuggingFace.
- A HuggingFace model directory cloned with plain `git clone` leaves `model.safetensors` as a ~134-byte LFS pointer when `git-lfs` is missing, and `transformers` then fails with `SafetensorError: header too large`. Re-fetch the blob instead of debugging the loader: `curl -L -o model.safetensors https://huggingface.co/<repo>/resolve/main/model.safetensors`, or use `modelscope download`.
- Weight filenames encode the config: `{weight}_{hidden_size}{_moe}.pth` (`full_sft_768.pth`, `grpo_768_moe.pth`). Eval, serving and conversion must receive matching `--hidden_size`, `--use_moe` and `--num_hidden_layers`, otherwise `load_state_dict(..., strict=False)` loads mismatched weights silently.
- `--from_resume 1` resumes from `../checkpoints/{weight}_{hidden_size}{_moe}_resume.pth` and tolerates a different GPU count.
- `out/`, `website/`, `docs-minimind/` are gitignored; `checkpoints/` is **not** — never commit weights, checkpoints or `tokenizer.model` output.
- `scripts/convert_model.py` hardcodes its source/destination paths in the `__main__` block; edit that block for a conversion instead of adding flags.

## Do not change without a strong reason

- `model/tokenizer.json` and `model/tokenizer_config.json`: the canonical 6400-vocab tokenizer. README explicitly advises against retraining it — a different vocab invalidates every published weight and chat template. `trainer/train_tokenizer.py` is a learning example writing to `../model_learn_tokenizer/`.
- `import datasets  # noqa: F401` as the first import in every trainer: a Windows pyarrow/torch DLL-loading order workaround (issue #771). Keep it, keep the `noqa`.
- Chinese comments, log strings and `--help` text throughout. Match the existing language.

## Verification

There is no test suite, linter, formatter, or type-check configuration, and no CI workflow (`.github/` does not exist; `pyproject.toml` has no `[tool.*]`). Do not claim a change is validated by a "test suite".

Smallest checks, in order:

```bash
python eval_llm.py --help                 # from repo root: imports model/, tokenizer, torch
cd trainer && python train_pretrain.py --help   # imports model/, dataset/, trainer_utils
cd scripts && python serve_openai_api.py --help
```

`--help` exercises imports and CLI wiring without data or GPU. For a behavioural change, run the affected stage for a few steps with small `--epochs`/`--batch_size`/`--save_interval` on the mini datasets and confirm the expected `out/*.pth` appears; for serving/UI changes, start the script and send one request to `POST /v1/chat/completions`.

A change is complete when: the touched entry point starts from its documented cwd, the stage it drives runs at least a few steps (or serves a request), and no weight, checkpoint, cache, or `.venv/` content was added to the diff.


## Attention
不要在用户没有明确要求的情况下修改代码，如需修改代码，需要询问用户，且只有一次，后续如果还需要修改，还要继续询问用户，关于用户对当前代码的看法和问题，只需要给出plan。
