# Training Template

A lightweight PyTorch Lightning + Hydra template for training small transformer models, with a working causal language modeling setup for modular addition (grokking-style experiments).

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (recommended)
- Optional: CUDA GPU (Linux) or Apple Silicon (MPS)

## Setup

Install dependencies:

```bash
uv sync
```

Run commands through the project environment:

```bash
uv run <command>
```

## Train a model

The main training entrypoint is:

```bash
uv run python -m project.train.run <hydra_overrides>
```

For the default causal LM training setup in this repo, use:

- `model=causal_lm`
- `data=modular`
- a trainer config (`trainer=cpu`, `trainer=mps`, `trainer=gpu_1`, or `trainer=ddp`)

### 1) Quick CPU sanity run

```bash
uv run python -m project.train.run \
  model=causal_lm \
  data=modular \
  trainer=cpu \
  trainer.max_epochs=1 \
  data.batch_size=256 \
  trainer.log_every_n_steps=1
```

### 2) Apple Silicon (MPS)

```bash
./scripts/smoke_local.sh
```

Or explicitly:

```bash
uv run python -m project.train.run \
  model=causal_lm \
  data=modular \
  trainer=mps \
  trainer.max_epochs=20 \
  trainer.log_every_n_steps=1
```

### 3) Single CUDA GPU

```bash
./scripts/train_gpu.sh
```

Equivalent command:

```bash
uv run python -m project.train.run \
  model=causal_lm \
  data=modular \
  trainer=gpu_1 \
  trainer.max_epochs=1000 \
  trainer.precision=16-mixed
```

### 4) Multi-GPU DDP

```bash
./scripts/train_ddp.sh
```

Example overriding number of GPUs:

```bash
./scripts/train_ddp.sh trainer.devices=4
```

## VM / cloud setup

For running on a cloud GPU instance (RunPod, Lambda, Vast.ai, GCP, etc.):

```bash
git clone <your-repo-url> && cd training-template
./scripts/setup_vm.sh
```

This installs `uv`, `rclone`, and all Python dependencies. Then train as usual:

```bash
./scripts/train_gpu.sh
```

### Cloud sync with rclone

Training runs can be automatically synced to cloud storage (Google Drive, S3, GCS, etc.) after each run. This is controlled by the `RCLONE_DEST` environment variable.

One-time setup:

```bash
rclone config  # interactive — follow the OAuth flow for your provider
```

Then set `RCLONE_DEST` before training:

```bash
export RCLONE_DEST=gdrive:training-runs
./scripts/train_gpu.sh
```

Each run directory is synced to `$RCLONE_DEST/<run_dir_name>/` after `trainer.fit()` completes. If `RCLONE_DEST` is not set, nothing happens.

### Weights & Biases

To log to W&B instead of TensorBoard, override the logger:

```bash
./scripts/train_gpu.sh logger=wandb logger.project=my-project
```

You can combine both cloud sync and W&B:

```bash
RCLONE_DEST=gdrive:training-runs ./scripts/train_gpu.sh logger=wandb logger.project=my-project
```

## Logs, checkpoints, and outputs

This project uses Hydra run directories under `outputs/`.

For each run, Hydra creates a directory like:

```text
outputs/2026-02-12/21-12-01/
```

Inside that run directory you will find:

- `resolved_config.yaml`
- `.hydra/` (Hydra config snapshots)
- `checkpoints/` (Lightning checkpoints, via `ModelCheckpoint`)

TensorBoard logs are written to `outputs/<date>/<time>/tb_logs/` alongside checkpoints and configs.

Launch TensorBoard:

```bash
uv run tensorboard --logdir outputs/
```

## Common Hydra overrides

Override any config value from the CLI, for example:

```bash
uv run python -m project.train.run \
  model=causal_lm \
  data=modular \
  trainer=gpu_1 \
  trainer.max_epochs=2000 \
  data.batch_size=2048 \
  run.name=grokking_exp1
```

Useful knobs:

- `trainer.max_epochs=<int>`
- `trainer.devices=<int>` (for DDP)
- `trainer.precision=16-mixed|bf16-mixed|32`
- `data.batch_size=<int>`
- `run.name=<str>`
- `seed=<int>`

## Project layout

- `src/project/train/run.py`: Hydra + Lightning training entrypoint
- `src/project/lit_causal_lm.py`: LightningModule for causal LM
- `src/project/models/examples.py`: TinyTransformer with HookPoints
- `src/project/data/lit_data.py`: modular addition DataModule
- `src/project/interp/`: mechanistic interpretability tools (ablation, patching, probes, viz)
- `src/project/utils/helpers.py`: QoL helpers (`load_checkpoint`, `auto_device`, `to_numpy`, etc.)
- `configs/`: Hydra configs (trainer/model/data/logger/callbacks)
- `scripts/`: launch and setup scripts (`smoke_local.sh`, `train_gpu.sh`, `train_ddp.sh`, `setup_vm.sh`)
