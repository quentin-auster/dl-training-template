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

## Docker

A `docker/Dockerfile` is provided for reproducible GPU training on any CUDA host without fighting driver or Python version mismatches.

### Build

```bash
docker build -f docker/Dockerfile -t training-template .
```

### Run

```bash
docker run --gpus all training-template
```

This runs `./scripts/train_gpu.sh` by default (single GPU, 1000 epochs, fp16). Override the command to use a different script:

```bash
docker run --gpus all training-template ./scripts/train_ddp.sh
```

Or pass Hydra overrides directly:

```bash
docker run --gpus all training-template \
  uv run python -m project.train.run \
    model=causal_lm data=modular trainer=gpu_1 \
    trainer.max_epochs=2000
```

### Cloud sync inside Docker

Pass `RCLONE_CONF_B64` and `RCLONE_DEST` as environment variables — the entrypoint writes the rclone config automatically before training starts:

```bash
docker run --gpus all \
  -e RCLONE_CONF_B64="$(base64 < ~/.config/rclone/rclone.conf)" \
  -e RCLONE_DEST="gdrive:training-runs" \
  training-template ./scripts/train_gpu.sh run.project=modular-addition
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

One-time setup (on a machine with a browser):

```bash
rclone config  # interactive — follow the OAuth flow for your provider
```

For Google Drive, you can get the OAuth token JSON without a full rclone install by running:

```bash
rclone authorize drive
```

This opens a browser flow and prints a JSON token. When running `rclone config`, choose advanced config and paste the JSON when prompted for the token.

Then base64-encode your finished config for use on the VM:

```bash
base64 < ~/.config/rclone/rclone.conf
```

For headless VMs, add your rclone config to `.env` so `setup_vm.sh` configures it automatically:

```
RCLONE_CONF_B64="<output of: base64 < ~/.config/rclone/rclone.conf>"
RCLONE_DEST="gdrive:training-runs"
```

Then train with `run.project` to organize uploads by project:

```bash
./scripts/train_gpu.sh run.project=modular-addition
```

Each run syncs to `$RCLONE_DEST/<project>/run_artifacts/<run_name>/` both periodically during training (every 50 epochs by default) and after `trainer.fit()` completes. If `RCLONE_DEST` is not set, nothing happens.

### Weights & Biases

1. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize).
2. Add it to a `.env` file in the project root (already in `.gitignore`):

   ```
   WANDB_API_KEY=your_key_here
   ```

3. Run with the `wandb` logger:

   ```bash
   ./scripts/train_gpu.sh logger=wandb logger.project=my-project
   ```

The API key is loaded automatically from `.env` via `python-decouple` whenever `logger=wandb` is set. You can also update the default project name in `configs/logger/wandb.yaml`.

You can combine both cloud sync and W&B:

```bash
RCLONE_DEST=gdrive:training-runs ./scripts/train_gpu.sh logger=wandb logger.project=my-project
```

For information on rclone client id setup, see [here](https://rclone.org/drive/#making-your-own-client-id)

## Resuming from a checkpoint

To resume a training run from a saved checkpoint, pass `run.ckpt_path`:

```bash
uv run python -m project.train.run \
  model=causal_lm data=modular trainer=mps \
  trainer.max_epochs=3000 \
  run.ckpt_path=/path/to/last.ckpt
```

This restores model weights, optimizer state, epoch counter, and LR scheduler. Checkpoints are saved locally in the Hydra output directory and synced to GDrive (if configured). To resume from a GDrive checkpoint, pull it down first with `rclone copy`.

## Adaptive log frequency

`LitCausalLM` supports adaptive logging to reduce noise during long runs. By default, metrics are logged to W&B/TensorBoard every 10 epochs for the first 100 epochs, then every 100 epochs after that. Console output is always printed regardless.

Control via Hydra overrides on the model config:

- `model.log_every_n_epochs_phase1=10` — log interval for early training
- `model.log_every_n_epochs_phase2=100` — log interval after the boundary
- `model.log_phase_boundary=100` — epoch where the transition happens

Since `ModelCheckpoint` monitors `val_loss`, checkpoints are only saved at logged epochs.

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
- `run.project=<str>` (required for cloud sync folder structure)
- `run.name=<str>`
- `run.ckpt_path=<path>` (resume from checkpoint)
- `model.log_every_n_epochs_phase1=<int>` (default 10)
- `model.log_every_n_epochs_phase2=<int>` (default 100)
- `model.log_phase_boundary=<int>` (default 100)
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
