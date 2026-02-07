#!/usr/bin/env bash

# Multi-GPU DDP training for modular addition (grokking).
#
# Intended for multi-GPU cloud instances. Uses DDP strategy with bf16
# mixed precision. Each GPU gets the full dataset; Lightning handles
# distributed sampling automatically.
#
# Usage:
#   ./scripts/train_ddp.sh                            # 2 GPUs (default)
#   ./scripts/train_ddp.sh trainer.devices=4           # 4 GPUs
#   ./scripts/train_ddp.sh data.batch_size=1024        # override any Hydra param
#
# Monitor with TensorBoard:
#   uv run tensorboard --logdir runs/

set -euo pipefail

uv run python -m project.train.run \
    data=modular \
    model=causal_lm \
    trainer=ddp \
    trainer.max_epochs=1000 \
    trainer.precision=bf16-mixed \
    "$@"
