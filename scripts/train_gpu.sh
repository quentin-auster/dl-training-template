#!/usr/bin/env bash

# Single-GPU training for modular addition (grokking).
#
# Intended for cloud instances with one CUDA GPU (e.g. RunPod, Lambda, Vast.ai).
# Uses mixed precision for faster training and lower memory usage.
#
# Usage:
#   ./scripts/train_gpu.sh                          # defaults
#   ./scripts/train_gpu.sh trainer.max_epochs=2000  # override any Hydra param
#
# Monitor with TensorBoard:
#   uv run tensorboard --logdir outputs/

set -euo pipefail

uv run python -m project.train.run \
    data=modular \
    model=causal_lm \
    trainer=gpu_1 \
    trainer.max_epochs=1000 \
    trainer.precision=16-mixed \
    "$@"
