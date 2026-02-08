#!/usr/bin/env bash

# - -e: Exit immediately if any command fails (non-zero exit code)
# - -u: Treat unset variables as an error instead of silently expanding to empty string
# - -o pipefail: If any command in a pipeline fails, the whole pipeline's exit code is the failing
# command's code (by default bash only reports the last command's exit code)

# Without it, the script would silently continue past errors. With it, you get fast, loud failures.
# Good default for any script that shouldn't keep running after something goes wrong.
set -euo pipefail

# Nanda grokking setup on MPS.
# Grokking typically happens around epoch 1000-3000.
# Use Ctrl-C to stop early; checkpoints are saved periodically.
uv run python -m project.train.run \
    data=modular \
    model=causal_lm \
    trainer=mps \
    trainer.max_epochs=20 \
    trainer.log_every_n_steps=1 \
    "$@"
