#!/usr/bin/env bash

# - -e: Exit immediately if any command fails (non-zero exit code)
# - -u: Treat unset variables as an error instead of silently expanding to empty string
# - -o pipefail: If any command in a pipeline fails, the whole pipeline's exit code is the failing
# command's code (by default bash only reports the last command's exit code)

# Without it, the script would silently continue past errors. With it, you get fast, loud failures.
# Good default for any script that shouldn't keep running after something goes wrong.
set -euo pipefail

# Quick smoke test: Nanda grokking setup, 20 epochs on MPS
uv run python -m project.train.run \
    data=modular \
    model=causal_lm \
    trainer=mps \
    trainer.max_epochs=20
