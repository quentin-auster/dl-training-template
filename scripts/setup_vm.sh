#!/usr/bin/env bash

# VM setup script for cloud GPU instances (RunPod, Lambda, Vast.ai, GCP, etc.).
#
# Installs:
#   - uv (Python package manager)
#   - rclone (cloud storage sync — optional, for syncing runs to Google Drive / S3 / etc.)
#   - Project Python dependencies (via uv sync)
#
# Usage:
#   git clone <your-repo-url> && cd training-template
#   ./scripts/setup_vm.sh
#
# After setup, configure rclone for cloud sync (optional):
#   rclone config          # interactive OAuth setup
#   export RCLONE_DEST=gdrive:training-runs  # then runs auto-sync after training

set -euo pipefail

echo "=== Installing uv ==="
if command -v uv &>/dev/null; then
    echo "uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "uv installed: $(uv --version)"
fi

echo ""
echo "=== Installing rclone ==="
if command -v rclone &>/dev/null; then
    echo "rclone already installed: $(rclone --version | head -1)"
else
    curl https://rclone.org/install.sh | sudo bash
    echo "rclone installed: $(rclone --version | head -1)"
fi

echo ""
echo "=== Installing Python dependencies ==="
uv sync

echo ""
echo "=== Verifying installation ==="
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
uv run python -c "from project.models import TinyTransformer; print('project imports OK')"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Run a smoke test:     uv run python -m project.train.run trainer=cpu trainer.max_epochs=1"
echo "  2. Train on GPU:         ./scripts/train_gpu.sh"
echo "  3. (Optional) Set up cloud sync:"
echo "       rclone config                           # one-time OAuth setup"
echo "       export RCLONE_DEST=gdrive:training-runs # enable auto-sync"
