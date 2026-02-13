"""Quality-of-life helpers for notebooks and scripts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from project.lit_causal_lm import LitCausalLM
    from project.models.examples import TinyTransformer


def auto_device() -> str:
    """Return the best available device: ``'cuda'``, ``'mps'``, or ``'cpu'``."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint(
    path: str | Path,
    device: str | None = None,
) -> tuple[LitCausalLM, TinyTransformer]:
    """Load a LitCausalLM checkpoint and return ``(lit_model, model)``.

    Args:
        path: Path to a ``.ckpt`` file.
        device: Device to map to (default: :func:`auto_device`).

    Returns:
        A tuple of the Lightning module (in eval mode) and the underlying
        ``TinyTransformer``.
    """
    from project.lit_causal_lm import LitCausalLM

    if device is None:
        device = auto_device()
    lit_model = LitCausalLM.load_from_checkpoint(
        str(path), map_location=device, weights_only=False,
    )
    lit_model.eval()
    return lit_model, lit_model.model


def find_latest_checkpoint(base_dir: str | Path = "outputs") -> Path:
    """Find the most recent ``last.ckpt`` under *base_dir*.

    Searches recursively and returns the one with the newest mtime.

    Raises:
        FileNotFoundError: If no checkpoint is found.
    """
    base = Path(base_dir)
    candidates = sorted(base.rglob("last.ckpt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No last.ckpt found under {base}")
    return candidates[-1]


def to_numpy(tensor: Tensor) -> np.ndarray:
    """Detach, move to CPU, cast to float32, and convert to numpy."""
    return tensor.detach().cpu().float().numpy()


def make_batch(
    dataset: Dataset,
    n: int,
    collate_fn: Callable,
    device: str | None = None,
) -> dict[str, Tensor]:
    """Collate the first *n* items from *dataset* and move to *device*.

    Args:
        dataset: A map-style dataset.
        n: Number of items to include.
        collate_fn: Collation function (e.g. ``causal_lm_collate``).
        device: Target device (default: :func:`auto_device`).

    Returns:
        A dict of tensors on the requested device.
    """
    if device is None:
        device = auto_device()
    items = [dataset[i] for i in range(n)]
    batch = collate_fn(items)
    return {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
