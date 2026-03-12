"""Generic train/eval loop utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> float:
    """Run a single training epoch.

    Args:
        model: The model to train.
        loader: Training data loader.
        optimizer: Optimizer instance.
        loss_fn: Loss function.
        device: Device to run on.

    Returns:
        Average loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run evaluation over a dataset.

    Args:
        model: The model to evaluate.
        loader: Evaluation data loader.
        loss_fn: Loss function.
        device: Device to run on.

    Returns:
        Dictionary with 'loss' and 'accuracy' metrics.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        total_loss += loss.item()
        preds = outputs.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.numel()

    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": correct / max(total, 1),
    }
