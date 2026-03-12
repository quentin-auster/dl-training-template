"""Optimizer and learning rate schedule utilities."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

if TYPE_CHECKING:
    from torch.nn import Module


def configure_adamw(
    model: Module,
    lr: float = 1e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    """Configure AdamW optimizer with weight decay exclusions.

    Excludes bias, LayerNorm, and embedding parameters from weight decay.

    Args:
        model: The model to optimize.
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        betas: Adam beta parameters.
        eps: Adam epsilon.

    Returns:
        Configured AdamW optimizer.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "bias" in name or "norm" in name.lower() or "embed" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(param_groups, lr=lr, betas=betas, eps=eps)


def cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LRScheduler:
    """Create a cosine learning rate schedule with linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        min_lr_ratio: Minimum LR as a fraction of initial LR.

    Returns:
        Learning rate scheduler.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def get_grad_norm(model: Module) -> float:
    """Compute the total gradient norm across all parameters.

    Args:
        model: The model to compute gradient norm for.

    Returns:
        Total L2 gradient norm.
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm)
