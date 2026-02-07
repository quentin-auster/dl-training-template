"""Training utilities for PyTorch models."""

from project.train.loop import eval_epoch, train_epoch
from project.train.losses import LOSSES, get_loss, register_loss
from project.train.metrics import METRICS, get_metric, register_metric
from project.train.optim import configure_adamw, cosine_schedule_with_warmup, get_grad_norm

__all__ = [
    # loop
    "train_epoch",
    "eval_epoch",
    # losses
    "LOSSES",
    "get_loss",
    "register_loss",
    # metrics
    "METRICS",
    "get_metric",
    "register_metric",
    # optim
    "configure_adamw",
    "cosine_schedule_with_warmup",
    "get_grad_norm",
]
