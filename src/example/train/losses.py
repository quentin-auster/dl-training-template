"""Loss function registry for Hydra integration."""

from __future__ import annotations

from typing import Any

from torch import nn

# Registry of available loss functions.
# Add custom losses here as needed.
LOSSES: dict[str, type[nn.Module]] = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "bce": nn.BCELoss,
    "bce_logits": nn.BCEWithLogitsLoss,
    "nll": nn.NLLLoss,
    "kl_div": nn.KLDivLoss,
    "huber": nn.HuberLoss,
    "smooth_l1": nn.SmoothL1Loss,
    "cosine_embedding": nn.CosineEmbeddingLoss,
}


def get_loss(name: str, **kwargs: Any) -> nn.Module:
    """Get a loss function by name.

    Args:
        name: Loss function name (see LOSSES registry).
        **kwargs: Arguments passed to the loss constructor.

    Returns:
        Instantiated loss module.

    Raises:
        KeyError: If loss name is not in registry.
    """
    if name not in LOSSES:
        available = ", ".join(sorted(LOSSES.keys()))
        raise KeyError(f"Unknown loss '{name}'. Available: {available}")
    return LOSSES[name](**kwargs)


def register_loss(name: str, loss_cls: type[nn.Module]) -> None:
    """Register a custom loss function.

    Args:
        name: Name to register the loss under.
        loss_cls: Loss class (not instance).
    """
    LOSSES[name] = loss_cls
