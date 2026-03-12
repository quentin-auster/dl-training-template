"""Metrics utilities.

For comprehensive metrics, we recommend using torchmetrics:
    https://lightning.ai/docs/torchmetrics/stable/

Example usage with torchmetrics:

    from torchmetrics import Accuracy, F1Score, MetricCollection

    metrics = MetricCollection({
        "accuracy": Accuracy(task="multiclass", num_classes=10),
        "f1": F1Score(task="multiclass", num_classes=10),
    })

    # In training loop:
    metrics.update(preds, targets)
    results = metrics.compute()
    metrics.reset()

torchmetrics integrates seamlessly with PyTorch Lightning:

    class MyModel(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.train_acc = Accuracy(task="multiclass", num_classes=10)
            self.val_acc = Accuracy(task="multiclass", num_classes=10)

        def training_step(self, batch, batch_idx):
            ...
            self.train_acc(preds, targets)
            self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
"""

from __future__ import annotations

from typing import Any

# Registry for custom metrics (if not using torchmetrics).
# Maps metric names to factory functions.
METRICS: dict[str, Any] = {}


def register_metric(name: str, metric_factory: Any) -> None:
    """Register a custom metric factory.

    Args:
        name: Name to register the metric under.
        metric_factory: Callable that returns a metric instance.
    """
    METRICS[name] = metric_factory


def get_metric(name: str, **kwargs: Any) -> Any:
    """Get a metric by name from the registry.

    Args:
        name: Metric name.
        **kwargs: Arguments passed to the metric factory.

    Returns:
        Metric instance.

    Raises:
        KeyError: If metric name is not in registry.
    """
    if name not in METRICS:
        available = ", ".join(sorted(METRICS.keys())) or "(none registered)"
        raise KeyError(f"Unknown metric '{name}'. Available: {available}")
    return METRICS[name](**kwargs)
