"""Linear probes for testing whether representations encode specific features.

Probing is a technique to test whether neural network representations contain
information about a feature, by training a simple (usually linear) classifier
on top of frozen activations. Note: probing tests correlation, not causation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ProbeResult:
    """Result of training a probe.

    Attributes:
        train_accuracy: Final training accuracy.
        val_accuracy: Validation accuracy (if val data provided).
        train_loss: Final training loss.
        val_loss: Validation loss (if val data provided).
        probe: The trained probe model.
    """

    train_accuracy: float
    val_accuracy: float | None
    train_loss: float
    val_loss: float | None
    probe: nn.Module


class LinearProbe(nn.Module):
    """Simple linear probe for classification or regression.

    Args:
        input_dim: Dimension of input features.
        output_dim: Number of output classes (classification) or 1 (regression).
        probe_type: "classification" or "regression".
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        probe_type: Literal["classification", "regression"] = "classification",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.probe_type = probe_type

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


def train_probe(
    activations: Tensor,
    labels: Tensor,
    probe_type: Literal["classification", "regression"] = "classification",
    val_activations: Tensor | None = None,
    val_labels: Tensor | None = None,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    device: torch.device | str | None = None,
) -> ProbeResult:
    """Train a linear probe on cached activations.

    Args:
        activations: Training activations of shape (n_samples, hidden_dim).
        labels: Training labels of shape (n_samples,) for classification,
                or (n_samples, output_dim) for regression.
        probe_type: "classification" or "regression".
        val_activations: Optional validation activations.
        val_labels: Optional validation labels.
        lr: Learning rate.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        weight_decay: L2 regularization.
        device: Device to train on.

    Returns:
        ProbeResult with trained probe and metrics.

    Example:
        # Probe whether layer 5 activations encode modular sum
        result = train_probe(
            activations=layer5_acts,  # shape: (1000, 768)
            labels=modular_sums,      # shape: (1000,)
            probe_type="classification",
        )
        print(f"Probe accuracy: {result.val_accuracy:.2%}")
    """
    if device is None:
        device = activations.device

    activations = activations.to(device)
    labels = labels.to(device)

    input_dim = activations.shape[-1]
    if probe_type == "classification":
        output_dim = int(labels.max().item()) + 1
    else:
        output_dim = labels.shape[-1] if labels.dim() > 1 else 1

    probe = LinearProbe(input_dim, output_dim, probe_type).to(device)

    if probe_type == "classification":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    # Create data loader
    dataset = TensorDataset(activations, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    probe.train()
    for epoch in range(epochs):
        for batch_acts, batch_labels in loader:
            optimizer.zero_grad()
            logits = probe(batch_acts)
            if probe_type == "regression" and logits.dim() > 1 and batch_labels.dim() == 1:
                logits = logits.squeeze(-1)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

    # Compute final metrics
    probe.eval()
    with torch.no_grad():
        train_logits = probe(activations)
        if probe_type == "regression" and train_logits.dim() > 1 and labels.dim() == 1:
            train_logits = train_logits.squeeze(-1)
        train_loss = loss_fn(train_logits, labels).item()

        if probe_type == "classification":
            train_preds = train_logits.argmax(dim=-1)
            train_accuracy = (train_preds == labels).float().mean().item()
        else:
            # For regression, report R² as "accuracy"
            ss_res = ((train_logits - labels) ** 2).sum()
            ss_tot = ((labels - labels.mean()) ** 2).sum()
            train_accuracy = (1 - ss_res / ss_tot).item()

        val_accuracy = None
        val_loss = None

        if val_activations is not None and val_labels is not None:
            val_activations = val_activations.to(device)
            val_labels = val_labels.to(device)

            val_logits = probe(val_activations)
            if probe_type == "regression" and val_logits.dim() > 1 and val_labels.dim() == 1:
                val_logits = val_logits.squeeze(-1)
            val_loss = loss_fn(val_logits, val_labels).item()

            if probe_type == "classification":
                val_preds = val_logits.argmax(dim=-1)
                val_accuracy = (val_preds == val_labels).float().mean().item()
            else:
                ss_res = ((val_logits - val_labels) ** 2).sum()
                ss_tot = ((val_labels - val_labels.mean()) ** 2).sum()
                val_accuracy = (1 - ss_res / ss_tot).item()

    return ProbeResult(
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        train_loss=train_loss,
        val_loss=val_loss,
        probe=probe,
    )


def probe_all_layers(
    layer_activations: dict[str, Tensor],
    labels: Tensor,
    probe_type: Literal["classification", "regression"] = "classification",
    val_layer_activations: dict[str, Tensor] | None = None,
    val_labels: Tensor | None = None,
    **train_kwargs,
) -> dict[str, ProbeResult]:
    """Train probes on activations from multiple layers.

    Args:
        layer_activations: Dict mapping layer names to activation tensors.
        labels: Labels for all samples.
        probe_type: Type of probe.
        val_layer_activations: Optional validation activations per layer.
        val_labels: Optional validation labels.
        **train_kwargs: Additional arguments passed to train_probe.

    Returns:
        Dict mapping layer names to ProbeResult.

    Example:
        results = probe_all_layers(
            layer_activations={
                "layer_0": acts_0,
                "layer_1": acts_1,
                "layer_2": acts_2,
            },
            labels=targets,
        )
        for layer, result in results.items():
            print(f"{layer}: {result.val_accuracy:.2%}")
    """
    results = {}

    for layer_name, activations in layer_activations.items():
        val_acts = val_layer_activations.get(layer_name) if val_layer_activations else None

        results[layer_name] = train_probe(
            activations=activations,
            labels=labels,
            probe_type=probe_type,
            val_activations=val_acts,
            val_labels=val_labels,
            **train_kwargs,
        )

    return results
