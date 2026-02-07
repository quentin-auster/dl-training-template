"""Ablation utilities using TransformerLens hooks.

Ablation studies help identify which components (attention heads, neurons, layers)
are important for model behavior by zeroing them out and measuring performance change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from transformer_lens.hook_points import HookedRootModule


@dataclass
class AblationResult:
    """Result of an ablation experiment.

    Attributes:
        component: Name of the ablated component.
        baseline_metric: Metric value without ablation.
        ablated_metric: Metric value with ablation.
        delta: Change in metric (ablated - baseline).
        relative_delta: Relative change (delta / baseline).`
    """

    component: str
    baseline_metric: float
    ablated_metric: float

    @property
    def delta(self) -> float:
        return self.ablated_metric - self.baseline_metric

    @property
    def relative_delta(self) -> float:
        if self.baseline_metric == 0:
            return float("inf") if self.delta != 0 else 0.0
        return self.delta / abs(self.baseline_metric)


def zero_ablation_hook(
    head_idx: int | None = None,
    neuron_idx: int | None = None,
    pos_idx: int | None = None,
) -> Callable[[Tensor, Any], Tensor]:
    """Create a hook that zeros out specific activations.

    Args:
        head_idx: If provided, zero only this attention head.
        neuron_idx: If provided, zero only this neuron/feature.
        pos_idx: If provided, zero only this sequence position.

    Returns:
        Hook function compatible with run_with_hooks.

    Example:
        # Zero head 3 in attention output
        hook_fn = zero_ablation_hook(head_idx=3)
        logits = model.run_with_hooks(
            input_ids,
            fwd_hooks=[("blocks.0.attn.hook_z", hook_fn)],
        )
    """

    def hook(tensor: Tensor, hook: Any) -> Tensor:
        modified = tensor.clone()

        if head_idx is not None and pos_idx is not None:
            # Zero specific head at specific position
            # Assumes shape [batch, seq, n_heads, d_head]
            modified[:, pos_idx, head_idx, :] = 0.0
        elif head_idx is not None:
            # Zero entire head across all positions
            modified[:, :, head_idx, :] = 0.0
        elif neuron_idx is not None and pos_idx is not None:
            modified[:, pos_idx, neuron_idx] = 0.0
        elif neuron_idx is not None:
            modified[:, :, neuron_idx] = 0.0
        elif pos_idx is not None:
            modified[:, pos_idx, :] = 0.0
        else:
            modified.zero_()

        return modified

    return hook


def mean_ablation_hook(
    mean_activation: Tensor,
    head_idx: int | None = None,
    pos_idx: int | None = None,
) -> Callable[[Tensor, Any], Tensor]:
    """Create a hook that replaces activations with their mean.

    Mean ablation often provides more signal than zero ablation since
    it preserves the activation scale.

    Args:
        mean_activation: Pre-computed mean activation tensor.
        head_idx: If provided, only replace this attention head.
        pos_idx: If provided, only replace this position.

    Returns:
        Hook function compatible with run_with_hooks.
    """

    def hook(tensor: Tensor, hook: Any) -> Tensor:
        modified = tensor.clone()

        if head_idx is not None:
            modified[:, :, head_idx, :] = mean_activation[:, :, head_idx, :]
        elif pos_idx is not None:
            modified[:, pos_idx, :] = mean_activation[:, pos_idx, :]
        else:
            modified = mean_activation.expand_as(tensor).clone()

        return modified

    return hook


def run_with_ablation(
    model: HookedRootModule,
    input_ids: Tensor,
    hook_name: str,
    ablation_type: Literal["zero", "mean"] = "zero",
    head_idx: int | None = None,
    neuron_idx: int | None = None,
    pos_idx: int | None = None,
    mean_activation: Tensor | None = None,
    **forward_kwargs,
) -> Tensor:
    """Run model with a component ablated.

    Args:
        model: A HookedRootModule.
        input_ids: Input token IDs.
        hook_name: Hook point to ablate.
        ablation_type: "zero" or "mean".
        head_idx: Specific head to ablate.
        neuron_idx: Specific neuron to ablate.
        pos_idx: Specific position to ablate.
        mean_activation: Required if ablation_type="mean".
        **forward_kwargs: Additional args passed to forward.

    Returns:
        Model output with ablation applied.
    """
    if ablation_type == "zero":
        hook_fn = zero_ablation_hook(head_idx, neuron_idx, pos_idx)
    else:
        if mean_activation is None:
            raise ValueError("mean_activation required for mean ablation")
        hook_fn = mean_ablation_hook(mean_activation, head_idx, pos_idx)

    return model.run_with_hooks(
        input_ids,
        fwd_hooks=[(hook_name, hook_fn)],
        **forward_kwargs,
    )


def ablation_sweep(
    model: HookedRootModule,
    input_ids: Tensor,
    metric_fn: Callable[[Tensor], float],
    components: list[tuple[str, int | None]],
    ablation_type: Literal["zero", "mean"] = "zero",
    mean_cache: dict[str, Tensor] | None = None,
    **forward_kwargs,
) -> list[AblationResult]:
    """Run ablation experiments across multiple components.

    Args:
        model: A HookedRootModule.
        input_ids: Input token IDs.
        metric_fn: Function mapping model output to scalar metric.
        components: List of (hook_name, head_idx) tuples. head_idx can be None.
        ablation_type: "zero" or "mean".
        mean_cache: Dict mapping hook names to mean activations (for mean ablation).
        **forward_kwargs: Additional args passed to forward.

    Returns:
        List of AblationResult objects sorted by delta (most important first).

    Example:
        components = [
            ("blocks.0.attn.hook_z", 0),  # head 0 in layer 0
            ("blocks.0.attn.hook_z", 1),  # head 1 in layer 0
            ("blocks.1.attn.hook_z", 0),  # head 0 in layer 1
        ]
        results = ablation_sweep(model, input_ids, loss_fn, components)
    """
    model.eval()

    # Compute baseline
    with torch.no_grad():
        baseline_output = model(input_ids, **forward_kwargs)
        baseline_metric = metric_fn(baseline_output)

    results = []

    for hook_name, head_idx in components:
        mean_act = mean_cache.get(hook_name) if mean_cache else None

        with torch.no_grad():
            ablated_output = run_with_ablation(
                model,
                input_ids,
                hook_name,
                ablation_type=ablation_type,
                head_idx=head_idx,
                mean_activation=mean_act,
                **forward_kwargs,
            )
            ablated_metric = metric_fn(ablated_output)

        component_name = f"{hook_name}:head_{head_idx}" if head_idx is not None else hook_name
        results.append(
            AblationResult(
                component=component_name,
                baseline_metric=baseline_metric,
                ablated_metric=ablated_metric,
            )
        )

    # Sort by absolute delta (most impactful first)
    results.sort(key=lambda r: abs(r.delta), reverse=True)
    return results


def head_ablation_sweep(
    model: HookedRootModule,
    input_ids: Tensor,
    metric_fn: Callable[[Tensor], float],
    hook_name_template: str = "blocks.{layer}.attn.hook_z",
    ablation_type: Literal["zero", "mean"] = "zero",
    mean_cache: dict[str, Tensor] | None = None,
    **forward_kwargs,
) -> list[AblationResult]:
    """Convenience function to ablate all attention heads.

    Args:
        model: A HookedRootModule with n_layers and n_heads attributes.
        input_ids: Input token IDs.
        metric_fn: Metric function.
        hook_name_template: Template for hook names with {layer} placeholder.
        ablation_type: "zero" or "mean".
        mean_cache: Mean activations for mean ablation.
        **forward_kwargs: Additional args.

    Returns:
        List of AblationResult for all heads.
    """
    components = []

    def _to_int(val):
        if isinstance(val, torch.Tensor):
            return int(val.item())
        try:
            return int(val)
        except Exception:
            if hasattr(val, "__len__"):
                return len(val)
            raise TypeError(f"Cannot convert value {val!r} to int for range()")

    n_layers = _to_int(model.n_layers)
    n_heads = _to_int(model.n_heads)

    for layer in range(n_layers):
        hook_name = hook_name_template.format(layer=layer)
        for head in range(n_heads):
            components.append((hook_name, head))

    return ablation_sweep(
        model,
        input_ids,
        metric_fn,
        components,
        ablation_type=ablation_type,
        mean_cache=mean_cache,
        **forward_kwargs,
    )


def compute_component_importance(
    results: list[AblationResult],
    normalize: bool = True,
) -> dict[str, float]:
    """Convert ablation results to importance scores.

    Args:
        results: List of ablation results.
        normalize: If True, normalize scores to sum to 1.

    Returns:
        Dict mapping component names to importance scores.
    """
    scores = {r.component: abs(r.delta) for r in results}

    if normalize:
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

    return scores
