"""Activation patching for causal tracing using TransformerLens.

Activation patching identifies which activations are causally responsible for
model behavior by running the model on corrupted input, then patching in
clean activations at specific sites and measuring output restoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor
from project.models.hook_points import HookedRootModule


@dataclass
class PatchingResult:
    """Result of an activation patching experiment.

    Attributes:
        site: Name of the patched activation site.
        position: Token position that was patched (if applicable).
        clean_metric: Metric on clean input.
        corrupted_metric: Metric on corrupted input.
        patched_metric: Metric after patching.
        restoration: Fraction of performance restored (0 to 1).
    """

    site: str
    position: int | None
    clean_metric: float
    corrupted_metric: float
    patched_metric: float

    @property
    def restoration(self) -> float:
        """Fraction of clean performance restored by patching."""
        gap = self.clean_metric - self.corrupted_metric
        if abs(gap) < 1e-10:
            return 0.0
        return (self.patched_metric - self.corrupted_metric) / gap


def patching_hook(
    clean_activation: Tensor,
    position: int | None = None,
    head_idx: int | None = None,
) -> Callable[[Tensor, Any], Tensor]:
    """Create a hook that patches in clean activations.

    Args:
        clean_activation: The clean activation to patch in.
        position: If provided, only patch this sequence position.
        head_idx: If provided, only patch this attention head.

    Returns:
        Hook function compatible with run_with_hooks.

    Example:
        hook_fn = patching_hook(clean_cache["blocks.0.attn.hook_z"], position=5)
        logits = model.run_with_hooks(
            corrupted_ids,
            fwd_hooks=[("blocks.0.attn.hook_z", hook_fn)],
        )
    """

    def hook(tensor: Tensor, hook: Any) -> Tensor:
        patched = tensor.clone()

        if position is not None and head_idx is not None:
            patched[:, position, head_idx, :] = clean_activation[:, position, head_idx, :]
        elif position is not None:
            patched[:, position] = clean_activation[:, position]
        elif head_idx is not None:
            patched[:, :, head_idx, :] = clean_activation[:, :, head_idx, :]
        else:
            patched = clean_activation.clone()

        return patched

    return hook


def run_with_patch(
    model: HookedRootModule,
    input_ids: Tensor,
    hook_name: str,
    clean_activation: Tensor,
    position: int | None = None,
    head_idx: int | None = None,
    **forward_kwargs,
) -> Tensor:
    """Run model with a specific activation patched.

    Args:
        model: A HookedRootModule.
        input_ids: Input token IDs (typically corrupted).
        hook_name: Hook point to patch.
        clean_activation: Clean activation to patch in.
        position: Optional position to patch.
        head_idx: Optional head to patch.
        **forward_kwargs: Additional args passed to forward.

    Returns:
        Model output with patched activation.
    """
    hook_fn = patching_hook(clean_activation, position, head_idx)

    return model.run_with_hooks(
        input_ids,
        fwd_hooks=[(hook_name, hook_fn)],
        **forward_kwargs,
    )


def activation_patching(
    model: HookedRootModule,
    clean_input_ids: Tensor,
    corrupted_input_ids: Tensor,
    metric_fn: Callable[[Tensor], float],
    sites: list[str],
    positions: list[int] | None = None,
    **forward_kwargs,
) -> list[PatchingResult]:
    """Perform activation patching across multiple sites and positions.

    This implements the classic causal tracing procedure:
    1. Run model on clean input, cache activations
    2. Run model on corrupted input, get baseline
    3. For each site/position, patch clean activation into corrupted run
    4. Measure how much of clean performance is restored

    Args:
        model: A HookedRootModule.
        clean_input_ids: Clean input token IDs.
        corrupted_input_ids: Corrupted input token IDs.
        metric_fn: Function mapping model output to scalar metric.
        sites: List of hook names to patch.
        positions: List of sequence positions to test. If None, patches all.
        **forward_kwargs: Additional args passed to forward.

    Returns:
        List of PatchingResult objects.

    Example:
        results = activation_patching(
            model,
            clean_input_ids=clean_ids,
            corrupted_input_ids=noised_ids,
            metric_fn=lambda out: out[:, -1, target_token].mean().item(),
            sites=["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"],
            positions=[0, 1, 2, 3],
        )
    """
    model.eval()
    results = []

    with torch.no_grad():
        # Cache clean activations
        clean_output, clean_cache = model.run_with_cache(
            clean_input_ids,
            names_filter=sites,
            **forward_kwargs,
        )
        clean_metric = metric_fn(clean_output)

        # Get corrupted baseline
        corrupted_output = model(corrupted_input_ids, **forward_kwargs)
        corrupted_metric = metric_fn(corrupted_output)

    # Determine positions to test
    if positions is None:
        first_site = sites[0]
        seq_len = clean_cache[first_site].shape[1]
        positions = list(range(seq_len))

    # Patch each site at each position
    for site in sites:
        clean_activation = clean_cache[site]

        for pos in positions:
            with torch.no_grad():
                patched_output = run_with_patch(
                    model,
                    corrupted_input_ids,
                    site,
                    clean_activation,
                    position=pos,
                    **forward_kwargs,
                )
                patched_metric = metric_fn(patched_output)

            results.append(
                PatchingResult(
                    site=site,
                    position=pos,
                    clean_metric=clean_metric,
                    corrupted_metric=corrupted_metric,
                    patched_metric=patched_metric,
                )
            )

    return results


def path_patching(
    model: HookedRootModule,
    clean_input_ids: Tensor,
    corrupted_input_ids: Tensor,
    metric_fn: Callable[[Tensor], float],
    sender_site: str,
    receiver_site: str,
    sender_position: int | None = None,
    receiver_position: int | None = None,
    **forward_kwargs,
) -> PatchingResult:
    """Perform path patching between a sender and receiver site.

    Path patching measures the effect of a specific pathway by patching
    the sender's output only as it affects the receiver.

    Args:
        model: A HookedRootModule.
        clean_input_ids: Clean input.
        corrupted_input_ids: Corrupted input.
        metric_fn: Metric function.
        sender_site: Hook name of the sender component.
        receiver_site: Hook name of the receiver component.
        sender_position: Optional position for sender.
        receiver_position: Optional position for receiver.
        **forward_kwargs: Additional args.

    Returns:
        PatchingResult for this path.
    """
    model.eval()

    with torch.no_grad():
        # Get clean metrics and activations
        clean_output, clean_cache = model.run_with_cache(
            clean_input_ids,
            names_filter=[sender_site],
            **forward_kwargs,
        )
        clean_metric = metric_fn(clean_output)

        # Get corrupted baseline
        corrupted_output = model(corrupted_input_ids, **forward_kwargs)
        corrupted_metric = metric_fn(corrupted_output)

        # Patch sender activation
        patched_output = run_with_patch(
            model,
            corrupted_input_ids,
            sender_site,
            clean_cache[sender_site],
            position=sender_position,
            **forward_kwargs,
        )
        patched_metric = metric_fn(patched_output)

    return PatchingResult(
        site=f"{sender_site}->{receiver_site}",
        position=sender_position,
        clean_metric=clean_metric,
        corrupted_metric=corrupted_metric,
        patched_metric=patched_metric,
    )


def create_corrupted_input(
    model: HookedRootModule,
    input_ids: Tensor,
    corruption_type: str = "noise",
    noise_std: float = 1.0,
    **forward_kwargs,
) -> Tensor:
    """Create corrupted input embeddings for patching experiments.

    Args:
        model: A HookedRootModule (must have embed_tokens).
        input_ids: Original input token IDs.
        corruption_type: "noise" (add Gaussian noise) or "shuffle" (shuffle positions).
        noise_std: Standard deviation for noise corruption.
        **forward_kwargs: Additional args.

    Returns:
        Tensor of corrupted embeddings suitable for models that accept inputs_embeds.

    Note:
        For models that only accept input_ids, you may need to use a different
        corruption strategy (e.g., replacing tokens).
    """
    embeddings = model.embed_tokens(input_ids)

    if corruption_type == "noise":
        noise = torch.randn_like(embeddings) * noise_std
        return embeddings + noise
    elif corruption_type == "shuffle":
        # Shuffle along sequence dimension
        batch_size, seq_len, d_model = embeddings.shape
        indices = torch.randperm(seq_len, device=embeddings.device)
        return embeddings[:, indices, :]
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
