"""Activation caching utilities wrapping TransformerLens.

TransformerLens provides powerful caching and hooking via HookedRootModule.
This module re-exports key utilities and adds convenience helpers.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, cast

import torch
from torch import Tensor
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookedRootModule, HookPoint

# Re-export TransformerLens types
__all__ = [
    "ActivationCache",
    "HookPoint",
    "HookedRootModule",
    "get_cache",
    "get_activation",
    "list_hook_names",
    "cache_subset",
]


HookFn = Callable[[Tensor, Any], Tensor | None]


def get_cache(
    model: HookedRootModule,
    input_ids: Tensor,
    names_filter: Callable[[str], bool] | Sequence[str] | None = None,
    **forward_kwargs,
) -> tuple[Tensor, ActivationCache]:
    """Run model and cache activations.

    Thin wrapper around model.run_with_cache() for convenience.

    Args:
        model: A HookedRootModule (e.g., TinyTransformer).
        input_ids: Input token IDs.
        names_filter: Which hooks to cache. Can be:
            - None: cache all hooks
            - Callable: function(name) -> bool
            - Sequence: list of hook names to cache
        **forward_kwargs: Additional args passed to forward().

    Returns:
        Tuple of (model_output, activation_cache).

    Example:
        logits, cache = get_cache(model, input_ids)
        attn_pattern = cache["blocks.0.attn.hook_attn_pattern"]
    """
    result = model.run_with_cache(
        input_ids,
        names_filter=names_filter,
        **forward_kwargs,
    )
    logits, cache = result
    return cast(Tensor, logits), cast(ActivationCache, cache)


def get_activation(
    model: HookedRootModule,
    input_ids: Tensor,
    hook_name: str,
    **forward_kwargs,
) -> Tensor:
    """Get a single activation from the model.

    Args:
        model: A HookedRootModule.
        input_ids: Input token IDs.
        hook_name: Name of the hook to extract.
        **forward_kwargs: Additional args passed to forward().

    Returns:
        The activation tensor at the specified hook.

    Example:
        residual = get_activation(model, input_ids, "blocks.0.hook_resid_post")
    """
    _, cache = model.run_with_cache(
        input_ids,
        names_filter=[hook_name],
        **forward_kwargs,
    )
    return cache[hook_name]


def list_hook_names(
    model: HookedRootModule,
    filter_fn: Callable[[str], bool] | None = None,
) -> list[str]:
    """List all hook point names in a model.

    Args:
        model: A HookedRootModule.
        filter_fn: Optional filter function.

    Returns:
        List of hook names.

    Example:
        # Get all attention pattern hooks
        attn_hooks = list_hook_names(model, lambda n: "attn_pattern" in n)
    """
    names = list(model.hook_dict.keys())
    if filter_fn:
        names = [n for n in names if filter_fn(n)]
    return names


def cache_subset(
    cache: ActivationCache,
    names: Sequence[str] | None = None,
    filter_fn: Callable[[str], bool] | None = None,
) -> dict[str, Tensor]:
    """Extract a subset of activations from cache as a plain dict.

    Args:
        cache: TransformerLens ActivationCache.
        names: Specific hook names to extract.
        filter_fn: Filter function for hook names.

    Returns:
        Dict mapping hook names to tensors.

    Example:
        residuals = cache_subset(cache, filter_fn=lambda n: "resid_post" in n)
    """
    if names is not None:
        return {name: cache[name] for name in names}
    if filter_fn is not None:
        return {name: cache[name] for name in cache.keys() if filter_fn(name)}
    return dict(cache)
