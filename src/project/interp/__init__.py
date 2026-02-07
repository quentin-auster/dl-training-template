"""Mechanistic interpretability utilities built on TransformerLens.

This module provides tools for understanding model internals:

- hooks: Activation caching via TransformerLens
- ablate: Zero/mean ablation to identify important components
- patch: Activation patching for causal tracing
- probes: Linear probes to test feature encoding
- viz: Lightweight visualization helpers

Example workflow:
    from project.models import TinyTransformer
    from project.interp import get_cache, head_ablation_sweep, train_probe

    model = TinyTransformer(vocab_size=100, d_model=64, n_layers=2, n_heads=4)

    # Cache activations
    logits, cache = model.run_with_cache(input_ids)

    # Or use helpers
    logits, cache = get_cache(model, input_ids)

    # Ablation study
    results = head_ablation_sweep(model, input_ids, loss_fn)

    # Probe representations
    result = train_probe(cache["blocks.0.hook_resid_post"][:, -1], labels)
"""

from project.interp.hooks import (
    ActivationCache,
    HookPoint,
    HookedRootModule,
    cache_subset,
    get_activation,
    get_cache,
    list_hook_names,
)
from project.interp.ablate import (
    AblationResult,
    ablation_sweep,
    compute_component_importance,
    head_ablation_sweep,
    mean_ablation_hook,
    run_with_ablation,
    zero_ablation_hook,
)
from project.interp.patch import (
    PatchingResult,
    activation_patching,
    create_corrupted_input,
    path_patching,
    patching_hook,
    run_with_patch,
)
from project.interp.probes import (
    LinearProbe,
    ProbeResult,
    probe_all_layers,
    train_probe,
)
from project.interp.viz import (
    plot_ablation_results,
    plot_activation_norms,
    plot_attention_heads,
    plot_attention_pattern,
    plot_patching_heatmap,
    plot_probe_accuracy_by_layer,
)

__all__ = [
    # hooks (TransformerLens re-exports)
    "ActivationCache",
    "HookPoint",
    "HookedRootModule",
    "get_cache",
    "get_activation",
    "list_hook_names",
    "cache_subset",
    # ablate
    "AblationResult",
    "ablation_sweep",
    "head_ablation_sweep",
    "run_with_ablation",
    "compute_component_importance",
    "mean_ablation_hook",
    "zero_ablation_hook",
    # patch
    "PatchingResult",
    "activation_patching",
    "path_patching",
    "create_corrupted_input",
    "patching_hook",
    "run_with_patch",
    # probes
    "LinearProbe",
    "ProbeResult",
    "probe_all_layers",
    "train_probe",
    # viz
    "plot_ablation_results",
    "plot_activation_norms",
    "plot_attention_heads",
    "plot_attention_pattern",
    "plot_patching_heatmap",
    "plot_probe_accuracy_by_layer",
]
