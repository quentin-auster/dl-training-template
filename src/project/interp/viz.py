"""Lightweight visualization helpers for mechanistic interpretability.

Provides simple plotting utilities for attention patterns, activation norms,
ablation results, and patching heatmaps. Uses matplotlib only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from project.interp.ablate import AblationResult
    from project.interp.patch import PatchingResult


def plot_attention_pattern(
    attention: np.ndarray,
    tokens: list[str] | None = None,
    ax: Axes | None = None,
    cmap: str = "Blues",
    title: str | None = None,
) -> Axes:
    """Plot a single attention pattern as a heatmap.

    Args:
        attention: Attention weights of shape (seq_len, seq_len) or (query, key).
        tokens: Optional token labels for axes.
        ax: Matplotlib axes to plot on. Creates new figure if None.
        cmap: Colormap name.
        title: Optional plot title.

    Returns:
        The matplotlib Axes object.

    Example:
        # attention shape: (seq_len, seq_len)
        plot_attention_pattern(attention, tokens=["The", "cat", "sat"])
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(attention, cmap=cmap, aspect="auto")
    ax.figure.colorbar(im, ax=ax, shrink=0.8)

    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)

    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")

    if title:
        ax.set_title(title)

    return ax


def plot_attention_heads(
    attention: np.ndarray,
    tokens: list[str] | None = None,
    n_cols: int = 4,
    figsize: tuple[int, int] | None = None,
    head_labels: list[str] | None = None,
) -> Figure:
    """Plot attention patterns for multiple heads.

    Args:
        attention: Attention weights of shape (n_heads, seq_len, seq_len).
        tokens: Optional token labels.
        n_cols: Number of columns in subplot grid.
        figsize: Figure size. Auto-computed if None.
        head_labels: Optional labels for each head.

    Returns:
        The matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    n_heads = attention.shape[0]
    n_rows = (n_heads + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for i in range(n_heads):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        label = head_labels[i] if head_labels else f"Head {i}"
        plot_attention_pattern(attention[i], tokens=tokens, ax=ax, title=label)

    # Hide unused subplots
    for i in range(n_heads, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    return fig


def plot_activation_norms(
    norms: dict[str, float] | list[float],
    labels: list[str] | None = None,
    ax: Axes | None = None,
    title: str = "Activation Norms",
) -> Axes:
    """Plot activation norms as a bar chart.

    Args:
        norms: Dict mapping layer names to norms, or list of norms.
        labels: Labels for x-axis (required if norms is a list).
        ax: Matplotlib axes. Creates new figure if None.
        title: Plot title.

    Returns:
        The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    if isinstance(norms, dict):
        labels = list(norms.keys())
        values = list(norms.values())
    else:
        values = norms
        if labels is None:
            labels = [str(i) for i in range(len(values))]

    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Norm")
    ax.set_title(title)

    plt.tight_layout()
    return ax


def plot_ablation_results(
    results: list[AblationResult],
    ax: Axes | None = None,
    title: str = "Component Importance (Ablation)",
    top_k: int | None = None,
) -> Axes:
    """Plot ablation results as a bar chart of deltas.

    Args:
        results: List of AblationResult from ablation_sweep.
        ax: Matplotlib axes. Creates new figure if None.
        title: Plot title.
        top_k: If provided, only show top K most important components.

    Returns:
        The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    # Sort by absolute delta
    sorted_results = sorted(results, key=lambda r: abs(r.delta), reverse=True)
    if top_k:
        sorted_results = sorted_results[:top_k]

    labels = [r.component for r in sorted_results]
    deltas = [r.delta for r in sorted_results]

    colors = ["red" if d > 0 else "blue" for d in deltas]
    ax.barh(range(len(deltas)), deltas, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Delta (metric change when ablated)")
    ax.set_title(title)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    return ax


def plot_patching_heatmap(
    results: list[PatchingResult],
    sites: list[str] | None = None,
    positions: list[int] | None = None,
    ax: Axes | None = None,
    cmap: str = "RdBu_r",
    title: str = "Activation Patching",
) -> Axes:
    """Plot patching results as a heatmap of restoration scores.

    Args:
        results: List of PatchingResult from activation_patching.
        sites: List of site names for y-axis. Inferred from results if None.
        positions: List of positions for x-axis. Inferred from results if None.
        ax: Matplotlib axes. Creates new figure if None.
        cmap: Colormap name.
        title: Plot title.

    Returns:
        The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    # Extract unique sites and positions
    if sites is None:
        sites = sorted(set(r.site for r in results))
    if positions is None:
        positions = sorted(set(r.position for r in results if r.position is not None))

    # Build restoration matrix
    matrix = np.zeros((len(sites), len(positions)))
    site_to_idx = {s: i for i, s in enumerate(sites)}
    pos_to_idx = {p: i for i, p in enumerate(positions)}

    for r in results:
        if r.position is not None and r.site in site_to_idx and r.position in pos_to_idx:
            matrix[site_to_idx[r.site], pos_to_idx[r.position]] = r.restoration

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, label="Restoration", shrink=0.8)

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([str(p) for p in positions])
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels(sites)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Site")
    ax.set_title(title)

    return ax


def plot_probe_accuracy_by_layer(
    layer_accuracies: dict[str, float],
    ax: Axes | None = None,
    title: str = "Probe Accuracy by Layer",
) -> Axes:
    """Plot probe accuracies across layers.

    Args:
        layer_accuracies: Dict mapping layer names to accuracy values.
        ax: Matplotlib axes. Creates new figure if None.
        title: Plot title.

    Returns:
        The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    layers = list(layer_accuracies.keys())
    accuracies = list(layer_accuracies.values())

    ax.plot(range(len(layers)), accuracies, marker="o", linewidth=2, markersize=6)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax
