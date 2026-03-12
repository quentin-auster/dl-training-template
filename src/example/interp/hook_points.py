"""Minimal HookPoint / HookedRootModule — drop-in replacement for transformer_lens.hook_points."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

from torch import nn


class HookPoint(nn.Module):
    """Passthrough module that applies registered forward hooks."""

    def __init__(self):
        super().__init__()
        self._fwd_hooks: list[Callable] = []

    def add_hook(self, fn: Callable) -> None:
        self._fwd_hooks.append(fn)

    def clear_hooks(self) -> None:
        self._fwd_hooks = []

    def forward(self, x):
        for fn in self._fwd_hooks:
            result = fn(x, self)
            if result is not None:
                x = result
        return x


class HookedRootModule(nn.Module):
    """nn.Module with run_with_cache / run_with_hooks support.

    Usage:
        class MyModel(HookedRootModule):
            def __init__(self):
                super().__init__()
                self.hook_x = HookPoint()
                self.setup()  # must call at end of __init__

            def forward(self, x):
                return self.hook_x(x)

        logits, cache = model.run_with_cache(inputs)
        logits = model.run_with_hooks(inputs, fwd_hooks=[("hook_x", fn)])
    """

    def setup(self):
        """Index all HookPoints by their module name. Call at end of __init__."""
        self.hook_dict: dict[str, HookPoint] = {
            name: mod
            for name, mod in self.named_modules()
            if isinstance(mod, HookPoint)
        }

    @contextmanager
    def _apply_hooks(self, fwd_hooks: list[tuple[str, Callable]]):
        for name, fn in fwd_hooks:
            self.hook_dict[name].add_hook(fn)
        try:
            yield
        finally:
            for name, _ in fwd_hooks:
                self.hook_dict[name].clear_hooks()

    def run_with_hooks(self, *args, fwd_hooks=(), **kwargs):
        with self._apply_hooks(list(fwd_hooks)):
            return self(*args, **kwargs)

    def run_with_cache(self, *args, names_filter=None, **kwargs):
        cache: dict = {}
        hook_names = list(self.hook_dict.keys())
        if names_filter is not None:
            hook_names = [n for n in hook_names if names_filter(n)]

        def make_hook(name):
            def hook(x, _hook):
                cache[name] = x
            return hook

        hooks = [(name, make_hook(name)) for name in hook_names]
        logits = self.run_with_hooks(*args, fwd_hooks=hooks, **kwargs)
        return logits, cache
