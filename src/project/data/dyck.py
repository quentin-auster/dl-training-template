"""Dyck-1 (balanced parentheses) next-token prediction dataset.

Generates valid Dyck-1 strings by maintaining a stack with depth constraints,
then wraps them as causal-LM sequences for next-token prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import random

from torch.utils.data import Dataset

from .tokenize import Vocab


@dataclass
class DyckConfig:
    max_depth: int = 8
    min_len: int = 8
    max_len: int = 64
    include_bos: bool = True
    include_eos: bool = True
    # If True: standard causal LM next-token supervision everywhere.
    next_token_supervision: bool = True


def _gen_valid_dyck_tokens(rng: random.Random, cfg: DyckConfig) -> List[str]:
    """Generate a valid Dyck-1 sequence with depth bounded by cfg.max_depth."""
    # Ensure even bounds.
    min_len = cfg.min_len + (cfg.min_len % 2)
    max_len = cfg.max_len - (cfg.max_len % 2)
    if max_len < 2:
        raise ValueError("max_len too small")
    if min_len > max_len:
        raise ValueError("min_len > max_len after even adjustment")

    L = rng.randrange(min_len, max_len + 1, 2)

    toks: List[str] = []
    depth = 0
    for t in range(L):
        remaining = L - t
        must_close = depth > (remaining - 1)

        can_open = (depth < cfg.max_depth) and not must_close
        can_close = depth > 0

        if can_open and can_close:
            # Bias toward closing as depth approaches half the remaining slots,
            # to avoid long forced-close tails.
            p_open = 0.2 if depth >= remaining // 2 else 0.55
            if rng.random() < p_open:
                toks.append("(")
                depth += 1
            else:
                toks.append(")")
                depth -= 1
        elif can_open:
            toks.append("(")
            depth += 1
        else:
            toks.append(")")
            depth -= 1

    # Defensive: close any remaining (should already be 0).
    while depth > 0:
        toks.append(")")
        depth -= 1

    return toks


class DyckNextTokenDataset(Dataset):
    """Valid Dyck-1 parentheses sequences in causal LM form.

    Returns (x_ids, y_ids) where y is the standard shifted next-token target.
    """

    def __init__(self, vocab: Vocab, size: int, cfg: DyckConfig, seed: int = 0) -> None:
        self.vocab = vocab
        self.size = int(size)
        self.cfg = cfg
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        rng = random.Random(self.seed + idx)

        toks: List[str] = []
        if self.cfg.include_bos:
            toks.append("<BOS>")

        toks.extend(_gen_valid_dyck_tokens(rng, self.cfg))

        if self.cfg.include_eos:
            toks.append("<EOS>")

        x = self.vocab.encode(toks)
        y = x[1:] + [-100]

        return x, y
