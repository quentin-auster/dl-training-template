"""Synthetic modular addition dataset in causal-LM form.

Supports two modes:
  1. Random sampling (ModularAdditionDataset) — arbitrary dataset size.
  2. Full enumeration (FullModularAdditionDataset) — all p^2 pairs with a
     deterministic train/val split, matching the Nanda grokking setup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import random

from torch.utils.data import Dataset

from .tokenize import Vocab


@dataclass
class ModularAdditionConfig:
    modulus: int = 113
    n_terms: int = 2
    include_bos: bool = True
    include_eos: bool = True
    # If True, only supervise the answer position (after "=").
    # If False, supervise next-token everywhere (standard causal LM shift).
    answer_only_supervision: bool = True


def _sample_terms(rng: random.Random, modulus: int, n_terms: int) -> List[int]:
    return [rng.randrange(modulus) for _ in range(n_terms)]


def _build_expression_tokens(terms: List[int]) -> List[str]:
    """[3, 7, 2] -> ["3", "+", "7", "+", "2", "="]"""
    toks: List[str] = []
    for i, a in enumerate(terms):
        toks.append(str(a))
        if i < len(terms) - 1:
            toks.append("+")
    toks.append("=")
    return toks


def _encode_pair(
    a: int, b: int, modulus: int, vocab: Vocab, cfg: ModularAdditionConfig,
) -> Tuple[List[int], List[int]]:
    """Encode a single (a, b) pair into (x_ids, y_ids)."""
    ans = (a + b) % modulus
    toks: List[str] = []
    if cfg.include_bos:
        toks.append("<BOS>")
    toks.extend(_build_expression_tokens([a, b]))
    toks.append(str(ans))
    if cfg.include_eos:
        toks.append("<EOS>")

    x = vocab.encode(toks)
    eq_id = vocab.token_to_id["="]

    if cfg.answer_only_supervision:
        eq_pos = x.index(eq_id)
        y = [-100] * len(x)
        y[eq_pos + 1] = x[eq_pos + 1]
    else:
        y = x[1:] + [-100]

    return x, y


class ModularAdditionDataset(Dataset):
    """Randomly sampled modular addition (arbitrary size, with replacement)."""

    def __init__(
        self,
        vocab: Vocab,
        size: int,
        cfg: ModularAdditionConfig,
        seed: int = 0,
    ) -> None:
        self.vocab = vocab
        self.size = int(size)
        self.cfg = cfg
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        rng = random.Random(self.seed + idx)
        terms = _sample_terms(rng, self.cfg.modulus, self.cfg.n_terms)
        return _encode_pair(terms[0], terms[1], self.cfg.modulus, self.vocab, self.cfg)


class FullModularAdditionDataset(Dataset):
    """All p^2 pairs with a fixed train/val split (Nanda grokking setup).

    Enumerates every (a, b) for a, b in [0, p), shuffles with a fixed seed,
    then takes the first frac_train fraction as train, rest as val.
    """

    def __init__(
        self,
        vocab: Vocab,
        cfg: ModularAdditionConfig,
        frac_train: float = 0.3,
        split: str = "train",
        seed: int = 0,
    ) -> None:
        self.vocab = vocab
        self.cfg = cfg
        p = cfg.modulus

        # Build all p^2 pairs and shuffle deterministically.
        all_pairs = [(a, b) for a in range(p) for b in range(p)]
        rng = random.Random(seed)
        rng.shuffle(all_pairs)

        n_train = int(len(all_pairs) * frac_train)
        if split == "train":
            self.pairs = all_pairs[:n_train]
        else:
            self.pairs = all_pairs[n_train:]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        a, b = self.pairs[idx]
        return _encode_pair(a, b, self.cfg.modulus, self.vocab, self.cfg)
