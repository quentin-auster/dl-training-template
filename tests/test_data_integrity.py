"""Tests that catch label leakage and causal-LM data integrity issues.

These tests verify the core invariant of causal LM datasets:
  - At every supervised position i (where y[i] != -100), y[i] == x[i+1].
    This guarantees the model is doing next-token prediction (seeing token i,
    predicting token i+1) and never has the answer at the position it's trying
    to predict.

Apply these to any new dataset class returning (x_ids, y_ids) pairs.
"""
from __future__ import annotations

import pytest

from project.data.tokenize import build_shared_vocab, Vocab
from project.data.modular import (
    ModularAdditionConfig,
    ModularAdditionDataset,
    FullModularAdditionDataset,
)
from project.data.dyck import DyckConfig, DyckNextTokenDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODULUS = 13  # small prime for fast tests


@pytest.fixture
def vocab() -> Vocab:
    return build_shared_vocab(MODULUS)


@pytest.fixture
def modular_cfg() -> ModularAdditionConfig:
    return ModularAdditionConfig(modulus=MODULUS, answer_only_supervision=True)


@pytest.fixture
def modular_cfg_full_supervision() -> ModularAdditionConfig:
    return ModularAdditionConfig(modulus=MODULUS, answer_only_supervision=False)


# ---------------------------------------------------------------------------
# Generic helpers — reusable for any (x, y) dataset
# ---------------------------------------------------------------------------

def assert_causal_lm_labels(x: list[int], y: list[int], context: str = "") -> None:
    """Assert the core causal-LM next-token prediction invariant.

    For every supervised position i (y[i] != -100):
      1. y[i] == x[i+1]  (label is the *next* token, not the current one)
      2. i+1 < len(x)     (there must be a next token to predict)
    """
    assert len(x) == len(y), f"x/y length mismatch: {len(x)} vs {len(y)} {context}"
    for i, label in enumerate(y):
        if label == -100:
            continue
        assert i + 1 < len(x), (
            f"Supervised position {i} is the last token — no next token exists. {context}"
        )
        assert label == x[i + 1], (
            f"Label leak or shift error at position {i}: "
            f"y[{i}]={label} but x[{i+1}]={x[i + 1]}. "
            f"x={x}, y={y} {context}"
        )


def assert_no_self_prediction(x: list[int], y: list[int], context: str = "") -> None:
    """Catch the degenerate case where y[i] == x[i] at supervised positions.

    If this holds everywhere, the model can achieve perfect accuracy by
    copying the current input token — a clear sign of label leakage.
    """
    supervised = [(i, yi) for i, yi in enumerate(y) if yi != -100]
    if not supervised:
        return
    self_predicting = [i for i, yi in supervised if yi == x[i]]
    # Allow occasional coincidence (e.g., a + a = 2a mod p where 2a == a only
    # if a == 0), but flag if ALL supervised positions are self-predicting.
    assert len(self_predicting) < len(supervised), (
        f"Every supervised position has y[i]==x[i] — likely label leakage. "
        f"Self-predicting positions: {self_predicting} {context}"
    )


# ---------------------------------------------------------------------------
# Modular addition tests
# ---------------------------------------------------------------------------

class TestModularAdditionIntegrity:
    """Verify (x, y) pairs from modular addition datasets."""

    def test_label_is_next_token_answer_only(self, vocab, modular_cfg):
        ds = ModularAdditionDataset(vocab, size=50, cfg=modular_cfg, seed=42)
        for idx in range(len(ds)):
            x, y = ds[idx]
            assert_causal_lm_labels(x, y, context=f"idx={idx}")

    def test_label_is_next_token_full_supervision(self, vocab, modular_cfg_full_supervision):
        ds = ModularAdditionDataset(vocab, size=50, cfg=modular_cfg_full_supervision, seed=42)
        for idx in range(len(ds)):
            x, y = ds[idx]
            assert_causal_lm_labels(x, y, context=f"idx={idx}")

    def test_full_dataset_label_is_next_token(self, vocab, modular_cfg):
        ds = FullModularAdditionDataset(
            vocab, cfg=modular_cfg, frac_train=0.3, split="train", seed=0,
        )
        for idx in range(len(ds)):
            x, y = ds[idx]
            assert_causal_lm_labels(x, y, context=f"idx={idx}")

    def test_no_self_prediction_across_samples(self, vocab, modular_cfg):
        """Ensure not all supervised positions trivially copy the input."""
        ds = ModularAdditionDataset(vocab, size=50, cfg=modular_cfg, seed=42)
        for idx in range(len(ds)):
            x, y = ds[idx]
            assert_no_self_prediction(x, y, context=f"idx={idx}")

    def test_answer_only_has_single_supervised_position(self, vocab, modular_cfg):
        """With answer_only_supervision, exactly one position should be supervised."""
        ds = ModularAdditionDataset(vocab, size=20, cfg=modular_cfg, seed=0)
        for idx in range(len(ds)):
            x, y = ds[idx]
            n_supervised = sum(1 for yi in y if yi != -100)
            assert n_supervised == 1, (
                f"Expected 1 supervised position, got {n_supervised}. y={y}"
            )

    def test_supervised_position_is_at_equals(self, vocab, modular_cfg):
        """The supervised position should be where '=' is, predicting the answer."""
        eq_id = vocab.token_to_id["="]
        ds = ModularAdditionDataset(vocab, size=20, cfg=modular_cfg, seed=0)
        for idx in range(len(ds)):
            x, y = ds[idx]
            supervised_positions = [i for i, yi in enumerate(y) if yi != -100]
            assert len(supervised_positions) == 1
            sup_pos = supervised_positions[0]
            assert x[sup_pos] == eq_id, (
                f"Supervised position {sup_pos} is not '='. "
                f"x[{sup_pos}]={x[sup_pos]}, eq_id={eq_id}"
            )

    def test_answer_is_correct(self, vocab, modular_cfg):
        """Verify the label actually encodes (a + b) % p."""
        ds = FullModularAdditionDataset(
            vocab, cfg=modular_cfg, frac_train=1.0, split="train", seed=0,
        )
        p = modular_cfg.modulus
        for idx in range(len(ds)):
            x, y = ds[idx]
            # Decode to check arithmetic
            tokens = vocab.decode(x)
            # Find operands (tokens between BOS and =)
            eq_idx = tokens.index("=")
            operands = [int(t) for t in tokens[1:eq_idx] if t != "+"]
            expected_ans = sum(operands) % p
            # Find the supervised label
            label_id = [yi for yi in y if yi != -100][0]
            label_token = vocab.id_to_token[label_id]
            assert int(label_token) == expected_ans, (
                f"Wrong answer: {operands} mod {p} = {expected_ans}, "
                f"but label is {label_token}"
            )


# ---------------------------------------------------------------------------
# Dyck dataset tests
# ---------------------------------------------------------------------------

class TestDyckIntegrity:
    """Verify (x, y) pairs from the Dyck next-token dataset."""

    def test_label_is_next_token(self, vocab):
        cfg = DyckConfig(max_depth=4, min_len=4, max_len=16)
        ds = DyckNextTokenDataset(vocab, size=50, cfg=cfg, seed=42)
        for idx in range(len(ds)):
            x, y = ds[idx]
            assert_causal_lm_labels(x, y, context=f"idx={idx}")


# ---------------------------------------------------------------------------
# Parameterized: run the core check across all dataset factories
# ---------------------------------------------------------------------------

def _make_modular_random(vocab: Vocab) -> list[tuple[list[int], list[int]]]:
    cfg = ModularAdditionConfig(modulus=MODULUS, answer_only_supervision=True)
    ds = ModularAdditionDataset(vocab, size=30, cfg=cfg, seed=0)
    return [ds[i] for i in range(len(ds))]


def _make_modular_full(vocab: Vocab) -> list[tuple[list[int], list[int]]]:
    cfg = ModularAdditionConfig(modulus=MODULUS, answer_only_supervision=False)
    ds = ModularAdditionDataset(vocab, size=30, cfg=cfg, seed=0)
    return [ds[i] for i in range(len(ds))]


def _make_dyck(vocab: Vocab) -> list[tuple[list[int], list[int]]]:
    cfg = DyckConfig(max_depth=4, min_len=4, max_len=16)
    ds = DyckNextTokenDataset(vocab, size=30, cfg=cfg, seed=0)
    return [ds[i] for i in range(len(ds))]


@pytest.mark.parametrize("factory", [
    _make_modular_random,
    _make_modular_full,
    _make_dyck,
], ids=["modular_random", "modular_full_sup", "dyck"])
def test_causal_lm_invariant_all_datasets(vocab, factory):
    """Smoke test: the next-token invariant holds across all dataset types."""
    samples = factory(vocab)
    for idx, (x, y) in enumerate(samples):
        assert_causal_lm_labels(x, y, context=f"factory={factory.__name__} idx={idx}")
