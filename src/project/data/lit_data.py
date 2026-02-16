"""Lightning DataModules for synthetic datasets."""
from __future__ import annotations

from functools import partial

import lightning as L
from torch.utils.data import DataLoader

from .tokenize import Vocab, build_shared_vocab, causal_lm_collate
from .modular import FullModularAdditionDataset, ModularAdditionConfig
from .dyck import DyckNextTokenDataset, DyckConfig


class ModularAdditionDataModule(L.LightningDataModule):
    """Lightning DataModule for modular addition (Nanda grokking setup).

    Enumerates all p^2 pairs, shuffles with a fixed seed, and splits
    into train/val by frac_train.
    """

    def __init__(
        self,
        modulus: int = 113,
        frac_train: float = 0.3,
        answer_only_supervision: bool = True,
        include_bos: bool = False,
        include_eos: bool = False,
        use_plus: bool = False,
        batch_size: int = 4096,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.vocab = build_shared_vocab(modulus)
        self.cfg = ModularAdditionConfig(
            modulus=modulus,
            answer_only_supervision=answer_only_supervision,
            include_bos=include_bos,
            include_eos=include_eos,
            use_plus=use_plus,
        )
        self.frac_train = frac_train
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = FullModularAdditionDataset(
            vocab=self.vocab, cfg=self.cfg,
            frac_train=self.frac_train, split="train", seed=self.seed,
        )
        self.val_ds = FullModularAdditionDataset(
            vocab=self.vocab, cfg=self.cfg,
            frac_train=self.frac_train, split="val", seed=self.seed,
        )
        assert set(self.train_ds.pairs).isdisjoint(set(self.val_ds.pairs))


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # persistent_workers=True,
            collate_fn=partial(causal_lm_collate, pad_id=self.vocab.pad_id),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # persistent_workers=True,
            collate_fn=partial(causal_lm_collate, pad_id=self.vocab.pad_id),
        )
