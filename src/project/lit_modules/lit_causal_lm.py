"""Lightning Module for causal-LM training with TinyTransformer."""
from __future__ import annotations

import logging
import torch
import torch.nn.functional as F
import lightning as L
from torch import Tensor
from typing import Sequence
from lightning.pytorch.utilities.types import OptimizerLRScheduler

log = logging.getLogger(__name__)

from ..models.examples import TinyTransformer


class LitCausalLM(L.LightningModule):
    """Causal language model wrapping TinyTransformer.

    Expects batches with keys: input_ids, target_ids, attn_mask
    (as produced by causal_lm_collate).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        d_mlp: int | None = None,
        max_seq_len: int = 64,
        dropout: float = 0.0,
        activation: str = "gelu",
        tie_embed: bool = True,
        use_ln: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        betas: Sequence[float] = (0.9, 0.999),
        warmup_steps: int = 100,
        log_every_n_epochs_phase1: int = 10,
        log_every_n_epochs_phase2: int = 100,
        log_phase_boundary: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = TinyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_mlp=d_mlp,
            max_seq_len=max_seq_len,
            dropout=dropout,
            activation=activation,
            tie_embed=tie_embed,
            use_ln=use_ln,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas: tuple[float, float] = (betas[0], betas[1])
        self.warmup_steps = warmup_steps
        self._train_correct: int = 0
        self._train_total: int = 0
        self._train_loss_sum: float = 0.0
        self._train_loss_count: int = 0
        self._val_correct: int = 0
        self._val_total: int = 0
        self._val_loss_sum: float = 0.0
        self._val_loss_count: int = 0
        self.log_every_n_epochs_phase1 = log_every_n_epochs_phase1
        self.log_every_n_epochs_phase2 = log_every_n_epochs_phase2
        self.log_phase_boundary = log_phase_boundary

    def _should_log_epoch(self) -> bool:
        """Check whether metrics should be logged to the logger this epoch."""
        epoch = self.current_epoch
        if epoch == 0:
            return True
        if epoch <= self.log_phase_boundary:
            return epoch % self.log_every_n_epochs_phase1 == 0
        return epoch % self.log_every_n_epochs_phase2 == 0

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self.model(input_ids, attention_mask)

    def _compute_metrics(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass and compute loss + accuracy on supervised positions."""
        attn_mask = batch["attn_mask"][:, None, None, :]
        logits = self(batch["input_ids"], attn_mask)  # (B, T, V)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch["target_ids"].view(-1),
            ignore_index=-100,
        )
        mask = batch["target_ids"] != -100
        preds = logits.argmax(dim=-1)
        acc = (preds[mask] == batch["target_ids"][mask]).float().mean()
        return logits, loss, acc

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        logits, loss, _ = self._compute_metrics(batch)
        mask = batch["target_ids"] != -100
        n_supervised = int(mask.sum().item())
        preds = logits.argmax(dim=-1)
        self._train_correct += int((preds[mask] == batch["target_ids"][mask]).sum().item())
        self._train_total += n_supervised
        self._train_loss_sum += loss.item() * n_supervised
        self._train_loss_count += n_supervised
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        logits, loss, _ = self._compute_metrics(batch)
        # Accumulate counts for proper micro-averaged accuracy.
        mask = batch["target_ids"] != -100
        n_supervised = int(mask.sum().item())
        preds = logits.argmax(dim=-1)
        self._val_correct += int((preds[mask] == batch["target_ids"][mask]).sum().item())
        self._val_total += n_supervised
        self._val_loss_sum += loss.item() * n_supervised
        self._val_loss_count += n_supervised
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_correct = 0
        self._val_total = 0
        self._val_loss_sum = 0.0
        self._val_loss_count = 0

    def on_train_epoch_start(self) -> None:
        self._train_correct = 0
        self._train_total = 0
        self._train_loss_sum = 0.0
        self._train_loss_count = 0

    def on_train_epoch_end(self) -> None:
        if self._train_total == 0:
            return
        acc = self._train_correct / self._train_total
        loss = self._train_loss_sum / self._train_loss_count
        if self._should_log_epoch():
            self.log("train_loss", loss, prog_bar=True, on_epoch=True)
            self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        msg = f"Epoch {self.current_epoch} | train_loss={loss:.4f} | train_acc={acc:.4f}"
        self.print(msg)
        log.info(msg)

    def on_validation_epoch_end(self) -> None:
        if self._val_total == 0:
            return
        acc = self._val_correct / self._val_total
        loss = self._val_loss_sum / self._val_loss_count
        if self._should_log_epoch():
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", acc, prog_bar=True)
        msg = f"  val_loss={loss:.4f} | val_acc={acc:.4f}"
        self.print(msg)
        log.info(msg)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
            betas=self.betas,
        )
        # Linear warmup then constant LR — simple and effective for small models.
        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(self.warmup_steps, 1)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
