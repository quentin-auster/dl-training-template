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

from .models.simple import TinyTransformer


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

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self.model(input_ids, attention_mask)

    def _shared_step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        # Reshape (B, T) -> (B, 1, 1, T) so it broadcasts over (B, n_heads, T, T) scores.
        attn_mask = batch["attn_mask"][:, None, None, :]
        logits = self(batch["input_ids"], attn_mask)  # (B, T, V)
        # Flatten for cross-entropy: ignore_index=-100 handles masked positions.
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch["target_ids"].view(-1),
            ignore_index=-100,
        )
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)

        # Compute accuracy over supervised positions only.
        mask = batch["target_ids"] != -100
        if mask.any():
            preds = logits.argmax(dim=-1)
            acc = (preds[mask] == batch["target_ids"][mask]).float().mean()
            self.log(f"{stage}_acc", acc, prog_bar=True, on_step=(stage == "train"), on_epoch=True)

        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def on_train_epoch_end(self) -> None:
        metrics = self.trainer.callback_metrics
        loss = metrics.get("train_loss_epoch", metrics.get("train_loss"))
        acc = metrics.get("train_acc_epoch", metrics.get("train_acc"))
        parts = [f"Epoch {self.current_epoch}"]
        if loss is not None:
            parts.append(f"train_loss={loss:.4f}")
        if acc is not None:
            parts.append(f"train_acc={acc:.4f}")
        msg = " | ".join(parts)
        self.print(msg)
        log.info(msg)

    def on_validation_epoch_end(self) -> None:
        metrics = self.trainer.callback_metrics
        loss = metrics.get("val_loss")
        acc = metrics.get("val_acc")
        if loss is None:
            return
        parts = [f"  val_loss={loss:.4f}"]
        if acc is not None:
            parts.append(f"val_acc={acc:.4f}")
        msg = " | ".join(parts)
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
