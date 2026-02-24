from __future__ import annotations

import csv
import logging
from pathlib import Path

from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)


class FourierLoggingCallback(Callback):
    """Log Fourier analysis of digit embeddings every N epochs.

    Saves:
    - ``<log_dir>/fourier/epoch_{N:05d}.npz``  — digit_embeds, dft_power, freq_power, top_freqs
    - ``<log_dir>/fourier_summary.csv``         — epoch, dc_fraction, top_freq_0..4 (append)
    - W&B histogram + scalars (only when a WandbLogger is active)

    Digit token IDs are derived from the standard shared-vocab layout:
    [<PAD>, <BOS>, <EOS>, 0, 1, ..., modulus-1, …] — see data/tokenize.py.

    Pass ``modulus=None`` to disable gracefully for non-modular runs.
    """

    def __init__(
        self,
        modulus: int | None = None,
        every_n_epochs: int = 100,
        n_top: int = 10,
    ) -> None:
        self.modulus = modulus
        self.every_n_epochs = every_n_epochs
        self.n_top = n_top

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.modulus is None:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        import numpy as np

        # Digits start at id=3 (after <PAD>, <BOS>, <EOS>) — see data/tokenize.py
        digit_ids = list(range(3, 3 + self.modulus))
        embeds = (
            pl_module.model.embed_tokens.weight[digit_ids]
            .detach().cpu().float().numpy()
        )  # (modulus, d_model)

        dft = np.fft.fft(embeds, axis=0)
        dft_power = np.abs(dft) ** 2
        freq_power = dft_power.sum(axis=1)  # (modulus,)

        non_dc = freq_power.copy()
        non_dc[0] = 0.0
        top_freqs = np.argsort(non_dc)[-self.n_top:][::-1]
        dc_fraction = float(freq_power[0] / freq_power.sum())

        self._save_npz(trainer, epoch, embeds, dft_power, freq_power, top_freqs)
        self._append_csv(trainer, epoch, dc_fraction, top_freqs)
        self._log_wandb(trainer, freq_power, dc_fraction, top_freqs)

    def _save_npz(self, trainer, epoch, embeds, dft_power, freq_power, top_freqs) -> None:
        import numpy as np
        save_dir = Path(trainer.log_dir) / "fourier"
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_dir / f"epoch_{epoch:05d}.npz",
            digit_embeds=embeds,
            dft_power=dft_power,
            freq_power=freq_power,
            top_freqs=top_freqs,
        )
        log.info("Fourier snapshot saved → %s/epoch_%05d.npz", save_dir, epoch)

    def _append_csv(self, trainer, epoch, dc_fraction, top_freqs) -> None:
        csv_path = Path(trainer.log_dir) / "fourier_summary.csv"
        row: dict = {"epoch": epoch, "dc_fraction": round(dc_fraction, 6)}
        row.update({f"top_freq_{i}": int(f) for i, f in enumerate(top_freqs[:5])})
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _log_wandb(self, trainer, freq_power, dc_fraction, top_freqs) -> None:
        try:
            from lightning.pytorch.loggers import WandbLogger
        except ImportError:
            return
        if not isinstance(trainer.logger, WandbLogger):
            return
        import wandb
        trainer.logger.experiment.log(
            {
                "fourier/freq_power_spectrum": wandb.Histogram(freq_power),
                "fourier/dc_fraction": dc_fraction,
                **{f"fourier/top_freq_{i}": int(f) for i, f in enumerate(top_freqs[:5])},
            },
            step=trainer.global_step,
        )
