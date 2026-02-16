from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional

from decouple import config as decouple_config
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from hydra.utils import instantiate

from wonderwords import RandomWord

log = logging.getLogger(__name__)


def _generate_run_name(cfg: DictConfig) -> str:
    """Generate a run name like 'zesty-causal_lm-adamw-gpu_1'."""
    adjective = RandomWord().word(include_parts_of_speech=["adjectives"])
    choices = HydraConfig.get().runtime.choices
    parts = [
        adjective,
        choices.get("model", "model"),
        choices.get("optim", "optim"),
        choices.get("trainer", "trainer"),
    ]
    return "-".join(parts)


def _get_output_dir() -> str:
    """
    Hydra creates a per-run output directory (e.g. outputs/2026-02-01/14-32-18).
    This is the cleanest "run directory" to use for checkpoints/artifacts.
    """
    hc = HydraConfig.get()
    return hc.runtime.output_dir


def _maybe_set_ckpt_dir(callbacks: list[Callback], ckpt_dir: str) -> None:
    """
    If a ModelCheckpoint callback exists and its dirpath is unset (None),
    set it to the run's checkpoint directory.
    """
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint):
            # Lightning uses dirpath=None to mean "default"; we override to be explicit & reproducible.
            if cb.dirpath is None:
                cb.dirpath = ckpt_dir


def _instantiate_callbacks(cfg: DictConfig) -> list[Callback]:
    cbs: list[Callback] = []
    if "callbacks" not in cfg or cfg.callbacks is None:
        return cbs

    # In our configs/callbacks/default.yaml we used a YAML list, so cfg.callbacks is a list-like.
    for cb_conf in cfg.callbacks:
        cbs.append(instantiate(cb_conf))
    return cbs


def _instantiate_logger(cfg: DictConfig) -> Optional[Logger]:
    if "logger" not in cfg or cfg.logger is None:
        return None
    return instantiate(cfg.logger)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Make config visible in logs/artifacts
    OmegaConf.set_struct(cfg, False)

    # Auto-generate a descriptive run name if not provided.
    if not cfg.get("run", {}).get("name"):
        cfg.run.name = _generate_run_name(cfg)
    log.info("Run name: %s", cfg.run.name)

    # Load W&B API key from .env when using the wandb logger.
    if cfg.get("logger", {}).get("_target_", "").endswith("WandbLogger"):
        wandb_key = str(decouple_config("WANDB_API_KEY", default=""))
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key

    # Reproducibility: seed everything (Lightning handles DDP-safe seeding)
    seed = int(cfg.get("seed", 123))
    L.seed_everything(seed, workers=True)

    run_dir = _get_output_dir()

    # You can customize this in configs/config.yaml (run.ckpt_dir)
    ckpt_subdir = str(cfg.get("run", {}).get("ckpt_dir", "checkpoints"))
    ckpt_dir = os.path.join(run_dir, ckpt_subdir)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Instantiate DataModule + LightningModule from Hydra configs
    datamodule = instantiate(cfg.data)
    lit_module = instantiate(cfg.model)

    # Logger + callbacks
    logger = _instantiate_logger(cfg)
    # Point the logger's output into the Hydra run directory so that
    # checkpoints, configs, and TB/W&B logs all live together.
    if hasattr(logger, "_root_dir"):
        logger._root_dir = os.path.join(run_dir, "tb_logs")  # type: ignore[union-attr]

    callbacks = _instantiate_callbacks(cfg)
    _maybe_set_ckpt_dir(callbacks, ckpt_dir)

    # Trainer: our trainer/*.yaml contains a lightning.pytorch.Trainer target
    trainer = instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Optional: save the resolved config into the run dir for perfect reproducibility
    # Hydra already writes .hydra/config.yaml + overrides.yaml, but this is handy too.
    with open(os.path.join(run_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    # Fit (and optionally test)
    trainer.fit(lit_module, datamodule=datamodule)

    # If you want a minimal "always test after fit" pattern, uncomment:
    # trainer.test(lit_module, datamodule=datamodule)

    # Sync run directory to cloud storage via rclone (opt-in).
    # Set RCLONE_DEST to enable, e.g. RCLONE_DEST=gdrive:training-runs
    _sync_to_cloud(run_dir)


def _sync_to_cloud(run_dir: str) -> None:
    """Sync *run_dir* to cloud storage if ``RCLONE_DEST`` is set.

    The directory is uploaded to ``$RCLONE_DEST/<run_dir_basename>/``.
    Requires ``rclone`` to be installed and configured.
    """
    dest = os.environ.get("RCLONE_DEST")
    if not dest:
        return

    remote_path = f"{dest.rstrip('/')}/{os.path.basename(run_dir)}"
    log.info("Syncing %s -> %s", run_dir, remote_path)
    try:
        subprocess.run(
            ["rclone", "sync", run_dir, remote_path, "--progress"],
            check=True,
        )
        log.info("Cloud sync complete.")
    except FileNotFoundError:
        log.warning("rclone not found on PATH — skipping cloud sync.")
    except subprocess.CalledProcessError as exc:
        log.warning("rclone sync failed (exit %d) — run data is still in %s", exc.returncode, run_dir)


if __name__ == "__main__":
    main()
