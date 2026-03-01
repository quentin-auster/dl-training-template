from __future__ import annotations

import logging
import os
from typing import Optional

from decouple import config as decouple_config
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("add", lambda *args: sum(int(a) for a in args))

import lightning as L
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from hydra.utils import instantiate

from wonderwords import RandomWord

from project.train.callbacks import RcloneSyncCallback, sync_to_cloud

log = logging.getLogger(__name__)


def _generate_run_name(cfg: DictConfig) -> str:
    """Generate a run name like 'zesty-causal_lm-p113-adamw-gpu_1'."""
    adjective = RandomWord().word(include_parts_of_speech=["adjectives"])
    choices = HydraConfig.get().runtime.choices
    parts = [
        adjective,
        choices.get("model", "model"),
    ]
    # Include modulus when using modular addition data.
    modulus = cfg.get("data", {}).get("modulus")
    if modulus is not None:
        parts.append(f"p{modulus}")
    parts += [
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


def _maybe_set_sync_info(callbacks: list[Callback], run_dir: str,
                         project: str | None, run_name: str | None) -> None:
    """Set run_dir, project, and run_name on any RcloneSyncCallback."""
    for cb in callbacks:
        if isinstance(cb, RcloneSyncCallback):
            if cb.run_dir is None:
                cb.run_dir = run_dir
            if cb.project is None:
                cb.project = project
            if cb.run_name is None:
                cb.run_name = run_name


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
    if (cfg.get("logger") or {}).get("_target_", "").endswith("WandbLogger"):
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

    project = cfg.get("run", {}).get("project")
    run_name = cfg.run.name

    callbacks = _instantiate_callbacks(cfg)
    _maybe_set_ckpt_dir(callbacks, ckpt_dir)
    _maybe_set_sync_info(callbacks, run_dir, project, run_name)

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
    ckpt_path = cfg.get("run", {}).get("ckpt_path")
    # weights_only=False needed because Lightning checkpoints contain OmegaConf
    # objects which PyTorch 2.6+ blocks under the default weights_only=True.
    trainer.fit(lit_module, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

    # If you want a minimal "always test after fit" pattern, uncomment:
    # trainer.test(lit_module, datamodule=datamodule)

    # Sync run directory to cloud storage via rclone (opt-in).
    # Set RCLONE_DEST to enable, e.g. RCLONE_DEST=gdrive:training-runs
    sync_to_cloud(run_dir, project, run_name)


if __name__ == "__main__":
    main()
