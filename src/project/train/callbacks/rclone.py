from __future__ import annotations

import logging
import os
import subprocess

from decouple import config as decouple_config
from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)


def sync_to_cloud(run_dir: str, project: str | None = None, run_name: str | None = None) -> None:
    """Sync *run_dir* to cloud storage if ``RCLONE_DEST`` is set.

    The directory is uploaded to ``$RCLONE_DEST/<project>/<run_name>/``.
    Falls back to ``$RCLONE_DEST/<run_dir_basename>/`` if project/run_name
    are not provided.  Requires ``rclone`` to be installed and configured.
    """
    dest = str(decouple_config("RCLONE_DEST", default=""))
    if not dest:
        return

    if project and run_name:
        remote_path = f"{dest.rstrip('/')}/{project}/run_artifacts/{run_name}"
    else:
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


class RcloneSyncCallback(Callback):
    """Periodically sync the run directory to cloud storage via rclone.

    Runs after validation (so ModelCheckpoint has already saved) every
    *every_n_epochs* epochs.  No-ops gracefully if ``RCLONE_DEST`` is
    unset or rclone is not installed.
    """

    def __init__(self, every_n_epochs: int = 50, run_dir: str | None = None,
                 project: str | None = None, run_name: str | None = None) -> None:
        self.every_n_epochs = every_n_epochs
        self.run_dir = run_dir
        self.project = project
        self.run_name = run_name

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.run_dir is None:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs == 0:
            log.info("Periodic cloud sync at epoch %d", epoch)
            sync_to_cloud(self.run_dir, self.project, self.run_name)
