# ──────────────────────────────────────────────────────────────
# Makefile — training template
#
# One-stop shop for environment setup, training, and monitoring.
# All commands run through `uv run` so you don't need to
# activate the venv manually.
#
# Quick start:
#   make setup          # create venv + install deps
#   make smoke_mps      # sanity check on Mac GPU
#   make tensorboard    # monitor in browser (localhost:6006)
#
# Override variables from the CLI:
#   make smoke_mps EPOCHS=50
#   make train_ddp DEVICES=4
# ──────────────────────────────────────────────────────────────

UV     ?= uv
PY     ?= $(UV) run python
TRAIN   = $(PY) -m project.train.run

DEVICES ?= 2
EPOCHS  ?= 1

# ── Help ─────────────────────────────────────────────────────

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Usage: make <target> [VAR=value ...]"
	@echo ""
	@echo "Setup:"
	@echo "  setup            Create venv and install dependencies"
	@echo "  sync             Sync dependencies (after editing pyproject.toml)"
	@echo ""
	@echo "Training:"
	@echo "  smoke_cpu        Quick sanity run on CPU (EPOCHS=1)"
	@echo "  smoke_mps        Quick sanity run on Mac MPS (EPOCHS=1)"
	@echo "  train_gpu        Single-GPU CUDA run"
	@echo "  train_ddp        Multi-GPU DDP run (DEVICES=2)"
	@echo ""
	@echo "Monitoring:"
	@echo "  tensorboard      Launch TensorBoard at http://localhost:6006"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean_outputs    Remove Hydra output directories"
	@echo "  clean_runs       Remove old TensorBoard run logs (runs/)"
	@echo "  clean            Remove both outputs and runs"
	@echo ""
	@echo "Variables: EPOCHS=$(EPOCHS)  DEVICES=$(DEVICES)"

# ── Environment setup ────────────────────────────────────────

# Create venv and install all dependencies in one step.
.PHONY: setup
setup:
	$(UV) venv
	$(UV) sync

# Re-sync dependencies (e.g. after adding a new package to pyproject.toml).
.PHONY: sync
sync:
	$(UV) sync

# ── Training ─────────────────────────────────────────────────

# Minimal CPU smoke test — verifies the full pipeline runs end-to-end.
.PHONY: smoke_cpu
smoke_cpu:
	$(TRAIN) trainer=cpu trainer.max_epochs=$(EPOCHS) data=dummy model=simple data.batch_size=16

# MPS smoke test — same as CPU but on Mac GPU.
.PHONY: smoke_mps
smoke_mps:
	$(TRAIN) trainer=mps trainer.max_epochs=$(EPOCHS) data=dummy model=simple data.batch_size=16

# Single-GPU CUDA training with mixed precision.
.PHONY: train_gpu
train_gpu:
	$(TRAIN) trainer=gpu_1 trainer.max_epochs=10 precision=16-mixed

# Multi-GPU DDP training. Override GPU count: make train_ddp DEVICES=4
.PHONY: train_ddp
train_ddp:
	$(TRAIN) trainer=ddp trainer.devices=$(DEVICES) trainer.max_epochs=10 precision=bf16-mixed

# ── Monitoring ───────────────────────────────────────────────

# Launch TensorBoard to view training curves in the browser.
# Polls for new data every 5 seconds.
.PHONY: tensorboard
tensorboard:
	$(UV) run tensorboard --logdir outputs/ --reload_interval 5

# ── Cleanup ──────────────────────────────────────────────────

# Remove Hydra output directories (configs, logs, checkpoints).
.PHONY: clean_outputs
clean_outputs:
	rm -rf outputs

# Remove TensorBoard run logs.
.PHONY: clean_runs
clean_runs:
	rm -rf runs

# Remove everything.
.PHONY: clean
clean: clean_outputs clean_runs
