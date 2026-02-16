# Project: training-template

PyTorch Lightning + Hydra template for training small transformers, focused on modular addition grokking and mechanistic interpretability.

## Commands

```bash
uv sync                          # install deps
uv run python -m project.train.run trainer=cpu model=causal_lm data=modular trainer.max_epochs=1  # smoke test
./scripts/smoke_local.sh         # MPS smoke test (20 epochs)
./scripts/train_gpu.sh           # single CUDA GPU (1000 epochs, fp16)
./scripts/train_ddp.sh           # multi-GPU DDP (1000 epochs, bf16)
./scripts/setup_vm.sh            # VM setup (installs uv, rclone, deps)
```

## Key architecture

- **Training**: Hydra config → `train/run.py` → instantiates LitCausalLM + DataModule → `trainer.fit()`
- **Model**: `TinyTransformer` (HookedRootModule) wrapped by `LitCausalLM` (LightningModule)
- **Interp**: TransformerLens hooks enable `model.run_with_cache()` and `model.run_with_hooks()`

## Source map (`src/project/`)

### `lit_causal_lm.py`
LitCausalLM — Lightning wrapper. Handles loss (cross_entropy, ignore_index=-100), accuracy on supervised positions, AdamW + linear warmup. Imports TinyTransformer from `models.examples`.

### `models/examples.py`
TinyTransformer (HookedRootModule), Attention, MLP, TransformerBlock — all with HookPoints. Hook names follow pattern: `blocks.{i}.attn.hook_{q,k,v,attn_scores,attn_pattern,z,result}`, `blocks.{i}.mlp.hook_{pre,post,result}`, `blocks.{i}.hook_resid_{pre,mid,post}`, `hook_embed`, `hook_pos_embed`, `hook_resid_final`.

### `data/`
- `tokenize.py` — `Vocab` dataclass, `build_shared_vocab(modulus)`, `causal_lm_collate(items, pad_id)`
- `modular.py` — `ModularAdditionConfig`, `ModularAdditionDataset` (random), `FullModularAdditionDataset` (all p² pairs, deterministic split)
- `dyck.py` — `DyckConfig`, `DyckNextTokenDataset` (balanced parentheses)
- `lit_data.py` — `ModularAdditionDataModule` (Lightning DataModule)

### `train/`
- `run.py` — Hydra entrypoint (`@hydra.main`). Seeds, instantiates components, fits, syncs to cloud via `RCLONE_DEST`.
- `loop.py` — Generic `train_epoch`, `eval_epoch` (placeholder, unused — Lightning handles training)
- `losses.py` — `LOSSES` registry dict, `get_loss`, `register_loss` (placeholder)
- `metrics.py` — `METRICS` registry dict, `get_metric`, `register_metric` (placeholder)
- `optim.py` — `configure_adamw`, `cosine_schedule_with_warmup`, `get_grad_norm` (placeholder — LitCausalLM has its own optimizer)

### `interp/`
- `ablate.py` — `AblationResult`, `zero_ablation_hook`, `mean_ablation_hook`, `run_with_ablation`, `ablation_sweep`, `head_ablation_sweep`, `compute_component_importance`
- `patch.py` — `PatchingResult`, `patching_hook`, `run_with_patch`, `activation_patching`, `path_patching`, `create_corrupted_input`
- `probes.py` — `LinearProbe`, `ProbeResult`, `train_probe`, `probe_all_layers`
- `viz.py` — `plot_attention_pattern`, `plot_attention_heads`, `plot_activation_norms`, `plot_ablation_results`, `plot_patching_heatmap`, `plot_probe_accuracy_by_layer`

### `utils/helpers.py`
- `auto_device()` — returns `cuda`/`mps`/`cpu`
- `load_checkpoint(path, device)` → `(LitCausalLM, TinyTransformer)`
- `find_latest_checkpoint(base_dir)` → newest `last.ckpt`
- `to_numpy(tensor)` — `.detach().cpu().float().numpy()`
- `make_batch(dataset, n, collate_fn, device)` → dict of tensors

## Configs (`configs/`)

- `config.yaml` — defaults: model=simple, data=dummy, optim=adamw, trainer=cpu, logger=tensorboard, callbacks=default
- `model/causal_lm.yaml` — Nanda grokking: 1 layer, 4 heads, d_model=128, d_mlp=512, ReLU, no LN, untied embeds, lr=1e-3, wd=1.0
- `model/simple.yaml` — placeholder (references non-existent LitModel)
- `data/modular.yaml` — modulus=113, frac_train=0.3, answer_only_supervision, batch_size=4096
- `data/dummy.yaml` — placeholder (references non-existent DummyDataModule)
- `optim/adamw.yaml` — placeholder (lr, wd, betas)
- `trainer/{cpu,mps,gpu_1,ddp}.yaml` — Lightning Trainer configs
- `logger/{tensorboard,wandb}.yaml` — logger configs
- `callbacks/default.yaml` — ModelCheckpoint (save_last, monitor val_loss) + LearningRateMonitor

## Notebooks

- `notebooks/examples/modular_interp.ipynb` — full mech interp walkthrough: load checkpoint → attention patterns → residual stream norms → head ablation → Fourier analysis of embeddings → linear probing by position

## Cloud sync

Set `RCLONE_DEST=gdrive:training-runs` (or any rclone remote) + `run.project=<name>` to sync to `RCLONE_DEST/<project>/run_artifacts/<run_name>/`. Syncs periodically (every 50 epochs via `RcloneSyncCallback`) and after `trainer.fit()`. Headless VMs: set `RCLONE_CONF_B64` in `.env` for auto-config. W&B via `logger=wandb logger.project=...`.
