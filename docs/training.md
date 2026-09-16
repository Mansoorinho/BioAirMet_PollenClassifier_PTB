# Training Guide

Details for both training stages (SSL pre-training and classification),
GPU configuration, and the `bioairmet-train` CLI.

**Related:** [Configuration & fine-tuning guide](configuration.md) ·
[Data preparation guide](data_preparation.md) · [Model building & loading guide](model_loading.md)

---

## Stage 1: SSL pre-training

Edit `src/bioairmet/config/ssl/SSL_config_general.yaml` and at minimum set:

```yaml
data:
  dataset_path: /path/to/unlabeled_data.h5   # CHANGE ME

logging:
  checkpoint_dir: /path/to/ssl_checkpoints   # CHANGE ME
```

Optional: `model_initialization.pretrained` (warm start from an existing SSL
checkpoint) or `model_initialization.resume` (continue an interrupted run —
the architecture is taken from the checkpoint's experiment directory).

Then run:

```bash
bash run_ssl_training.sh
# or with a custom config:
bash run_ssl_training.sh /path/to/custom_ssl_config.yaml
```

Or call the CLI directly (the dry-run flag prints the fully resolved config
and a CHANGE-ME checklist without starting training):

```bash
bioairmet-train --config_path src/bioairmet/config/ssl/SSL_config_general.yaml --mode ssl
bioairmet-train --config_path src/bioairmet/config/ssl/SSL_config_general.yaml --mode ssl --show_config
```

---

## Stage 2: Classification training

Edit `src/bioairmet/config/classification/config_general.yaml` and at minimum set:

```yaml
model_initialization:
  pretraining:
    experiment_path: /path/to/ssl_experiment_dir   # CHANGE ME — Stage 1 output dir

data:
  train_data_path: /path/to/labeled_data.h5        # CHANGE ME

logging:
  checkpoint_dir: /path/to/cls_checkpoints         # CHANGE ME
```

Then run:

```bash
bash run_classification_training.sh
# or with a custom config:
bash run_classification_training.sh /path/to/custom_cls_config.yaml
```

Or call the CLI directly:

```bash
bioairmet-train --config_path src/bioairmet/config/classification/config_general.yaml --mode classification
```

Whether the pretrained encoders are frozen or fine-tuned (fully, or "last N
layers") is controlled by `train.fine_tuning` — see
[Freezing & fine-tuning](configuration.md#freezing--fine-tuning).

**Point at a *finished* Stage-1 run.** `pretraining.weights: last` reads a file the
SSL run rewrites at the end of every epoch, so launching Stage 2 while Stage 1 is
still training picks up an arbitrary epoch — and two Stage-2 runs started minutes
apart then differ in their initialisation, not just in the setting you are
comparing. The builder logs the file, its age and its `BN num_batches_tracked`
fingerprint, and warns with `[V2 Builder][STALE-INIT]` when the file is younger
than two minutes. For a comparison you intend to trust: wait for Stage 1 to
finish, or copy its checkpoint to a private directory and point
`experiment_path` at the copy. See
[Model initialization modes](configuration.md#model-initialization-modes--learning-rate-guidance).

**Train from scratch (no SSL).** Set `pretraining.enable: false` to skip the
Stage-1 weights entirely; the builder auto-unfreezes both encoders and warns
you to use ~1.0 encoder LR multipliers (the `0.001` fine-tuning default leaves
randomly-initialised encoders barely learning). See
[Model initialization modes](configuration.md#model-initialization-modes--learning-rate-guidance).

**Resume an interrupted run.** Set `model_initialization.resume.enable: true`
plus `resume.checkpoint_path` — the architecture is resolved from the
checkpoint's experiment directory (same self-contained rule as inference), so
you only need the checkpoint path.

---

## Reading the BatchNorm / freeze lines in `training.log`

| Line | When | What it tells you |
|---|---|---|
| `--- BatchNorm / Freeze audit [PRE-POLICY state: ...] ---` | once, at startup | Two tables: per-module trainable/total parameters and per-module BatchNorm state (`mode(tr/ev)`, `track`, `updating`, `affine(tr/tot)`). Labelled **PRE-POLICY** because the epoch loop has not run yet. |
| `[BN-policy] update_batchnorm_stats_...: true and PENDING (not a problem)` | once, at startup | The flag is set but *cannot* be visible yet: `_apply_fine_tuning_policy()` only records the intent, and `enforce_frozen_modes()` has not run. Nothing to fix. |
| `[BN-state] epoch N after model.train()/enforce_frozen_modes: img_encoder[BN 49: eval 0, train 49; updating 49; affine-trainable 0/98]; ...` | every epoch | The authoritative reading. `updating N/M` = BN layers in `train()` mode **and** tracking running stats — the only case where buffers change. `update_batchnorm_stats_img_encoder: true` must give `updating 49/49`; the frozen default must give `updating 0/49`. |
| `[BN-DIVERGENCE] ...` | at the first epoch, once | A switch really has no effect (or `track_running_stats=False`, or trainable parameters missing from the optimizer). This is the case worth acting on. |
| `[BN-DDP] ...` | only with `update_batchnorm_stats_*: true` on more than one GPU | DDP synchronises gradients, not BatchNorm **buffers**: every rank builds its own running-statistics estimate and only rank 0's is saved. Use `nn.SyncBatchNorm.convert_sync_batchnorm()` if that matters. |
| `[V2 Builder] Initialising from checkpoint ... modified ... source BN num_batches_tracked=...` | once, at startup | Which Stage-1 file this run started from, and how trained it was — two runs with different values here are not comparable. |

Why the startup audit cannot judge `update_batchnorm_stats_*`: a BatchNorm updates
its buffers only while it is in `train()` mode, and the encoders are in `eval()`
right after the build (that is the freeze policy doing its job). The first
`[BN-state]` line is the first moment the request can be checked — with
`update_batchnorm_stats_*: true` and the encoders otherwise frozen, expect the
image encoder's buffers to advance **twice per batch** (once per view), which is
also visible as `num_batches_tracked` growing by `2 × steps_per_epoch` per epoch.

---

## GPU configuration

GPU mode is controlled at two levels (both must agree):

**1. Bash script — physical visibility.** Set `GPU_IDS` in the training
scripts (the script exports `CUDA_VISIBLE_DEVICES="${GPU_IDS}"`):

```bash
GPU_MODE="single"; GPU_IDS="0"        # single GPU
GPU_MODE="multi";  GPU_IDS="0,1,2,3"  # multi-GPU (DDP)
```

**2. Config YAML — distributed training settings:**

```yaml
# Use every GPU the launcher script made visible (the default):
distributed:
  enable: true
  world_size: -1     # -1 = one process per visible GPU (DDP)
  gpu_ids: []        # [] = all visible GPUs

# Pin a single specific GPU:
distributed:
  enable: true
  world_size: 1      # 1 = single process
  gpu_ids: [0]       # which visible GPU index to use
```

> **Important**: if `GPU_IDS="0,1"` in the bash script but `world_size: 1` in
> the YAML, only GPU 0 will be used. The launcher's `GPU_IDS` decides which
> physical GPUs are *visible*; the YAML decides how many of them the run
> *uses*.

---

## CLI reference: `bioairmet-train`

```
bioairmet-train --config_path PATH --mode {ssl,classification}
                [--show_config] [--set KEY=VALUE ...]
```

| Argument | Required | Description |
|---|---|---|
| `--config_path` | Yes | Path to the YAML config file |
| `--mode` | Yes | `ssl` or `classification` |
| `--show_config` | No | Print the fully resolved config (architecture merged in) + a CHANGE-ME checklist, then exit without training |
| `--set` | No | Repeatable `KEY=VALUE` override (dotted path, YAML-parsed value). Applied after the architecture merge, so it wins over the file — e.g. `--set train.epochs=5 --set distributed.gpu_ids="[0, 1]"`. Useful for one-off runs without editing the template. |

A normal run (without `--show_config`) first performs a **preflight check**
and aborts before creating anything if a `CHANGE ME` placeholder is still
present, the config shape is invalid, or the requested GPUs exceed what is
visible — see [Configuration guide](configuration.md).
