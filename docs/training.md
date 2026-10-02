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

Resume restores the complete training state: model weights, the optimizer
(step counts and AdamW moments), the LR scheduler's *progression* (epoch
position and base LRs — the resumed run keeps the **current** config's
schedule hyper-parameters, e.g. an extended `T_max` when you resume to train
more epochs than the original run planned), the AMP scaler, the best metric,
and the early-stopping patience window. Checkpoints are written after each
epoch's early-stopping evaluation, so resuming is exactly equivalent to
continuing the uninterrupted run: the first resumed epoch trains at the
learning rate the uninterrupted run would have used, and the final
scheduler/optimizer states are identical (verified by the resume round-trip
test for both SSL and classification).

Invalid combinations fail fast before anything is created, e.g.
`resume.enable: true` with an empty or missing `checkpoint_path`, or
`pretrained.enable: true` together with `resume.enable: true` (enable at most
one — a resumed run takes its weights from the checkpoint, not from a
pretrained bundle).

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

## DataLoader worker seeding (`train.worker_seeding`)

One master on/off switch per stage (both templates default `True`), resolved by
`worker_seeding_enabled()` in `src/bioairmet/trainers/base_trainer_v2.py`. The
legacy key `train.vary_augmentation_seed_per_epoch` is still honoured as a
fallback when the new key is absent.

**ON (default) — the "robust" scheme.** PyTorch's DataLoader already seeds each
worker's torch RNG from a fresh draw of the main process RNG plus the worker id
(before `worker_init_fn` runs). `build_worker_init_fn` does NOT touch that
generator — the auto-seeded stream is the one that gave the most robust SSL
runs — and seeds only what the loader does not manage, with
`torch.initial_seed()` (the auto-seed value, already distinct per worker and per
DDP rank, and reproducible across runs with the same `config.seed`):

* the module-level `random` / `numpy.random` generators (fork-duplicated across
  workers without this), and
* the augmentation pipeline's own RNG (`ImageAugmentation_Generic`'s internal
  `random.Random`), also fork-duplicated — without it, all workers replay the
  *same* rotation/translation sequence, and because the DataLoader hands
  sample *i* to worker `i % num_workers`, one batch ends up sharing a handful
  of angles instead of getting one per sample.

The same function registers a shared epoch counter on the dataset
(`HDF5Dataset.attach_augmentation_epoch`); the trainer publishes each new epoch
into it (`BaseTrainerV2._advance_augmentation_epoch`), and the worker refreshes
the `random`/`numpy`/augmentation streams at
`torch.initial_seed() + epoch * 100003`
(`HDF5Dataset._sync_augmentation_epoch`) — the torch RNG still keeps its
auto-seed for the whole run. This matters because the training loader uses
`persistent_workers=True`: workers are forked once, so `worker_init_fn` runs
once.

**OFF — no seeding at all.** No `worker_init_fn` is passed to the loader: we seed
nothing. PyTorch still auto-seeds the worker torch RNG, and the
`random`/`numpy`/augmentation RNGs keep the fork-duplicated state (identical
streams in every worker). This is the bare "no reproducibility" setting; for
SSL with the legacy pipeline (torch-driven only, `fluo_aug_prob: 0`) the data
stream is then exactly the one without our intervention.

Properties worth knowing:

* **ON is reproducible** across runs with the same `config.seed` (the auto-seed
  is drawn from the main RNG whose state is deterministic), with distinct
  streams per worker and per rank.
* **Cost** of the per-epoch refresh: one shared-memory read plus one integer
  comparison per sample; the re-seeding happens once per worker per epoch.
* **Prefetching**: samples already queued when the epoch advances keep the old
  stream, so the exact non-torch stream depends on `num_workers` / prefetch
  depth (as it always does).
* **`num_workers: 0` is unaffected** either way — `worker_init_fn` never runs,
  and the main process' generators are left alone (they also drive model
  internals such as dropout).
* Both stage logs print the active policy as a `Worker RNG seeding: …` line at
  the start of training.

This is a diversity knob, not a known accuracy win: which of ON/OFF is better
has to be decided over several seeds per stage, the same as any other
augmentation change.

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
