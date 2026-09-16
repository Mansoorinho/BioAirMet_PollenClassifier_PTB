# Troubleshooting

Common errors, warnings and log lines explained — installer, config, GPU and
training-runtime issues.

**Related:** [Training guide](training.md) · [Configuration guide](configuration.md) · [Custom encoders](custom_models.md)

---

## Installation

**`torch.cuda.is_available()` is False after install.**
Your NVIDIA driver is older than the CUDA runtime bundled with the installed
PyTorch wheels. Re-run `bash build_and_install.sh` and choose
**"Match your NVIDIA driver"** (or pin non-interactively with
`BIOMET_TORCH_BUILD=driver` / `cu118` / …); the installer picks a wheel series
your driver can run. Updating the driver is the clean way to stay on the
latest build. CPU-only installs are expected to report `False`.

**`bioairmet-train: command not found`.**
The venv is not active. Activate it or `export ENV_PATH=/path/to/env` for the
launcher scripts. Re-install with `pip install -e .` if needed.

---

## Config & startup

**Training aborts with a `CHANGE ME` message before anything runs.**
The preflight check refuses to start while placeholder paths
(`/path/to/...`) remain in the config — this is intentional (nothing is
written). Fill in the fields listed in the message, or override from the
command line: `bioairmet-train ... --set data.dataset_path=/real/path.h5`.

**`ConfigError: Could not resolve the architecture config`.**
Classification runs pin their architecture to the SSL experiment bundle. Fix one of:
- `model_initialization.pretraining.experiment_path` → the SSL experiment
  directory (must contain `architecture.yaml`);
- for a **train-from-scratch** run set `model_initialization.pretraining.enable: false`
  (the architecture then comes from `architecture_setup.architecture_config_name`
  / `architecture_config_path`);
- for **resume**, `model_initialization.resume.checkpoint_path` → a checkpoint
  inside the original experiment directory.

**`Unknown image encoder name` / `Unknown fluorescence encoder name`.**
The name must match a registered family/encoder (see the
[supported encoders table](../README.md#supported-encoders)). Custom names
require the module that registers them to be imported — see
[Custom encoders](custom_models.md).

---

## GPU / distributed

**Only one GPU is used although several are visible.**
Two levels must agree: the launcher's `GPU_IDS` (which GPUs are *visible*) and
the YAML's `distributed.world_size` / `distributed.gpu_ids` (how many the run
*uses*). `GPU_IDS="0,1"` with `world_size: 1` runs on GPU 0 alone.

**`CUDA out of memory` at the first batch.**
Lower `data.batch_size` (per process — DDP multiplies total memory by the
number of processes), or enable `train.mixed_precision: True`. The
`channels_last` switch changes memory layout, not the budget.

---

## Training-runtime log lines

| Log line | Meaning | Action |
|---|---|---|
| `[V2 Builder][STALE-INIT] …` | Stage 2 was launched while the referenced SSL checkpoint is younger than two minutes — the Stage-1 run may still be training, so the initialisation is an arbitrary epoch. | Wait for Stage 1 to finish (or copy its checkpoint and point `experiment_path` at the copy) before trusting comparisons. |
| `[BN-policy] … PENDING` | Informational, not a warning: `update_batchnorm_stats_*: true` is stored at build time but can only take effect once the epoch loop runs `model.train()` + `enforce_frozen_modes()`. | Read the per-epoch `[BN-state]` line to confirm (`updating N/M` with N > 0). |
| `[BN-DIVERGENCE] …` | A `*_batchnorm_*` switch has no effect on the live model, a BN layer has `track_running_stats=False`, or `unfreeze_last_*_layers` selected no parameters. | Fix the config (the message names the switch); see [Freezing & fine-tuning](configuration.md#freezing--fine-tuning). |
| `[FREEZE-DIVERGENCE] …` | Parameters with `requires_grad=True` are missing from the optimizer — they would never update. | Usually a custom optimizer setup; with the shipped builders this should not happen — please report it. |
| `[BN-DDP] …` | `update_batchnorm_stats_*: true` on a DDP-wrapped model: BN buffers are per-rank, only rank 0's end up in the checkpoint. | Accept (frozen encoders unaffected) or convert with `nn.SyncBatchNorm.convert_sync_batchnorm()`. |
| `[augmentation] WARNING: …` | A typo inside a known augmentation entry (e.g. `kernal_size`), or a missing `enabled` key. | Fix the key; the transform is *not* silently ignored, but its config is incomplete. |

**`[BN-state] … updating 0/49` in an epoch line** while
`update_batchnorm_stats_*: true`: the flag is not in force (a `[BN-DIVERGENCE]`
warning accompanies it). `updating 49/49` proves it is.

**Loading a checkpoint logs many missing keys.**
Expected when loading an encoder-only SSL checkpoint into the full
classification model (`strict=False` load) — the classifier head legitimately
has no weights yet. Unexpected *unexpected* keys usually mean a backbone
mismatch between config and checkpoint.

---

## torchvision compatibility

The EfficientNet config classes were renamed between torchvision 0.24 and
0.25 (`EfficientNetBlockConfig` → `MBConvConfig`); `models/backbones.py`
imports either. If you hit an import error from `torchvision.models.efficientnet`
after upgrading by hand, align your torchvision with
[`requirements.txt`](../requirements.txt).