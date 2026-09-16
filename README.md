# BioAirMet

> **Acknowledgement.** The project 23NRM03 BioAirMet has received funding from the
> European Partnership on Metrology, cofinanced from the European Union's
> Horizon Europe Research and Innovation Programme and by the Participating
> States.

**BioAirMet-Classifier** classifies pollen particles captured with in-line
holographic images and fluorescence spectra. It is a two-stage framework:

1. **SSL pre-training** — self-supervised contrastive learning on *unlabelled* data
2. **Supervised classification** — training classifier head on the frozen learned encoders using *labelled* data

Everything runs through three CLI tools: `bioairmet-train`,
`bioairmet-validate`, and `bioairmet-inference`.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [I already have a trained classification experiment](#i-already-have-a-trained-classification-experiment)
- [Fine-tuning an existing classifier on your own dataset](#fine-tuning-an-existing-classifier-on-your-own-dataset)
- [Use your own encoder](#use-your-own-encoder)
- [Documentation](#documentation)
- [Project structure](#project-structure)
- [License](#license)
- [Citation](#citation)

---

## Requirements

Python ≥ 3.10, PyTorch 2.8.0 (CUDA build), plus `h5py`, `pandas`, `timm`,
`scikit-learn`, and a few more — the full pinned list is in
[`requirements.txt`](requirements.txt).

---

## Installation

**Recommended — automated installer** (creates a fresh venv, installs the
package editable + all deps, verifies the CLI entry points):

```bash
bash build_and_install.sh
```

**Or manually**, into any active Python ≥ 3.10 environment:

```bash
pip install -r requirements.txt
pip install -e .            # editable — changes to src/ take effect immediately
bioairmet-train --help      # verify the entry points
```

Both approaches support the `ENV_PATH` variable: after installing, either
activate the venv or just `export ENV_PATH=/path/to/env` and the `run_*.sh`
launcher scripts will pick it up.

**CUDA driver compatibility.** The installer checks your NVIDIA driver and
lets you choose the PyTorch build:

1. **Match your NVIDIA driver** (recommended) — installs the newest
   PyTorch build your driver can run (e.g. `cu128` wheels on a CUDA 12.8
   driver, `cu118` wheels on an older one);
2. **Latest** — the newest default PyPI build (currently bundles CUDA 12.8,
   so it needs a driver supporting ≥ 12.8 for GPU use);
3. **CPU only** — no CUDA libraries.

On a machine with an old driver the "match driver" build may be an older
PyTorch (the builder tells you, with a note) — updating the NVIDIA driver is
the clean way to get the latest build. In non-interactive contexts the
choice can be pinned with
`BIOMET_TORCH_BUILD=auto|driver|latest|cpu|cuXXX` (default `auto`).

---

## How it works

```
Unlabeled data
      │
      ▼
┌─────────────────────────┐
│  Stage 1: SSL Training  │  run_ssl_training.sh
└────────────┬────────────┘
             │  saves: experiment_dir/best_ssl_model.pth
             ▼
┌─────────────────────────────────────┐
│  Stage 2: Classification Training  │  run_classification_training.sh
└────────────────┬────────────────────┘
                 │  saves: experiment_dir/best_classification_model.pth
         ┌───────┴───────┐
         ▼               ▼
   Validate            Run Inference
   run_validation_v2.sh   run_inference.sh
```

Each stage reads an **HDF5 file** built from your raw events, writes a
**self-contained experiment directory** (resolved config + architecture +
checkpoints), and every downstream step (validation, inference, resume) only
needs that directory.

### Supported encoders

Both towers are selected by name in `architecture_setup` (see
[`src/bioairmet/config/architecture/architecture_base.yaml`](src/bioairmet/config/architecture/architecture_base.yaml)):

| Tower | `model_name` options |
|---|---|
| `image_tower` (grayscale holographic images) | `efficientnet_{b0…b7, v2_s, v2_m, v2_l, v1_100k, v1_250k, v2_100k, v2_250k}`, `mobilenet_v3_{tiny, small, large}`, `fastvit_{t8, t12, s12}`, `shufflenet_v2_{x0_5, x1_0, x1_5, x2_0}`, `mnasnet_{0_5, 0_75, 1_0, 1_3}` |
| `fluorescence_tower` (relative fluorescence spectra) | `FluorescenceMLP`, `FluorescenceMLP_Simple` |


Need a different architecture? Register your own in one file — see
[Use your own encoder](#use-your-own-encoder).

---

## Quick start

**1. Prepare the data files** — turn raw events into `unlabeled_data.h5`
(Stage 1) and `train_data.h5` / `test_data.h5` (Stage 2) with the provided
notebook [`notebooks/data.ipynb`](notebooks/data.ipynb).
Schema and details: [docs/data_preparation.md](docs/data_preparation.md).

**2. Stage 1 — SSL pre-training.** Point the SSL config at your unlabeled
data and run:

```yaml
# src/bioairmet/config/ssl/SSL_config_general.yaml
data:
  dataset_path: /path/to/unlabeled_data.h5   # CHANGE ME
```

```bash
bash run_ssl_training.sh
```

**3. Stage 2 — Classification.** Point the classification config at the
Stage-1 output and your labeled data, then run:

```yaml
# src/bioairmet/config/classification/config_general.yaml
model_initialization:
  pretraining:
    experiment_path: /path/to/ssl_experiment_dir   # CHANGE ME
data:
  train_data_path:  /path/to/train_data.h5         # CHANGE ME
  validation_path:  /path/to/test_data.h5          # CHANGE ME
```

```bash
bash run_classification_training.sh
```

> Tip: `bioairmet-train ... --show_config` prints the fully resolved config
> and a CHANGE-ME checklist before any training starts. Any remaining
> placeholder aborts the run **before** anything is written — and
> `--set KEY=VALUE` (repeatable) can override any config value, e.g.
> `--set train.epochs=5 --set data.val_split_ratio=0.05`.

> **No SSL stage?** Set `model_initialization.pretraining.enable: false` to
> train the classifier **from scratch** (randomly-initialised encoders). Both
> encoders are auto-unfrozen, so set the encoder LR multipliers to ~1.0 — see
> [Model initialization modes](docs/configuration.md#model-initialization-modes--learning-rate-guidance).

**4. Validate & infer.**

```bash
bash run_validation_v2.sh    # set CHECKPOINT_DIR + DATA_PATH inside first
bash run_inference.sh        # set EXPERIMENT_DIR + DATA_PATH + SAVE_PATH first
```

Full details (CLI flags, outputs, raw-directory mode, Python API):
[docs/inference.md](docs/inference.md).
---
## Download Model Weights

**You can download the model weights from [Zenodo](https://doi.org/10.5281/zenodo.22793167).**

* `SSL_Base_Model_.zip` — Contains the self-supervised base pre-trained encoders used for classification and transfer learning.
* `Classification_Base_Model_.zip` — Contains the base classification model.
* `Classification_TransferLearning_Model_.zip` — Contains the classification model trained with minimal data from the Turku, Novi Sad, and Córdoba datasets.

Extract the ZIP files into your desired directory and provide the full path when running inference or classification scripts.

---
## I already have a trained classification experiment

If you already have a trained classification experiment, you can skip straight to using it — the experiment directory is self-contained
(config + architecture + weights). Pick one:

```bash
# A) Launcher script (set the variables inside first):
bash run_inference.sh

# B) CLI directly:
bioairmet-inference \
    --experiment_dir /path/to/cls_experiment_dir \
    --data_path /path/to/data.h5 \
    --save_path ./inference_results \
    --checkpoint_type best \
    --gpu_id 0
```

```python
# C) Python API:
from bioairmet.training.inference import run_inference
result = run_inference(
    data_path="/path/to/data.h5",
    checkpoint_path="/path/to/cls_experiment_dir/best_classification_model.pth",
    save_path="./inference_results",
    config_path="/path/to/cls_experiment_dir/config.yaml",
    gpu_id=0,
)
```

**Data quality checks (raw-directory input only).** When `--data_path` is a raw
event directory, every event is quality-checked **by default — even when you
pass no flags at all**. Each event must pass three gates before it enters the
predictions CSV: `area >= --min_area` (default `500` px),
`solidity >= --min_solidity` (default `0.7`), and
`max − min intensity >= --min_intensity_delta` (default `0.1`). Events failing
any gate are dropped.

To relax or tighten the gates, override only the values you care about — e.g.
lower the area gate to 100 px while keeping the default solidity and intensity
gates:

```bash
bioairmet-inference \
    --experiment_dir /path/to/cls_experiment_dir \
    --data_path /path/to/raw_events \
    --save_path ./inference_results \
    --min_area 100
```

To disable the quality checks entirely, pass **`--skip_validation` on its
own** — the three `--min_*` flags are not needed (and are ignored):

```bash
bioairmet-inference \
    --experiment_dir /path/to/cls_experiment_dir \
    --data_path /path/to/raw_events \
    --save_path ./inference_results \
    --skip_validation
```

- The flags only change *which events are kept* — predictions for a given
  image are unaffected.
- Even with `--skip_validation`, every event still needs usable image files
  and JSON data (both image pairs must carry `rec_mag_properties` and the
  spectra must be extractable).
- The console prints `Number of valid spectra: X out of Y`; the run log
  records the final `Dataset size`.
- With a labeled HDF5 input these flags are ignored (the HDF5 was already
  filtered during data preparation).

Validation works the same way — see [docs/inference.md](docs/inference.md).

---

## Fine-tuning an existing classifier on your own dataset

Already have a trained classification experiment and new labeled data? Adapt the
existing model — encoders **and** classifier head — to your own dataset with the
`classification_finetuning` initialization mode. No SSL re-run needed.

**1. Copy the classification template** to your own config and fill in the paths:

```yaml
# your_cls_config.yaml (copy of src/bioairmet/config/classification/config_general.yaml)
model_initialization:
  pretraining:
    enable: true
    mode: "classification_finetuning"   # load encoders AND head from a previous classifier
    experiment_path: /path/to/existing_cls_experiment_dir   # CHANGE ME — the dir with
                                                           #   config.yaml, architecture.yaml, *.pth
    weights: "best"                     # 'best' or 'last'
data:
  train_data_path:  /path/to/your/train_data.h5   # CHANGE ME — your new labeled HDF5
  validation_path:  /path/to/your/test_data.h5    # CHANGE ME
  cat_map_path: /path/to/your/cats.txt            # CHANGE ME if your categories differ
logging:
  checkpoint_dir: /path/to/new_checkpoints        # CHANGE ME — a NEW experiment dir
train:
  epochs: 50
  scheduler:
    T_max: 50     # keep in sync with train.epochs
  
  optimizer:
    name: "AdamW"
    lr: 0.00001   # choose a small lr for fine-tuning
    weight_decay: 0.001
```

**2. Dry-run check, then train:**

```bash
bioairmet-train --config_path your_cls_config.yaml --mode classification --show_config
bash run_classification_training.sh your_cls_config.yaml
```

**What happens at startup:** the model architecture is pinned to the *old*
experiment's `architecture.yaml` (the backbone is unchanged), and the encoders
plus classifier head are loaded from `best_classification_model.pth` (or
`last_…`). Training then starts fresh — epoch 0, new optimizer — on your data.

> **Notes**
>
> - Keep `architecture_setup.classification_model.num_classes` the same as in the
>   old experiment — the saved head has to fit the rebuilt model. Classes that
>   are missing from your data are reported at startup (empty classifier heads;
>   see "Class coverage" in [docs/configuration.md](docs/configuration.md)), and
>   samples with labels ≥ `num_classes` are excluded.
> - Your dataset has **more** classes than the old experiment? The saved head
>   cannot be reused — fine-tune from the SSL experiment instead
>   (`mode: "ssl_pretraining"`, `experiment_path` → the SSL dir) or train from
>   scratch (`pretraining.enable: false`).
> - This is **not** `model_initialization.resume`: resume continues an
>   *interrupted* run on the *same* data (it restores epoch + optimizer state);
>   fine-tuning is a new run on new data.
> - Encoders stay frozen by default — only the classifier head trains. To also
>   adapt the encoders to your distribution, use the `train.fine_tuning` flags
>   (e.g. `unfreeze_image_encoder` / `unfreeze_last_img_layers`), see
>   [Freezing & fine-tuning](docs/configuration.md#freezing--fine-tuning).

---

## Use your own encoder

Swap either tower for your own architecture without touching the training
code. The package ships ready-to-modify templates in
[`src/bioairmet/models/custom_models.py`](src/bioairmet/models/custom_models.py)
— a small grayscale CNN for images and a 1-D convolutional spectrum encoder —
both already registered:

```yaml
# architecture_setup (or your architecture YAML)
image_tower:
  model_name: mycnn_small          # or mycnn_wide
fluorescence_tower:
  model_name: FluorescenceCNN
```

To add your own: define an `nn.Module` with `forward()` and
`get_output_dim()`, register it with `register_backbone(...)` /
`register_fluorescence_encoder(...)` in that file, and select it by name in
the config. Custom encoders work in **both** training stages, including
freezing, `unfreeze_last_*` and the BatchNorm switches.
Full walkthrough: [docs/custom_models.md](docs/custom_models.md).

---

## Documentation

| Topic | File |
|---|---|
| Raw data layout, HDF5 schema, labeled/unlabeled pipelines | [docs/data_preparation.md](docs/data_preparation.md) |
| Stage 1 & Stage 2 training, GPU configuration, `bioairmet-train` CLI | [docs/training.md](docs/training.md) |
| Config reference, data augmentation, config save/load, freezing & fine-tuning | [docs/configuration.md](docs/configuration.md) |
| Building & loading models, checkpoints, `load_model` | [docs/model_loading.md](docs/model_loading.md) |
| Custom image / fluorescence encoders, `register_backbone`, templates | [docs/custom_models.md](docs/custom_models.md) |
| Validation, inference, Python API, outputs | [docs/inference.md](docs/inference.md) |
| Common errors and log messages explained | [docs/troubleshooting.md](docs/troubleshooting.md) |

---

## Project structure

```
.
├── build_and_install.sh             # Automated installer
├── run_ssl_training.sh              # Stage 1 launcher
├── run_classification_training.sh   # Stage 2 launcher
├── run_validation_v2.sh             # Validation launcher
├── run_inference.sh                 # Inference launcher
├── requirements.txt
├── pyproject.toml
├── docs/                            # Guides (data, training, config, models, inference, troubleshooting)
├── notebooks/                       # Data-preparation notebook
├── scripts/                         # main.py — run from the source tree without installing
└── src/
    └── bioairmet/
        ├── cli.py                   # Entry points: train / validate / inference
        ├── config/                  # Architecture + stage YAML templates
        ├── data/                    # Dataset classes and data loaders
        ├── models/                  # Backbones, encoders, classifier architectures
        │   └── custom_models.py     # Custom-encoder templates (register yours here)
        ├── trainers/                # V2 trainer base and specializations
        ├── training/                # inference.py — inference engine
        └── utils/                   # Losses, metrics, optimizers, calibration,
                                     # plot_augmentations.py (augment-viz tool)
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

If you use BioAirMet in your research, please cite it as the
**23NRM03 BioAirMet** project (Physikalisch-Technische Bundesanstalt, PTB,
within the European Partnership on Metrology) and refer to
<https://www.bioairmet.ptb.de/home> for the project's publications.

```bibtex
@misc{bioairmet2026,
  title  = {Enhanced Pollen Classification Using Unlabelled Environmental Data and Self-Supervised Learning},
  author = {Nabawi, M., Martin, J., Bilson, S., Kadantsev, E., Zeder, Y., Schwendimann, A., Crouzy, B., Erb, S., Haufe, S., & Klein, T. (2026). },
  year   = {2026},
  url    = {https://www.bioairmet.ptb.de/home},
  note   = {Project 23NRM03, European Partnership on Metrology}
}
```
