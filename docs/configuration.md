# Configuration Guide

Key configuration fields, how configs are saved & loaded, and the
freezing / fine-tuning semantics of Stage 2.

**Related:** [Training guide](training.md) · [Data preparation guide](data_preparation.md) · [Model building & loading guide](model_loading.md)

---

## Config file reference

Both stage templates share one fixed section layout —
`RUN → DATA → MODEL INITIALIZATION → ARCHITECTURE → TRAINING → OUTPUT →
DISTRIBUTED` — with USER SETTINGS first and ADVANCED last inside each
section. Everything you need to edit is marked `CHANGE ME`:

```yaml
# ─── RUN ─────────────────────────────────────────────────────────────────────
project_name: my_experiment              # used in the experiment folder / log names
seed: 42

# ─── DATA ────────────────────────────────────────────────────────────────────
data:
  dataset_path: /path/to/data.h5      # SSL config — unlabeled HDF5 (CHANGE ME)
  train_data_path: /path/to/data.h5   # classification config — labeled HDF5 (CHANGE ME)
  validation_path: /path/to/data.h5   # classification config (CHANGE ME)
  cat_map_path: /path/to/cats.txt     # classification only; falls back to built-in cats_dict.txt

# ─── MODEL INITIALIZATION ────────────────────────────────────────────────────
# Where the starting weights come from (unified shape in both stages):
model_initialization:
  # SSL stage — warm start from an existing SSL checkpoint (mutually
  # exclusive with resume):
  pretrained:
    enable: false
    checkpoint_path: null
  # classification — fine-tune / transfer / from scratch:
  pretraining:
    enable: true
    experiment_path: /path/to/ssl_dir   # CHANGE ME — Stage 1 output dir
    weights: last                       # 'best' or 'last'
  # both stages — continue an interrupted run:
  resume:
    enable: false
    checkpoint_path: null
    load_optimizer: false               # SSL only

# ─── OUTPUT / LOGGING ────────────────────────────────────────────────────────
logging:
  checkpoint_dir: /path/to/outputs    # CHANGE ME
  tensorboard: false                  # write TensorBoard event files

# ─── DISTRIBUTED (GPUs) ──────────────────────────────────────────────────────
# gpu_ids are indices INTO the GPUs made visible by the launcher script's
# GPU_IDS variable (CUDA_VISIBLE_DEVICES).
distributed:
  enable: true
  world_size: -1     # -1 = all visible GPUs, 1 = single GPU, or a count
  gpu_ids: []        # [] = all visible GPUs, [0] = first visible GPU
```

The model architecture (backbones, encoders, projector) is **not** part of
the stage config — it comes from an architecture file in
`src/bioairmet/config/architecture/` (see below).

---

## How configs are saved & loaded

BioAirMet uses a **single fail-fast architecture merge**. Every stage config
points at a model architecture (`architecture_setup`); at startup the
architecture file is resolved, deep-merged into `architecture_setup`
(stage-config keys win), and the run either starts or fails with an actionable
error that lists every path that was searched.

**Resolution order** (first existing file wins):

1. `<experiment_dir>/architecture.yaml` — when loading an experiment bundle
   (validation / inference / model reuse).
2. classification + `model_initialization.resume.enable: true` — the
   architecture next to `resume.checkpoint_path`.
3. SSL + resume enabled (`model_initialization.resume`, or the legacy
   top-level `resume:` section in old configs) — the architecture next to
   the checkpoint, the same self-contained rule.
4. classification + `pretraining.experiment_path` set — the **SSL bundle's**
   `architecture.yaml` (stage 2 is deliberately locked to the exact model shape
   the SSL weights were trained with).
5. `architecture_setup.architecture_config_name` /
   `architecture_setup.architecture_config_path` — the package's
   `config/architecture/` folder (fresh SSL training, or classification
   train-from-scratch).

### Command-line overrides (`--set`)

Any config value can be overridden from the command line without touching
the file. `--set` is repeatable; values are parsed as YAML (numbers, booleans,
null, lists):

```bash
bioairmet-train --config_path config.yaml --mode ssl \
    --set train.epochs=5 \
    --set data.val_split_ratio=0.05 \
    --set distributed.gpu_ids="[0, 1]"
```

Overrides are applied **after** the architecture merge, so they win over
everything in the file — including filling `CHANGE ME` placeholders for a
one-off run.

---

## Model architecture

The model is **not** configured in the stage config — it comes from an
architecture file, deep-merged into `architecture_setup` at startup (see
[How configs are saved & loaded](#how-configs-are-saved--loaded)). You pick
one in `architecture_setup` of the stage config:

```yaml
architecture_setup:
  # Preset file in src/bioairmet/config/architecture/ (the default):
  architecture_config_name: "architecture_base.yaml"
  # …or any file with the same shape, instead of the preset name:
  # architecture_config_path: /path/to/my_architecture.yaml
```

**Shipped preset** (`src/bioairmet/config/architecture/`):

| `architecture_config_name` | Image backbone | Projection dim |
|---|---|---|
| `architecture_base.yaml` | `efficientnet_b0` | 256 |

To use a different model, copy this file, point
`architecture_config_path` at the copy, and change any of the three components
below.

### Image backbone — `image_tower.model_name`

| Family | One-line summary | Variants |
|---|---|---|
| `efficientnet` | The workhorse mobile-CNN family with a smooth accuracy/size trade-off; the `v1_*/v2_*` 100k/250k variants are custom ultra-tiny, random-init models for small datasets. | `b0`–`b7`, `v2_s`, `v2_m`, `v2_l`, `v1_100k`, `v1_250k`, `v2_100k`, `v2_250k` |
| `mobilenet_v3` | Lightweight inverted-residual CNN with squeeze-and-excitation attention — among the fastest CNN options. | `tiny`, `small`, `large` |
| `fastvit` | Hybrid CNN–ViT with a very small footprint — good accuracy at tiny parameter counts. | `t8`, `t12`, `s12` |
| `shufflenet_v2` | Channel-shuffle CNN with the smallest footprint — for the most compute-limited runs. | `x0_5`, `x1_0`, `x1_5`, `x2_0` |
| `mnasnet` | MnasNet-SE CNN, NAS-tuned for the latency/accuracy trade-off. | `0_5`, `0_75`, `1_0`, `1_3` |

Copy-paste block (inside an architecture file):

```yaml
image_tower:
  model_name: efficientnet_b0   # any family_variant from the table above
  pretrained: False             # True = ImageNet weights (ignored for the custom
                                # v1_*/v2_* ultra-tiny variants, which are random-init)
```

### SSL model & projector — `ssl_model`

`SSLModel_SingleIMG` is the one available SSL model

```yaml
ssl_model:
  model_name: SSLModel_SingleIMG
  projection_dim: 256          # projector output dimension
  projector_type: small        # 'small' | 'small_norm' | 'big'
  projector_bias: True
  # only used when projector_type is 'small_norm' or 'big':
  projector_norm: bn           # 'bn' (batch norm) | 'ln' (layer norm)
  projector_activation: relu   # 'relu' | 'gelu'
```

### Fluorescence tower — `fluorescence_tower`

| `model_name` | One-line summary |
|---|---|
| `FluorescenceMLP` (default) | Fully configurable MLP — hidden dims, dropout, activation and normalization are all yours. |
| `FluorescenceMLP_Simple` | Minimal fixed-shape MLP — fewer knobs, same input/output contract. |

```yaml
fluorescence_tower:
  model_name: FluorescenceMLP
  input_dim: 13                # 3 x 5 relative-spectra grid
  output_dim: 256
  hidden_dim: [128]            # add more values to add more hidden layers
  dropout: 0.25
  activation: relu             # 'relu' | 'gelu'
  norm: bn                     # 'bn' | 'ln' | null
```

> **Stage 2 is locked to the SSL architecture.** When you fine-tune from an
> SSL bundle, the *SSL experiment's* `architecture.yaml` is merged in at
> startup and the selection above is ignored — so to train a different model
> you either re-run Stage 1 with the new architecture, or train from scratch
> (`model_initialization.pretraining.enable: false`, which uses the
> `architecture_setup` selection).

---

**Experiment bundles are self-contained.** When training starts, the trainer
saves into the experiment directory:

| File | What it is |
|---|---|
| `config.yaml` | The **fully resolved** config — stage config with the architecture merged in, plus `architecture_merged: true` and `architecture_source: <absolute path>` so you can see exactly which architecture file built the model. |
| `architecture.yaml` | A byte copy of that same architecture file (the one in `architecture_source`). |
| `*.pth` | `best_*` / `last_*` checkpoints. |
| `logs/` | Training logs, CSV metrics, plots. |

So any downstream step (validation, inference, resuming) only needs the
experiment directory: the config it reads is exactly the config that was used
to build the model — no package-folder lookups, no mismatched shapes.

Old experiment directories (created before this scheme) still work: when their
`config.yaml` has no merged architecture, the adjacent `architecture.yaml` is
merged on the fly at load time.

**Check before you train:**

```bash
bioairmet-train --config_path my_config.yaml --mode classification --show_config
```

This prints the full resolved config and a checklist of values that still look
like unfilled placeholders (`/path/to/...`, `CHANGE ME`) — without creating
any files.

When you actually start a run (without `--show_config`), the same checks run
in **fail-fast** mode: remaining placeholders, an invalid config shape
(missing sections, non-positive `epochs`, `pretrained` and `resume` both
enabled, …), or a GPU request that exceeds the visible GPUs all abort the run
before any directory or process is created.

---

## Data augmentation

Both stages configure augmentation under `data.augmentation`: one part for the
holographic **image**, one for the **fluorescence** spectrum. The same code
(`src/bioairmet/data/augmentation.py`) drives SSL and classification, and the
pipeline that actually runs is written to the training log at startup under
`--- Data Augmentation (TRAIN) ---`, so every run records exactly what was
applied.

```yaml
data:
  img_reader_type: legacy        # how raw 16-bit files become model input
  image_normalization:           # optional mean/std scaling, see below
    enable: False
    img_mean: [0.883232]
    img_std: [0.139312]
  augmentation:
    img_aug_prob: 0.5              # probability an image is augmented at all
    legacy_img_aug: True           # True = legacy pipeline (per-transform entries ignored)
    fluo_aug_prob: 0.5             # probability a spectrum is augmented
    fluo_augment_style: "covariance_style_augmentation"  # or "pca_jitter" / null
    fluo_pca_std: 0.1              # jitter std used by "pca_jitter"
    image_transforms: { ... }      # per-transform config; entries used when legacy_img_aug: false
```

> **Template defaults.** SSL ships `img_aug_prob: 0.25` (the pre-refactor
> outer gate), `fluo_aug_prob: 0.0` and `legacy_img_aug: false`;
> classification ships `img_aug_prob: 0.5`, `fluo_aug_prob: 0.5` and
> `legacy_img_aug: true`.

### Image normalization: `data.image_normalization`

Optional per-channel mean/std scaling of the image, applied **after**
augmentation (so every transform, and its `fill`, works on the [0, 1] image):

```yaml
data:
  image_normalization:
    enable: False        # True or False: the only switch
    img_mean: [0.883232] # used only when enable: True
    img_std: [0.139312]  # used only when enable: True
```

* `enable: false` (the template default) → **no-op**: `img_mean`/`img_std` are
  ignored completely, whatever their values.
* `enable: true` → every image is scaled as `(x − img_mean) / img_std` after
  augmentation, in training, validation and inference alike. If you enable it,
  use the SAME `img_mean`/`img_std` in the SSL and the classification config,
  or the pretrained weights see differently-scaled inputs.

### Image augmentation: the `legacy_img_aug` switch

`legacy_img_aug` selects which image pipeline runs. The **legacy** pipeline is
*stage-specific*: SSL and classification were trained with deliberately
different augmentation, so `legacy_img_aug: true` resolves to a different set of
transforms depending on the stage you are running. In legacy mode the
per-transform entries of the `image_transforms` block are ignored, but the
block's `fill` setting still applies (see below).

| Value | Pipeline |
|---|---|
| `true`, **SSL stage** | **Legacy (SSL)**: a jitter+blur **group**: colour jitter (brightness/contrast 0.5) and Gaussian blur (kernel 3, σ 0.1–2.0) applied *together* under one 0.5 gate, plus horizontal + vertical flips (each *p* = 0.5). |
| `true`, **classification stage** | **Legacy (classification)**: the old supervised pipeline: horizontal + vertical flips (each *p* = 0.5), an always-on rotation of ±25°, an always-on translation of up to 5% of the image size (x and y, with nearest interpolation and integer-pixel shifts), colour jitter (brightness / contrast 0.7) gated at *p* = 0.8 and Gaussian blur (kernel 3, σ 0.1–2.5) gated at *p* = 0.5. No Gaussian noise. |
| `false` | **Per-transform**: every transform is switched and gated individually from the `image_transforms` block; an entry you omit or leave `enabled: false` is simply not part of the pipeline (a missing/empty block means no augmentation at all). Identical for both stages. |

`img_aug_prob` is a master gate applied first: with probability `img_aug_prob`
an image is passed through the pipeline, otherwise it is used unedited.

### Per-transform config: `image_transforms` (entries used when `legacy_img_aug: false`)

Each transform is either a bare `true`/`false` or an object with `enabled`,
`prob`, and its own parameters. Transforms run in the table's order: geometry
first, then pixel ops:

| Transform | Parameters | Template values (SSL / classification) |
|---|---|---|
| `horizontal_flip` | — | on, *p* 0.5 / on, *p* 0.5 |
| `vertical_flip` | — | on, *p* 0.5 / on, *p* 0.5 |
| `rotation` | `angle` (degrees), `translation` `[dx, dy]` (fraction of width/height); optional `interpolation: 'bilinear'` / `'nearest'` and `translate_round: bool` | on, *p* 0.5, 180°, `[0.05, 0.05]`, nearest, rounded / off |
| `affine` | `translation` `[dx, dy]` (fraction), `scale` `[lo, hi]`; optional `interpolation` and `translate_round` as for `rotation` | off (not in the templates) |
| `color_jitter` | `brightness`, `contrast` (plus `saturation`, `hue`) | on, *p* 0.5, 0.5 / 0.5 for both stages |
| `gaussian_blur` | `kernel_size`, `sigma` `[lo, hi]` | on, *p* 0.5, kernel 3, σ 0.1–2.0 for both stages |
| `gaussian_noise` | `std` | on, *p* 0.5, 0.05 for both stages |
| `speckle_noise` | `sigma` | **off** / **off** |
| `gamma_correction` | `gamma`: `g` or `[lo, hi]` | **off** / **off** |
| `cutout` | `size` `[lo, hi]` (fraction of the width), `clear_center` (fraction of half the image), `count` (int or `[lo, hi]`, number of squares per draw) | **off** / **off** |

Every entry is **off by default in the code**: a transform joins the pipeline
only when its entry has an explicit `enabled: true`.  The newer `speckle_noise`,
`gamma_correction` and `cutout` are in particular **disabled in both
templates**; enable them deliberately, after visual tuning (see
[Visual tuning of the augmentation values](#visual-tuning-of-the-augmentation-values)
below).

The last three are pixel ops tailored to this corpus (200×200 grayscale
holograms, particle in the center):

* `speckle_noise`: multiplicative coherent-imaging speckle:
  `x · (1 + σ·N(0,1))`, clamped to [0, 1]. It textures the fringes like real
  speckle; the white background and the centered particle stay in distribution.
* `gamma_correction`: `x ** g` with `g` drawn uniformly from the range. The
  white background is a fixed point (`1.0 ** g = 1.0`), so only the particle
  intensities are re-mapped (an exposure/contrast-like variation).
* `cutout`: erases up to `count` small squares (per draw) filled with `fill`.
  Every square (even its corners) is kept outside a protected central disk of
  radius `clear_center · image/2`, and each square has its own size drawn from
  `size`, so with the template defaults nothing within 60 px of the center of
  a 200 px image is ever touched.

Two keys live at the level of the block, next to the transforms: `fill` and
`parallel_flipping_rotation`, both described in
[Fill colour, pairs and strict validation](#fill-colour-pairs-and-strict-validation).

### Fill colour, pairs and strict validation

Two block-level keys sit next to the transforms:

| Key | Default | Meaning |
|---|---|---|
| `fill` | `'border'` (templates) / `'white'` (code default) | Colour of the pixels exposed by `rotation` / `affine` in **both** legacy and custom mode, and of the pixels erased by `cutout` (custom mode). `'border'` (the template default) is the median of each image's outer 1-pixel ring, computed per image at call time, so the exposed edge matches the image's own edge colour. `'white'` (PIL 255 = tensor 1.0, the background of this corpus) and `'black'` are constants; a number in `[0, 1]` (tensor domain) is also accepted. All values are resolved per image type at call time. |
| `parallel_flipping_rotation` | `false` | **Custom mode, pairs only** (SSL, and classification with two views). `true` → both views share ONE flip/rotation/affine decision, so they stay geometrically aligned; `false` → each view gets its own geometry. Pixel-level ops (jitter, blur, noise, speckle, gamma, cutout) are always independent per view. Legacy mode has no shared pair geometry: each view runs the pipeline independently. |

The block is validated **strictly at startup**: an unknown transform name, a
probability outside `[0, 1]`, a bad `translation` / `scale` / `sigma` range or an
invalid `fill` raise (an invalid `fill` raises in legacy mode too, because the
fill applies to the legacy rotation/translation); typos *inside* a known entry
(e.g. `kernal_size`) and an entry without an `enabled` key are reported as
`[augmentation] WARNING: …` instead of being silently ignored (`enabled: false`
is the only way to switch a transform off). `summary()` (printed in the
training log) reports the mode, the geometry policy, the resolved fill and the
**effective** probabilities, i.e. `img_aug_prob × prob` of each transform, so a
transform that never fires because of the outer gate is visible.

### Reproducing the pre-refactor pipeline

Before the `image_transforms` refactor, each stage had two hard-coded
pipelines, selected by `legacy_img_aug`.  Every one of them can be reproduced
with custom mode today:

**SSL stage: old config (`legacy_img_aug: true`, what old SSL runs actually
used):** exactly the **legacy SSL pipeline** that still exists (the jitter+blur
group @ p0.5 plus both flips).  Keep `legacy_img_aug: true` to run it.

**SSL stage: old `legacy_img_aug: false` branch** ("enhanced" SSL):

| Old hard-coded transform | Exact `image_transforms` entry |
|---|---|
| `RandomHorizontalFlip(p=0.5)` | `horizontal_flip: {enabled: true, prob: 0.5}` |
| `RandomVerticalFlip(p=0.5)` | `vertical_flip: {enabled: true, prob: 0.5}` |
| `RandomApply([RandomAffine(degrees=180, translate=(0.05, 0.05), fill=255)], p=0.5)` | `rotation: {enabled: true, prob: 0.5, angle: 180, translation: [0.05, 0.05], interpolation: nearest, translate_round: true}` (plus block-level `fill: 'white'` for the old fill colour) |
| `RandomApply([ColorJitter(brightness=0.5, contrast=0.5)], p=0.5)` | `color_jitter: {enabled: true, prob: 0.5, brightness: 0.5, contrast: 0.5}` |
| `RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5)` | `gaussian_blur: {enabled: true, prob: 0.5, kernel_size: 3, sigma: [0.1, 2.0]}` |
| `AddGaussianNoise(std=0.05, p=0.5)` | `gaussian_noise: {enabled: true, prob: 0.5, std: 0.05}` |
| — (none) | `speckle_noise` / `gamma_correction` / `cutout`: `enabled: false` |

This is what the SSL template's `image_transforms` block currently contains,
so `legacy_img_aug: false` on the SSL template reproduces the old SSL
"enhanced" pipeline; the one intentional difference is the fill (the
template ships `fill: 'border'`, which only changes the colour of the small
corner regions a rotation exposes; set `fill: 'white'` to match the old runs
exactly).

**Classification stage: old `legacy_img_aug: false` branch** ("enhanced"
classification): the same entries as above but with `color_jitter:
{prob: 0.8, brightness: 0.3, contrast: 0.3}`, `gaussian_blur:
{sigma: [0.1, 1.0]}` and `gaussian_noise: {std: 0.04}`.

**Classification stage: old config (`legacy_img_aug: true`, what old
classification runs actually used):** `RandomHorizontalFlip(p=0.5)`,
`RandomVerticalFlip(p=0.5)`, `RandomRotation(degrees=25)` and
`RandomAffine(degrees=0, translate=(0.05, 0.05))` (both always on),
`RandomApply([ColorJitter(0.7, 0.7)], p=0.8)` and
`RandomApply([GaussianBlur(3, (0.1, 2.5))], p=0.5)`. The **legacy
classification pipeline** (`legacy_img_aug: true`) runs exactly these stages,
with the old parameters and torchvision's old geometry (nearest
interpolation, integer-pixel translations), and stays hard-coded.

What makes the reproduction exact:

* `interpolation: nearest` + `translate_round: true`: torchvision's
  `RandomAffine` used **nearest-neighbor interpolation** and
  **integer-pixel translations** by default (0.28 and 0.29 alike), while the
  custom pipeline defaults to bilinear/sub-pixel.  With these two keys, a
  draw of fixed angle/translation in custom mode is **bit-identical** to the
  same `RandomAffine` call (on both the 8-bit PIL and the float-tensor
  domain).
* The old code set the exposed-pixel fill to white (`fill=255` for PIL,
  `fill=1.0` for tensors), which is what `fill: 'white'` does. The templates
  now ship `fill: 'border'` instead (the median of each image's own edge
  pixels), which only changes the small corner regions a rotation exposes;
  set `fill: 'white'` to match the old runs exactly.
* The old code applied the pipeline **independently** to each view of a pair,
  which is `parallel_flipping_rotation: false` (the template default).

What still differs (statistics, not bit-for-bit):

* **RNG streams.** The old pipelines drew *everything* from the global torch
  RNG; custom mode draws the geometry (flips, angle, translation, scale) from
  a dedicated `random.Random(seed)` and only the pixel ops from the global
  torch RNG.  Same distributions, different streams: individual draws will
  not line up between old and new runs, even with equal seeds.
* **Reader domain and outer gate.** Old runs used
  `img_reader_type: 'legacy'` (16-bit → min-max → 8-bit PIL → augment →
  `ToTensor`) and the old SSL config used `img_aug_prob: 0.25`.  Both stage
  templates ship those same values, so they mirror old runs as-is; switch the
  reader to `'minmax'` or `'global'` only for new runs that want float
  tensors.
* Expect **statistical** equivalence (per-transform firing rates, output
  statistics) between the old `Compose` pipelines and the custom pipelines
  built from the entries above, not sample-by-sample identity.

### Visual tuning of the augmentation values

Before enabling or retuning any transform, see what it actually does on your
own images; no training run needed:

```
python src/bioairmet/utils/plot_augmentations.py \
    --images /path/to/a.rec_mag.png /path/to/b.rec_mag.png \
    --draws 3 --seed 0 --domain tensor
```

(Or `python -m bioairmet.utils.plot_augmentations …` with the package
installed.)  For each input image the script writes one figure with one row
per transform (that transform alone, at `prob: 1.0`) plus a final row with the
**full** pipeline from the config's `image_transforms` block — all rendered by
the actual training code path (`ImageAugmentation_Generic`, custom mode).
A good transform is clearly visible but keeps the particle centered and
recognizable and keeps the white background in distribution; adjust the
matching config entry, re-run, and compare.  Remember the effective
probability of a transform is `img_aug_prob × prob` (the reader gate is
applied first).

### Fluorescence augmentation

A spectrum is augmented by a single probability gate (`fluo_aug_prob`) and a
strategy chosen with `fluo_augment_style`:

| `fluo_augment_style` | Effect |
|---|---|
| `pca_jitter` (alias `pca`) | Gaussian jitter in PCA space, magnitude `fluo_pca_std`. |
| `covariance_style_augmentation` | Blend the spectrum toward a random reference style (the packaged unlabeled-data statistics). |
| `null` | Disabled. |

---

## Optimizers

`train.optimizer.name` selects the optimizer (case-insensitive); `lr` and
`weight_decay` apply to all of them, the remaining arguments are
per-optimizer (effective defaults in parentheses):

| `name` | One-line summary | Extra arguments |
|---|---|---|
| `adamw` | Adam with *decoupled* weight decay — the robust default for both stages, especially small-batch SSL. | `eps` (1e-8; the SSL template uses 2.5e-4 for stability), `betas` ([0.9, 0.999]) |
| `adam` | Classic Adam — same per-parameter adaptivity, but weight decay enters as plain L2 regularization. | `eps` (1e-8), `betas` ([0.9, 0.999]) |
| `sgd` | Momentum SGD — non-adaptive; usually wants a higher LR and suits restart-style schedules. | `momentum` (0.9) |

Copy-paste blocks (replace the `optimizer:` block under `train:`):

```yaml
train:
  optimizer:
    name: adamw
    lr: 0.0002
    weight_decay: 1.0e-5
    eps: 2.5e-4
    betas: [0.9, 0.999]
```

```yaml
train:
  optimizer:
    name: adam
    lr: 0.0002
    weight_decay: 1.0e-5
    eps: 2.5e-4
    betas: [0.9, 0.999]
```

```yaml
train:
  optimizer:
    name: sgd
    lr: 0.01
    weight_decay: 1.0e-4
    momentum: 0.9
```

---

## Learning-rate schedulers

`train.scheduler.name` selects the scheduler; set `name: "None"` (or `null`)
to disable scheduling entirely (constant LR). The templates ship one active
scheduler each (SSL: `OneCycleLR`, classification: `CosineAnnealingLR`) —
the copy-paste blocks below replace the whole `scheduler:` block under
`train:`.

| `name` | One-line summary | Arguments |
|---|---|---|
| `CosineAnnealingLR` | Cosine-decays the LR from its start value down to `eta_min` over `T_max` epochs — the smooth, safe schedule for fine-tuning. | `T_max`, `eta_min` |
| `OneCycleLR` | Sweeps the LR up to `max_lr` early in training, then cosine-decays it to the end — the SSL default. **Stepped once per optimizer update.** | `max_lr`, `epochs` (cycle length in epochs — keep equal to `train.epochs`), `anneal_strategy` (`cos`/`linear`), `pct_start`, `div_factor`, `final_div_factor`, optional `total_steps` |
| `StepLR` | Multiplies the LR by `gamma` every `step_size` epochs — simple step decay. | `step_size`, `gamma` |
| `ReduceLROnPlateau` | Cuts the LR by `factor` whenever the monitored metric stops improving for `patience` epochs — adaptive to your data. | `mode` (`min`), `factor`, `patience` |
| `CosineAnnealingWarmRestarts` | Cosine decay that periodically resets to the initial LR every `T_0` epochs, with the period growing by `T_mult` after each restart. | `T_0`, `T_mult`, `eta_min` |
| `CLIPCosineWarmupScheduler` | The exact OpenAI-CLIP recipe: linear warmup over `warmup_steps`, then cosine decay to `min_lr` over the whole run. **Stepped once per optimizer update**; the decay length is derived automatically (see below). | `warmup_steps`, `min_lr`, optional `total_steps` |
| `None` | Disables scheduling (constant LR). | — |

### How long the schedule is: epoch-based vs step-based

Schedulers differ in **what one `step()` means**, and the trainers follow that
distinction (`is_step_based_scheduler()` / `is_epoch_based_scheduler()` in
`utils/optimizers.py` — one source of truth used by the SSL and classification
trainers alike):

| Kind | Schedulers | Stepped |
|---|---|---|
| **step-based** | `OneCycleLR`, `CLIPCosineWarmupScheduler` | exactly once per **optimizer update** — i.e. once per `train.gradient_accumulation_steps` micro-batches, never per epoch, never before the loop |
| **epoch-based** | `CosineAnnealingLR`, `StepLR`, `CosineAnnealingWarmRestarts` | once per epoch, after the validation pass |
| **metric-driven** | `ReduceLROnPlateau` | once per epoch, fed the validation loss |

A step-based schedule has to know its length in **optimizer steps**.
`resolve_total_steps()` derives it at startup and it is written to the log:

```text
[scheduler] CLIPCosineWarmupScheduler: total_steps=1560
  [train.epochs (30) * optimizer_steps_per_epoch (52
   = ceil(steps_per_epoch 155 / gradient_accumulation_steps 3))],
  steps_per_epoch=155, epochs=30
```

* `train.scheduler.total_steps` — explicit override, an absolute **optimizer-step**
  count. Only needed when the run does not begin at step 0 (a hand-built
  mid-run resume); leave it out otherwise.
* otherwise `train.epochs × ceil(steps_per_epoch / gradient_accumulation_steps)`.

> **Why this matters:** the CLIP schedule used to multiply by `steps_per_epoch`
> twice, so the cosine leg was stretched by ~100× and the LR barely left warmup
> over an entire run; with gradient accumulation, schedules could equally end
> long before the last epoch. The decay now finishes on the **last optimizer
> update of the last epoch** whatever `gradient_accumulation_steps` is, and it
> re-derives itself when you change `batch_size`, the dataset size or the
> accumulation factor. For the epoch-wise schedulers you still keep `T_max`
> (or `scheduler.epochs`) equal to `train.epochs`.

> **Gradient accumulation and the tail of an epoch.** Both trainers apply a final
> update for the *partial* last window of an epoch (normalised by its real size)
> instead of dropping those micro-batches, so the number of updates per epoch is
> exactly `ceil(steps_per_epoch / gradient_accumulation_steps)` — the number the
> schedule is sized from. Before that, the classification trainer silently
> skipped the tail: the last `steps_per_epoch mod A` batches of every epoch
> contributed nothing to the weights, and a step-based schedule stopped short of
> `min_lr`.

Copy-paste blocks:

```yaml
train:
  scheduler:
    name: CosineAnnealingLR
    T_max: 50          # = number of epochs
    eta_min: 0.0
```

```yaml
train:
  scheduler:
    name: OneCycleLR
    max_lr: 0.001      # upper LR boundary
    epochs: 100        # overridden automatically at startup
    anneal_strategy: cos   # 'cos' | 'linear'
    pct_start: 0.1       # fraction of the cycle spent ramping the LR up
    div_factor: 30       # initial LR = max_lr / div_factor
    final_div_factor: 1000000  # minimum LR = initial_LR / final_div_factor
```

```yaml
train:
  scheduler:
    name: StepLR
    step_size: 10      # shrink every 10 epochs
    gamma: 0.1
```

```yaml
train:
  scheduler:
    name: ReduceLROnPlateau
    mode: min          # monitor a lower-is-better metric (validation loss)
    factor: 0.1
    patience: 10
```

```yaml
train:
  scheduler:
    name: CosineAnnealingWarmRestarts
    T_0: 10            # first restart period (epochs)
    T_mult: 1          # multiply the period by this after each restart
    eta_min: 0.0
```

```yaml
train:
  scheduler:
    name: CLIPCosineWarmupScheduler
    warmup_steps: 500  # linear warmup length (in optimizer steps)
    min_lr: 5.0e-6
```

```yaml
train:
  scheduler:
    name: "None"       # no scheduling — constant LR
```

## Training throughput

Small backbones (≈ 100 k parameters) spend most of their step time in kernel
launches and memory traffic, not in arithmetic, so the layout and algorithm
switches matter more than the model size. Three optional switches are read from
`train:` (`_configure_performance()` in `trainers/base_trainer_v2.py`), all **off
in code** and therefore opt-in per experiment:

| Key | Default | Effect |
|---|---|---|
| `cudnn_benchmark` | `False` | Lets cuDNN auto-tune convolution algorithms per input shape. Worth it because every step here uses the same fixed `data.image_size`; the price is a slower first pass (and re-tuning if shapes vary, e.g. multi-resolution). |
| `tf32` | `False` | Enables TF32 for fp32 matmul/conv on Ampere and newer. Consistent with `mixed_precision: True`; turn it off when you need the full fp32 mantissa. |
| `channels_last` | `False` | Converts the models and the image batches to NHWC (`memory_format=torch.channels_last`). Biggest single win for these conv nets. |

Measured during development on this project's backbones (batch 64, 200×200,
forward **and** backward, RTX PRO 6000 Blackwell):

| Variant | params | BN layers | fp32 NCHW | fp32 NHWC | AMP NCHW | AMP NHWC |
|---|---|---|---|---|---|---|
| EfficientNet `v1_100k` | 78,680 | 12 | 13.8 k | 18.4 k | 14.9 k | **21.3 k** img/s |
| EfficientNet `v2_100k` | 91,140 | 9 | 17.2 k | 21.2 k | 21.3 k | **36.3 k** img/s |
| EfficientNet `v2_250k` | 207,784 | 11 | 13.8 k | 16.2 k | 18.7 k | 27.7 k img/s |
| MobileNetV3 `tiny` | 100,296 | 28 | 13.6 k | 16.8 k | 12.0 k | 15.2 k img/s |
| ShuffleNetV2 `x0_5` | 341,360 | 56 | 10.5 k | 11.9 k | 9.0 k | 10.7 k img/s |

Two conclusions worth acting on: NHWC pays off everywhere (+20 … +70 %), and
**many small BatchNorm layers** (MobileNet/ShuffleNet) make AMP *slower* than
fp32 — the custom EfficientNet variants are both smaller and faster than the
off-the-shelf tiny models, so they remain the default. Re-measure on your own
hardware before changing the defaults.

A fourth switch lives in the model rather than the loop:
`architecture_setup.classification_model.fuse_frozen_image_views` (default
`True`). When the image encoder is completely frozen and in `eval()` mode, the
two views of a pair are encoded in **one** concatenated forward pass instead of
two, which roughly halves the encoder's launch overhead. It is bit-for-bit
identical to two separate calls and is skipped automatically whenever the
encoder is trainable or its BatchNorm statistics update, because then the two
views must not share a batch.

## Losses

SSL (`train.loss.name`): `ContrastiveLoss` — plus the DDP loss behaviour
flags `emulate_old_loss`, `local_loss`, `gather_with_grad` and the
temperature settings (`temperature`, `learnable_temp`).

Classification (`train.loss.name`):

| `name` | Parameters |
|---|---|
| `CrossEntropyLoss` | `label_smoothing` |
| `FocalLoss_gt_Corrected` | `gamma`, `alpha` |

`FocalLoss_gt_Corrected` uses the standard RetinaNet form
(`alpha_t · (1 − p_t)^gamma · CE`, unit-tested against a reference
implementation) and the `alpha` weights handle class imbalance:

- `"auto"` (what the classification templates ship with) — the training
  worker reads the class counts of the **training** set and derives
  inverse-frequency weights (`alpha_c ∝ 1/count_c`, normalised to sum to 1).
  Classes absent from the training set get weight 0 and are reported in the
  log. The computed weights are written into the experiment's `config.yaml`,
  so standalone validation of the experiment reuses the same weights.
- an explicit list of `num_classes` values (or a scalar) — used verbatim.

---

## Class coverage — `classification_model.num_classes`

The classifier head has exactly `architecture_setup.classification_model.num_classes`
outputs, so labels must live in `[0, num_classes)`. The configured
`num_classes` is **authoritative** — the training worker checks the actual
dataset labels against it:

- **Out-of-range labels are excluded automatically.** If the data contains
  labels ≥ `num_classes` (e.g. 37 classes in the file but `num_classes: 35`),
  those samples are dropped from the train and validation datasets up front
  (deterministic on every rank) instead of crashing the loss with an
  index-out-of-bounds error. A startup warning lists every excluded class
  with its sample count and how to bring it back (raise `num_classes` and/or
  adjust the category map):

  ```
  Class-coverage: the following classes are excluded from the training
  dataset and will be skipped during training: class 35: 214 sample(s) —
  label is outside the configured range (label >= num_classes=35);
  class 36: 98 sample(s) — label is outside the configured range
  (label >= num_classes=35). To include them, raise
  architecture_setup.classification_model.num_classes and/or adjust the
  category map.
  ```

- **Missing classes are warned about.** If `num_classes` is larger than the
  classes present in the training set, a warning lists the empty classes —
  their heads will remain untrained (and get focal `alpha = 0`):

  ```
  Class-coverage: num_classes=37 but the training set has no samples for
  class(es) [35, 36] — their classifier heads will remain untrained
  (focal alpha = 0). Reduce num_classes or provide samples for them.
  ```

- **Category map mismatches keep the configured `num_classes` (single
  warning).** When `data.cat_map_path` is set in the config and the map
  contains a label ≥ `num_classes` (e.g. the default `num_classes: 35`
  against the shipped 36-class `cats_dict.txt`, whose largest label is 35),
  the configured `num_classes` stays **authoritative**: the classifier head
  keeps `num_classes` outputs and samples with out-of-range labels are
  excluded from the datasets (see above). The training worker prints
  **one** warning (rank 0 only, also recorded in `training.log`) listing the
  affected labels:

  ```
  num_classes=35 but the category map (cats_dict.txt) contains label(s) [35]
  — the configured num_classes is kept and samples with those labels are
  excluded from the datasets (classifier head: 35 outputs). To train on
  them, raise architecture_setup.classification_model.num_classes.
  ```

  The check runs on the **training** path only — at inference the head size
  comes from the checkpoint and stays authoritative.

- **The classes actually present in the training data are reported.** At
  startup an INFO-level summary states how many of the configured classes
  have samples in the training set — i.e. which classifier heads will be
  trained at all (empty ones stay untrained and get focal `alpha = 0`):

  ```
  Class-coverage: training data contains 34/35 configured classes
  (12345 samples); classes with NO samples: [2, 17]
  ```

The same exclusion is visible in the focal-loss alpha derivation: `alpha` is
computed from the *post-exclusion* training-set class counts.

---

## Freezing & fine-tuning

How much of the pretrained model trains in Stage 2 is controlled by the
`train.fine_tuning` block of the classification config. The semantics are
enforced **consistently** by the model, the optimizer grouping, and the
training loop — all three read the same config keys.

| Config key (`train.fine_tuning`) | Default | Effect |
|---|---|---|
| `unfreeze_image_encoder` | `False` | `True` → **full** end-to-end fine-tuning of the image encoder. |
| `unfreeze_fluorescence_encoder` | `False` | `True` → **full** end-to-end fine-tuning of the fluorescence encoder. |
| `unfreeze_last_img_layers` | `False` | `N ≥ 1` → only the **last N layer units** of the image encoder train (used when `unfreeze_image_encoder` is `False`). A *layer unit* is one block of the backbone's `features` container (EfficientNet/MobileNet: `1` = the final conv block, ≈ 14 % of `v2_100k`; ShuffleNet: the last stage); parameter-free modules such as the pooling ride along with the block. `N` ≥ number of blocks unfreezes the whole encoder. |
| `unfreeze_last_fl_layers` | `False` | `N ≥ 1` → same for the fluorescence encoder, where one `Linear` is a unit and the `BatchNorm`/activation behind it ride along. |
| `update_batchnorm_stats_img_encoder` | `False` | `True` → puts the image encoder's BN in `train()` mode: running stats are **re-estimated from each batch (AdaBN)**, which also normalises the training forward with **batch** statistics and **overwrites** the pretrained running stats. No parameter becomes trainable. |
| `update_batchnorm_stats_fl_encoder` | `False` | Same, for the fluorescence encoder. |
| `train_batchnorm_affine_img_encoder` | `False` | `True` → the image encoder's BN **affine** `weight`/`bias` become trainable + enter the optimizer (running stats stay fixed unless the stats flag is also on). |
| `train_batchnorm_affine_fl_encoder` | `False` | Same, for the fluorescence encoder. |
| `unfreeze_batchnorm_img_encoder` | `False` | **Shortcut** = `update_batchnorm_stats_img_encoder` **+** `train_batchnorm_affine_img_encoder`. |
| `unfreeze_batchnorm_fl_encoder` | `False` | **Shortcut** for the fluorescence encoder. |
| `image_encoder_lr_multiplier` | `1.0` | LR multiplier for the (unfrozen) image-encoder group. |
| `fluorescence_encoder_lr_multiplier` | `1.0` | LR multiplier for the (unfrozen) fluorescence-encoder group. |
| `classifier_lr_multiplier` | `1.0` | LR multiplier for the classifier head (always trainable). |
| `bn_bias_no_decay` | `False` | Exclude BN/bias parameters (ndim < 2) from weight decay. |

### What actually trains, verified at startup

Because these flags interact, every V2 run prints a **BatchNorm / freeze audit**
at startup and a one-line summary per epoch (right after
`model.train()` + `enforce_frozen_modes()`, so the log proves the policy is in
force), and warns when the config, the live model and the optimizer disagree:

| Prefix | Raised when |
|---|---|
| `[BN-DIVERGENCE]` | a `*_batchnorm_*` switch has no effect on the live model **while that switch can have an effect** (see the two phases below), a BN layer has `track_running_stats=False` (normalisation then silently depends on the batch), or `unfreeze_last_*_layers` selected no parameters at all. Checked again at the first epoch, when the policy is guaranteed to be applied. |
| `[FREEZE-DIVERGENCE]` | parameters with `requires_grad=True` are missing from the optimizer's parameter groups — they would never be updated. |
| `[BN-policy] … PENDING` | **informational, not a warning.** Printed by the startup audit for `update_batchnorm_stats_*: true`: `_apply_fine_tuning_policy()` only *stores* the intent, and no BatchNorm can touch its running statistics until `train_epoch()` calls `model.train()` + `enforce_frozen_modes()`. Read the epoch line to confirm. |
| `[BN-DDP]` | `update_batchnorm_stats_*: true` is actually updating BN layers on a **DDP-wrapped** model: DDP synchronises gradients, not buffers, so each rank estimates the running statistics from its own shard and only rank 0's end up in the checkpoint. Convert with `nn.SyncBatchNorm.convert_sync_batchnorm()` (or update the statistics in one process) when that estimate matters. |

**The two phases matter for reading the log.** The startup audit runs *before* the
epoch loop — its header says `PRE-POLICY state` — and in that state the encoders
sit in `eval()` (put there by the freeze policy at build time), so
`updating 0/49` is the *expected* reading for `update_batchnorm_stats_*: true`,
not evidence that the flag was ignored. The epoch line is the authoritative one:

```
[BN-state] epoch 0 after model.train()/enforce_frozen_modes:
  img_encoder[BN 49: eval 0, train 49; updating 49; affine-trainable 0/98]; ...
```

`updating N/M` counts BN layers that are in `train()` mode **and** track running
statistics — the only combination in which buffers change. `updating 49/49` proves
the flag is in force; `updating 0/49` while the model is training means it is not
(and then `[BN-DIVERGENCE]` is printed, once per run).

A buffer-level cross-check that needs no log at all: `num_batches_tracked`
increments once per train-mode BatchNorm forward, so comparing two saved
checkpoints of the same encoder tells you definitively whether the statistics
moved.

### Model initialization modes & learning-rate guidance

The classification model can be started in one of three ways (see
`model_initialization`):

| Mode | How it's set | What trains | Encoder LR multiplier | Head LR multiplier |
|---|---|---|---|---|
| **Fine-tune from SSL** | `pretraining.enable: true` + `experiment_path` | head (+ unfrozen encoders) | `~0.001` (protect the SSL weights) | `~1.0` |
| **Train from scratch** | `pretraining.enable: false` | head + **both encoders (auto-unfrozen)** | `~1.0` (random init needs a full LR) | `~1.0` |
| **Resume** | `resume.enable: true` + `checkpoint_path` | whatever the resumed config specified | as in the resumed config | as in the resumed config |

> **Why it matters:** the fine-tuning defaults are tuned for *pretrained*
> encoders — a small encoder LR multiplier (e.g. `0.001`) gently nudges good
> weights. For **train-from-scratch** there are no good weights to protect: if
> you keep the `0.001` encoder multiplier, the randomly-initialised encoders
> barely learn and the head is effectively training on a frozen random encoder.
> When `pretraining.enable: false` (and resume is off), the builder
> automatically unfreezes both encoders (so training *can* happen) and prints
> a warning reminding you to set `image_encoder_lr_multiplier` /
> `fluorescence_encoder_lr_multiplier` to `~1.0`. This is a **hard
> guarantee, not a config convenience**: in train-from-scratch mode *every*
> model parameter is trainable (`requires_grad=True`) and the same unfreeze
> flags feed the optimizer's parameter groups, so the encoders are actually
> optimised from scratch — the grouping has an integrity check that would
> crash the run if a trainable parameter were ever missing from the
> optimizer (covered by `test_train_from_scratch_optimizer_covers_all_parameters`).

**Which checkpoint did I actually start from?** The builder logs the provenance of
the initialisation file:

```
[V2 Builder] Initialising from checkpoint .../last_ssl_model.pth | modified
2026-09-08 14:09:28 (3 s ago), 51.3 MB, source epoch=1, source BN
num_batches_tracked=9834
```

`weights: last` is **overwritten at the end of every epoch** of the source
experiment. Starting a classification run while that experiment is still training
therefore picks up whatever epoch happened to be flushed at that second, and a
second run started a minute later initialises from *different* weights — an A/B
comparison of the two then mixes your setting under test with a different
initialisation and says nothing about either. The builder warns
(`[V2 Builder][STALE-INIT]`) when the file is younger than two minutes. Safe
practice:

- wait for the source run to finish, **or**
- copy its checkpoint into a private directory and point
  `model_initialization.pretraining.experiment_path` at the copy, **or**
- use `weights: best` (rewritten only when the metric improves — still not
  immutable, so prefer the copy for published comparisons).

The `source BN num_batches_tracked` value is a fingerprint of how much training
the file contains, so two runs can be checked afterwards: if it differs between
their logs, they did not start from the same weights (frozen encoders keep those
weights byte-identical all run, which makes the check exact).


and no `unfreeze_batchnorm_*` flag) is not just weight-frozen:

- `requires_grad=False` on all of its parameters,
- held in `eval()` mode on every forward step (dropout off),
- its BatchNorm layers stay in `eval()` with `track_running_stats=True`, so
  they normalise with the **pretrained running statistics** (deterministic at
  training *and* inference) and those statistics can never drift (eval() mode
  blocks updates).

When an encoder (or a tail of it) **is** unfrozen, its BatchNorm layers are put
in train mode, so statistics update with the new fine-tuning distribution.

**BatchNorm: stats vs. affine.** A BatchNorm layer has two kinds of state,
which the fine-tuning config controls **independently** (encoder-wide, on top of
the full / last-N unfreeze above):

- **Running stats** (`running_mean`, `running_var`) are *buffers*, not
  parameters. They are never trained and never appear in the optimizer; they are
  updated by a moving average during the **forward pass**, and only while the BN
  layer is in `train()` mode. Enable with
  `update_batchnorm_stats_{img,fl}_encoder` → the statistics adapt to the new
  distribution **without a single extra trainable parameter** (a.k.a. AdaBN).
- **Affine params** (`weight`, `bias`) *are* trainable parameters. Enable with
  `train_batchnorm_affine_{img,fl}_encoder` → they get gradients and are added
  to the optimizer under that encoder's LR multiplier (`bn_bias_no_decay` still
  applies).

`unfreeze_batchnorm_{img,fl}_encoder` is a convenience **"both"** shortcut
(running stats update *and* affine trainable). BN layers always keep
`track_running_stats=True` (frozen encoders included), so normalisation uses
running statistics and inference is deterministic in every mode. The four
combinations per encoder:

| stats | affine | BN mode (training) | Effect |
|---|---|---|---|
| ❌ | ❌ | `eval()` | normalise with **pretrained running stats**; nothing trains (the safe default, matches the SSL features) |
| ✅ | ❌ | `train()` | running stats **re-estimated/overwritten** on the new data; training forward uses **batch** stats; no trainable params (AdaBN) |
| ❌ | ✅ | `eval()` | affine `weight`/`bias` train; running stats fixed (pretrained) |
| ✅ | ✅ | `train()` | both (`unfreeze_batchnorm_*`) |

> **Which should I use?** For a *frozen* encoder the ❌/❌ default (use the
> pretrained running stats) is almost always best: recomputing the statistics on
> a small fine-tuning set (✅/❌) discards the SSL-pretrained stats, makes the
> forward pass batch-size dependent, and introduces a train↔eval mismatch that
> usually *hurts* validation accuracy. Reach for `update_batchnorm_stats_*` only
> when the target distribution genuinely differs and you have enough data.

The shipped classification template leaves all six BatchNorm flags `False`, so a
frozen encoder uses its pretrained running statistics; turn on a granular flag
(only stats, only affine) or the `unfreeze_batchnorm_*` shortcut to adapt BN.

**"Last N layers" granularity** depends on the backbone — a *layer unit* is a
parameter-bearing child module of the encoder (single-container wrappers like
`encoder.encoder.*` are unwrapped automatically):

| Encoder | N = 1 trains | N = 2 trains |
|---|---|---|
| EfficientNet (`img_encoder.model.*`) | head (`classifier`) + pool | `conv5` + pool + head |
| ShuffleNet | `conv5` | `conv4` + `conv5` |
| Fluorescence MLP (`encoder.*`) | final `Linear` | `Linear → Act → Dropout → Linear` tail |

If `N` exceeds the number of available layers it is clamped to the whole
encoder. If both the full-unfreeze flag and the last-N key are set for the
same encoder, the **full** unfreeze wins.

> **Fail-fast:** if the classification config yields *no* trainable group at
> all (e.g. the model exposes no `classifier` head), training aborts with a
> clear error instead of silently training randomly-initialised parameters.

**Resuming / reusing:** `model_initialization.resume.enable: true`
(+ `checkpoint_path`) restarts from a previous Stage 2 checkpoint;
`pretraining.weights: best|last` selects which SSL checkpoint the encoders
are initialised from.

