# Model Building & Loading Guide

How to build the BioAirMet models (SSL and classification) from a config, load
checkpoint weights, and use the Python API.

**Related:** [Configuration guide](configuration.md) · [Training guide](training.md) · [Validation & inference guide](inference.md)

---

## The two model types

BioAirMet has one model per training stage, selected by `architecture_setup.type`
in the config:

| `architecture_setup.type` | Model class | Builder |
| ------------------------- | ----------- | ------- |
| `ssl`            | `SSLModel_SingleIMG` | `build_ssl_model_from_config(config)` |
| `classification` | `HoloClassifierV2`     | `build_classification_model_from_config_v2(config)` |

`build_model_from_config(config)` is the top-level dispatcher that calls the
right builder based on `architecture_setup.type`.

## Loading a model: `load_model`

`load_model` is the single documented entry point for building a model and
(optionally) loading checkpoint weights. It returns the model on the requested
device, in `eval()` mode:

```python
from bioairmet import load_model
from bioairmet.utils import parse_config

config = parse_config("src/bioairmet/config/classification/config_general.yaml")

# Fresh (random) weights
model = load_model(config)

# Load checkpoint weights (strict=False, so encoder-only checkpoints are fine)
model = load_model(config, "experiments/cls/last_model.pth", device="cpu")

# On a GPU
model = load_model(config, "experiments/cls/best_model.pth", device="cuda:0")
```

### Signature

```python
load_model(config, checkpoint_path: str | None = None, device: str = "cpu") -> torch.nn.Module
```

- **config** — the merged configuration object (`parse_config` returns one).
- **checkpoint_path** — optional path to a checkpoint. When given, weights are
  loaded with `strict=False` so an encoder-only checkpoint can be loaded into a
  full model. Missing/unexpected keys are logged.
- **device** — target device (default `"cpu"`).

`load_model` raises `FileNotFoundError` if `checkpoint_path` is given but does
not exist.

## Checkpoint format

A checkpoint is a dictionary (or a plain state dict). `load_model` and the
builders accept both:

```python
{"model_state_dict": {...}, "epoch": 12, ...}   # full checkpoint (what the trainers save)
{...}                                           # plain state dict
```

If the weights were saved under a `DistributedDataParallel` wrapper, a leading
`module.` prefix on every key is stripped automatically.

## Building without `load_model`

The lower-level builders are available directly when you need more control
(e.g. building only one encoder, or loading weights yourself):

```python
from bioairmet.models import (
    build_model_from_config,                    # dispatch on architecture_setup.type
    build_ssl_model_from_config,                # SSLModel_SingleIMG
    build_classification_model_from_config_v2,  # HoloClassifierV2
    load_weights_into,                          # load a checkpoint into an existing model
)

model = build_model_from_config(config)
load_weights_into(model, "path/to/checkpoint.pth", strict=False)
```

`load_weights_into` reports the number of missing/unexpected keys (expected when
loading an encoder-only SSL checkpoint into the full classification model) and
returns the `load_state_dict` result.

## SSL pretraining -> classification

When fine-tuning a classification head on top of an SSL-pretrained model, the
`model_initialization` block (see
[Configuration guide](configuration.md#model-initialization-modes--learning-rate-guidance))
pins the architecture to the SSL experiment and loads the SSL encoders into the
classification model. `build_classification_model_from_config_v2` (and therefore
`load_model`) handles this automatically from the config.

## Classification head configuration

`classifier_type` selects the head shape and `classifier_layernorm` controls the
LayerNorm on the concatenated feature input for **both** the `one_layer` and
`two_layers` heads — the flag is authoritative, so it always matches the built
architecture (see [Configuration guide](configuration.md)).

## Public model API

```python
from bioairmet.models import (
    build_model_from_config,                  # config -> SSL or classification model
    load_model,                               # config (+ optional checkpoint) -> model
    build_ssl_model_from_config,
    build_classification_model_from_config_v2,
    load_weights_into,                        # load a checkpoint into an existing model
    load_state_dict,                          # load + prefix-strip a checkpoint file
    strip_ddp_prefix,                         # remove a leading "module." from keys
)
```