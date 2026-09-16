# Custom Encoders (Images & Fluorescence)

Plug your own architectures into both training stages — no changes to the
training code. Two ready-to-modify templates ship with the package in
[`src/bioairmet/models/custom_models.py`](../src/bioairmet/models/custom_models.py):

| Template | Domain | Select in config with |
|---|---|---|
| `SimpleCnn_grayscale` | `(B, 1, H, W)` holographic images | `image_tower.model_name: mycnn_small` or `mycnn_wide` |
| `FluorescenceCNN` | `(B, F)` fluorescence spectra | `fluorescence_tower.model_name: FluorescenceCNN` |

Because the package is installed editable (`pip install -e .`), you can edit
`custom_models.py` directly and the change is live on the next run — the CLI
launchers pick it up without extra wiring.

**Related:** [Model building & loading](model_loading.md) ·
[Configuration guide](configuration.md) · [Training guide](training.md)

---

## The encoder contract

Everything the builders, trainers, freezing logic and fusion layer need from a
custom encoder is:

1. **An `nn.Module`** whose `forward(x)` returns a `(B, D)` float tensor:
   - image tower: `x` is `(B, 1, H, W)` (grayscale holographic images);
   - fluorescence tower: `x` is `(B, input_dim)` (the spectrum).
   `D` must be fixed for a given instance.
2. **`get_output_dim() -> int`** returning `D`. `SSLModel_SingleIMG` and
   `HoloClassifierV2` size their projection/fusion layers from this — it must
   match what `forward` actually produces.
3. **Nothing else.** Freezing, `unfreeze_last_*_layers`, the
   `update_batchnorm_stats_*` / `train_batchnorm_affine_*` switches and the
   frozen-view fusion all walk the module tree generically, so plain PyTorch
   modules (Conv/Linear/BatchNorm/…) just work.

Two conventions worth keeping:

- **Checkpoint stability.** Class names and module paths are stored in every
  saved state dict. Renaming a custom class — or moving it to another file —
  makes old checkpoints unloadable. Pick a name, keep it.
- **BatchNorm honesty.** A custom encoder with BatchNorm layers participates in
  the full BN audit (frozen `eval()` by default in Stage 2). That is usually
  what you want; if your module has no BN, the switches simply do nothing.

---

## Walkthrough 1 — image backbone (`SimpleCnn_grayscale`)

The template is a plain Conv → BatchNorm → ReLU stack with global average
pooling. The interesting parts are the **builder** and the **registration** at
the bottom of `custom_models.py`:

```python
class SimpleCnn_grayscale(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128), in_channels=1):
        ...                                   # your architecture here
    def forward(self, x):                     # (B, 1, H, W) -> (B, D)
        return self.pool(self.features(x)).flatten(1)
    def get_output_dim(self):                 # fusion reads this
        return self._output_dim


def _build_mycnn(variant, pretrained, in_channels):
    """Registry builder: (variant, pretrained, in_channels) -> module."""
    if pretrained:
        logger.warning("No pretrained weights available; random init.")
    return SimpleCnn_grayscale(channels=_MYCNN_VARIANTS[variant],
                               in_channels=in_channels)


register_backbone(
    "mycnn",                                  # -> model_name: mycnn_<variant>
    build=_build_mycnn,
    variants=("small", "wide"),
    default_variant="small",
    name_prefixes=("mycnn_",),
)
```

Then select it — in your stage config or architecture YAML:

```yaml
architecture_setup:
  image_tower:
    model_name: mycnn_small     # <family>_<variant>
    pretrained: False
```

That is all: `run_ssl_training.sh`, `run_classification_training.sh`,
validation and inference all resolve the name through the registry.

### `register_backbone` reference

```python
from bioairmet.models import register_backbone

register_backbone(
    family,                # registry key; config names look like f"{family}_{variant}"
    build=...,             # (variant, pretrained, in_channels) -> nn.Module, OR the
                           # recipe pair select= + adapt= (the built-in style)
    get_output_dim=...,    # optional (model) -> int if your module has no method
    variants=(...),        # accepted variant names
    default_variant=...,   # used when the wrapper gets no variant
    name_prefixes=(...),   # prefixes stripped to recover the variant
    override=False,        # replace an existing family
)
```

Registering the same family twice raises `ValueError` unless you pass
`override=True` — accidental shadowing of a built-in (e.g. `"efficientnet"`)
fails loudly instead of silently swapping models.

---

## Walkthrough 2 — fluorescence encoder (`FluorescenceCNN`)

The fluorescence counterpart registers under the exact tower name:

```python
class FluorescenceCNN(nn.Module):
    def __init__(self, input_dim, output_dim=256, channels=(32, 64),
                 kernel_size=5, dropout=0.1):
        ...                                   # Conv1d blocks + global pool
    def forward(self, x):                     # (B, F) -> (B, output_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)                # (B, F) -> (B, 1, F)
        return self.head(self.features(x))
    def get_output_dim(self):
        return self.output_dim


def _build_fluorescence_cnn(fluo):
    """Builder receives the whole fluorescence_tower config section."""
    return FluorescenceCNN(
        input_dim=int(fluo.input_dim),
        output_dim=int(fluo.get("output_dim", 256)),
        channels=tuple(fluo.get("channels", (32, 64))),
        kernel_size=int(fluo.get("kernel_size", 5)),
        dropout=float(fluo.get("dropout", 0.1)),
    )


register_fluorescence_encoder("FluorescenceCNN", _build_fluorescence_cnn)
```

```yaml
architecture_setup:
  fluorescence_tower:
    model_name: FluorescenceCNN
    input_dim: 13          # must match your HDF5 spectrum length
    output_dim: 256
    channels: [32, 64]     # optional — your own builder can read any keys
```

Unlike the image tower (where the config name splits into family + variant),
the fluorescence builder receives the **entire `fluorescence_tower` section**,
so you can add your own keys freely — unknown keys are simply ignored unless
your builder reads them.

---

## 30-second sanity check

Before launching a run, build both models with your encoder and push one
dummy batch through:

```python
import torch
from easydict import EasyDict
from bioairmet.models import build_ssl_model_from_config, HoloClassifierV2
from bioairmet.utils import parse_config
from bioairmet.utils.config_parser import load_and_merge_architecture_config

cfg = parse_config("src/bioairmet/config/ssl/SSL_config_general.yaml")
cfg.architecture_setup = EasyDict(cfg.architecture_setup or {})
cfg.architecture_setup.type = "ssl"
cfg, _ = load_and_merge_architecture_config(cfg)
cfg.architecture_setup.image_tower.model_name = "mycnn_small"
cfg.architecture_setup.fluorescence_tower.model_name = "FluorescenceCNN"

model = build_ssl_model_from_config(cfg).eval()
out = model(torch.randn(2, 1, 200, 200), torch.randn(2, 13))
print(sorted(out.keys()))          # image/fl embeddings + logit_scale
```

A successful build logs `Registered backbone family 'mycnn' …` /
`Registered fluorescence encoder 'FluorescenceCNN'` at import time.

---

## Using your own module instead of `custom_models.py`

`custom_models.py` is imported automatically by `bioairmet.models`, which is
the simplest wiring. If you keep your encoders elsewhere (your own package, a
project directory), registration is just an import side effect — import your
module **before** the model is built:

```python
import my_project.encoders            # runs register_backbone / register_..._encoder

model = build_model_from_config(config)   # now resolves your names
```

For the shipped CLI entry points (`bioairmet-train` etc.) the automatic import
of `custom_models.py` is the reliable hook — add a one-line import of your own
module at the bottom of that file if you do not want to edit it directly.

---

## Interaction with the training features

| Feature | Behaviour with a custom encoder |
|---|---|
| Stage 1 (SSL) | The encoder trains contrastively like any built-in one. |
| Stage 2 freezing | Encoders are frozen (`requires_grad=False`, `eval()`) by default; `train_batchnorm_affine_*` etc. apply as documented. |
| `unfreeze_last_img_layers` / `..._fl_layers` | Wrapper modules are flattened to real layer granularity, so "last N layers" selects N blocks/conv layers of your module, not one monolith. |
| Frozen-view fusion (Stage 2) | Automatic: one concatenated forward when your image encoder is frozen and fully in `eval()`; disabled automatically the moment any part of it trains. |
| `channels_last` switch | Applies to your module too — Conv2d stacks benefit; nothing special to do. |
| DDP | No special handling needed (plain module). If `update_batchnorm_stats_*: true` on DDP, the standard `[BN-DDP]` warning applies (per-rank running statistics). |

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Unknown image encoder name: 'mycnn_small'` | Your module was never imported (registration is import-time). Use `custom_models.py` or import it before building. |
| `has no get_output_dim()` | Add the method, or pass `get_output_dim=` to `register_backbone`. |
| Shape error inside the fusion layer | `get_output_dim()` disagrees with `forward()` — make them match. |
| `is already registered` | Two registrations of the same family/name; rename, or pass `override=True` intentionally. |
| Loading an old checkpoint fails (`Invalid module name`) | The custom class was renamed/moved after training. Restore the original class + module path (or re-register under the old name). |
| `pretrained: True` has no effect | Expected — templates are random-init. Wire up `timm`/`torchvision` weights inside your own builder if you need them. |