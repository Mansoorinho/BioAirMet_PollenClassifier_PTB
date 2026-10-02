# ONNX Export Guide

How to export a trained V2 classification experiment to a self-contained
ONNX model and consume it with ONNX Runtime.

**Related:** [Inference guide](inference.md) · [Model building & loading guide](model_loading.md)

---

## What gets exported

`bioairmet.utils.export_onnx.export_experiment()` (the default CLI mode)
exports BOTH standard checkpoints of an experiment —
`best_classification_model.pth` and `last_classification_model.pth` — each to
`<experiment>/onnx/<best|last>/model.onnx`. For each checkpoint,
`bioairmet.utils.export_onnx.convert_to_onnx()` loads it through the same
validated resume path used for training, wraps the model so it takes ONE flat
input, exports with `torch.onnx.export` (legacy exporter, `dynamo=False`;
falls back to the dynamo exporter on torch builds where the legacy one is
unavailable), validates the graph with `onnx.checker`, writes metadata, and —
by default — verifies numerical parity against the PyTorch model on real
validation samples. It also writes a `README.md` next to the `.onnx` file
describing the model version, training context, feature definitions, and the
required image preprocessing (including the training-time image reader), so
the model package is self-describing per the generic ONNX contract.
Everything about the model — num_classes, image_size, image reader, class
names, verification data — is resolved from the FULL experiment directory
(`config.yaml` + `architecture.yaml` + `cats_dict.txt`); nothing is passed by
hand.

| Property | Value |
|---|---|
| Opset | 17 (override with `--opset`) |
| Batch axis | dynamic (`batch_size`) |
| Precision | float32, CPU-safe (verified with ONNX Runtime `CPUExecutionProvider`) |
| Metadata | `sws.class_names`, `sws.input_features` (JSON arrays) + companion `sws.input_layout`, `sws.image_reader` (JSON) |

### Input

```text
name:  X
type:  float32
shape: [batch, 80013]      # = 13 + 2 * (200 x 200)
```

| Slices of each row | Content |
|---|---|
| `[0:13]` | 13-dim fluorescence spectrum — the 15-dim raw spectrum with the always-zero channels at indices 5 and 10 dropped (same reduction the datasets apply) |
| `[13:40013]` | image view A, 200x200, row-major, float in `[0, 1]` |
| `[40013:80013]` | image view B, 200x200, row-major, float in `[0, 1]` |

The exact layout (offsets, dtypes, notes) is written into the model as the
companion `sws.input_layout` metadata property, and `sws.input_features`
holds the 80013 per-column feature names in input-column order
(`fl_raw_0..4`, `fl_raw_6..9`, `fl_raw_11..14`, then
`imgA_r0000_c0000` … `imgB_r0199_c0199`), as required by the generic ONNX
contract.

### Outputs (in this order)

```text
label         int64    [batch]      argmax class index (0..C-1)
probabilities float32  [batch, C]   softmax probabilities
```

`sws.class_names` is a JSON list of the C class names in index order (from
the experiment's archived `cats_dict.txt`); C is the experiment's number of
classes.

### Feature values must match the training-time image reader

The ONNX graph is reader-agnostic — it is a pure function of the 80013
vector. What matters is how the 40000+40000 image values were produced:
they MUST be produced with the same `data.img_reader_type` the model was
trained with. All supported readers yield 200x200 views in `[0, 1]`, but
the value mapping differs:

| `img_reader_type` | Pipeline |
|---|---|
| `legacy` | 16-bit raw → per-image min-max → **8-bit quantisation** → bilinear resize to 200x200 **only if the image is not already at that size** (200×200 is the native size of the corpus, so normally no resize happens) |
| `minmax` | 16-bit raw → per-image min-max → float32 (no quantisation) → bilinear resize to 200x200 if needed |
| `global` | absolute full-scale division (8-bit /255, 16-bit /65535) → float32 → bilinear resize to 200x200 if needed |

All three readers share the same pipeline order — read/normalise → resize
(only if needed) → augmentation → final normalization — and the final
`(x - img_mean) / img_std` is applied ONLY when `data.image_normalization.enable: true`
is set in the config file (enabling it without `img_mean`/`img_std` fails fast).

**Which reader did a given model use?** Read the `sws.image_reader`
metadata of the exported ONNX model — it records the training-time reader
(type, full pipeline, `image_size`, and the optional mean/std
normalization), so the answer is self-contained in the model file. The
export log prints it as well:

```text
Training-time image reader: 'legacy' (recorded in the sws.image_reader metadata; inputs must be built with it)
```

The raw 15-dim fluorescence spectrum is reduced to 13 dims by dropping the
always-zero channels at indices 5 and 10.

`build_flat_features(images, spectra, image_size, img_reader_type,
img_mean, img_std)` in `bioairmet.utils.export_onnx` is the reference
implementation for both steps (verified bit-for-bit against the actual
training-time reader classes):

```python
import json
import numpy as np
import onnxruntime as ort
from bioairmet.utils.export_onnx import build_flat_features

session = ort.InferenceSession('model.onnx', providers=['CPUExecutionProvider'])
reader = next(
    json.loads(p.value) for p in session.model.metadata_props
    if p.key == 'sws.image_reader'
)  # e.g. {'type': 'legacy', 'pipeline': '...', 'image_size': 200, ...}

images  = np.asarray(raw_images)   # (N, 2, H, W) uint16 — col 0 = view A, col 1 = view B
spectra = np.asarray(raw_spectra)  # (N, 15)

X = build_flat_features(
    images, spectra,
    image_size=reader['image_size'],
    img_reader_type=reader['type'],
    img_mean=reader['image_normalization']['img_mean'],
    img_std=reader['image_normalization']['img_std'],
)  # (N, 80013) float32
```

Equivalently, feed the exact tensors the training/validation datasets produce
(`sample['image']` and `sample['fluorescence']`), which already carry the
---

## Exporting

CLI:

```bash
# Default: export BOTH checkpoints (best + last) of the experiment —
# everything is read from the full experiment directory:
python -m bioairmet.utils.export_onnx \
    --experiment-dir /path/to/exp \
    --verify-samples 16

# Or export a single checkpoint:
python -m bioairmet.utils.export_onnx \
    --checkpoint /path/to/exp/last_classification_model.pth
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--checkpoint` | No | — | One specific classification checkpoint (`*.pth`, `last_` or `best_`). Omit it to export both best and last of the experiment. |
| `--output` | No | `<experiment>/onnx/<best\|last\|<stem>>/model.onnx` | Output `.onnx` path (single-checkpoint mode only) |
| `--experiment-dir` | Yes when `--checkpoint` is omitted | checkpoint directory | Full experiment bundle (`config.yaml` + `architecture.yaml` + `cats_dict.txt` + checkpoints) — the single source of truth for the export |
| `--opset` | No | `17` | ONNX opset version |
| `--verify-samples` | No | `16` | Validation samples used for the parity check |
| `--batch-size` | No | `4` | Batch size inside the parity check |
| `--skip-verify` | No | off | Skip the ONNX Runtime parity check |

### Output layout

Each exported checkpoint gets its own self-contained package directory inside
the experiment:

```text
<experiment>/
  onnx/
    best/
      model.onnx     <- exported from best_classification_model.pth
      README.md      <- self-describing model package
    last/
      model.onnx     <- exported from last_classification_model.pth
      README.md
```

Python API:

```python
from bioairmet.utils.export_onnx import export_experiment, convert_to_onnx

# Both standard checkpoints of the experiment (best + last):
outs = export_experiment(experiment_dir='/path/to/exp',
                         verify=True, verify_samples=16)
# ['/path/to/exp/onnx/best/model.onnx', '/path/to/exp/onnx/last/model.onnx']

# Or a single checkpoint (everything else resolved from the experiment dir):
out = convert_to_onnx(checkpoint_path='/path/to/exp/last_classification_model.pth',
                      verify=True, verify_samples=16)
# default: '/path/to/exp/onnx/last/model.onnx'
```

### What the verification checks

* `onnx.checker.check_model()` on the exported graph.
* ONNX Runtime (CPU) vs the PyTorch model (eval mode) on real validation
  samples drawn from the experiment's own data config — in small batches
  including a ragged final batch, to exercise the dynamic batch axis.
* Labels must agree exactly; probability vectors must match within
  `max |dP| < 1e-4`. Any mismatch raises an error and the export fails.

A successful run logs, for example:

```text
Exported ONNX graph to /path/to/exp/onnx/last/model.onnx
Metadata written: sws.class_names (C classes), sws.input_features
  verify batch 1: B=4 labels_match=True max|dP|=2.384e-07
  ...
ONNX Runtime parity check PASSED on 16 validation samples (max|dP|=2.794e-07)
```
---

## Consuming the model with ONNX Runtime

```python
import json
import numpy as np
import onnxruntime as ort
from bioairmet.utils.export_onnx import build_flat_features

session = ort.InferenceSession('model.onnx', providers=['CPUExecutionProvider'])

class_names = next(
    json.loads(p.value) for p in session.model.metadata_props
    if p.key == 'sws.class_names'
)
reader = next(
    json.loads(p.value) for p in session.model.metadata_props
    if p.key == 'sws.image_reader'
)

X = build_flat_features(
    images, spectra,
    image_size=reader['image_size'],
    img_reader_type=reader['type'],
    img_mean=reader['image_normalization']['img_mean'],
    img_std=reader['image_normalization']['img_std'],
)  # (N, 80013) float32
labels, probabilities = session.run(None, {'X': X})

for i in range(len(labels)):
    print(class_names[labels[i]], probabilities[i][labels[i]])
```

Notes:

* The model follows the generic ONNX classifier interface: one input `X`,
  outputs `label` (int64 tensor — not a string map) and `probabilities`
  (float32 tensor — not a ZipMap); `sws.input_features` carries the 80013
  per-column feature names in input-column order.
* The `sws.image_reader` metadata is the authoritative statement of which
  image preprocessing (reader) the model expects — build `X` with that
  reader, exactly as documented in
  [Feature values must match the training-time image reader](#feature-values-must-match-the-training-time-image-reader).
* Batch size may be 1 or any positive integer (dynamic axis).
* `probabilities` rows are finite, in `[0, 1]`, and sum to 1 within float32
  tolerance.
* The frozen image encoders run in eval mode inside the graph; BatchNorm
  running statistics are baked into the weights, so no train/eval mode
  switching is needed at inference time.
* `onnx` and `onnxruntime` are only needed for export/consumption — training
  and the PyTorch inference path do not require them.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Experiment ... is not a classification bundle` | The checkpoint's directory has no classification `config.yaml`/`architecture.yaml` — pass `--experiment-dir` explicitly. |
| `Checkpoint not found` | Use the full path to `last_classification_model.pth` or `best_classification_model.pth`. |
| ONNX Runtime shape error | The input width must be exactly `13 + 2*image_size^2` (80013 for 200x200) — rebuild `X` with the same `image_size` as the experiment's `data.image_size`. |
| Probabilities off by a lot (but shapes fine) | Feature values do not match the training pipeline — use `build_flat_features()` with the reader recorded in the `sws.image_reader` metadata (or the dataset-produced tensors) instead of a hand-rolled resize/normalisation. |
| Predictions sensible but systematically wrong (e.g. always one class) | Wrong image reader — inputs were built with `minmax`/`global` but the model was trained with `legacy` (or vice versa). Check the `sws.image_reader` metadata and rebuild `X`. |
| `ONNX Runtime probabilities deviate beyond tolerance` | Model/data mismatch (wrong checkpoint, different image size, or different fluorescence reduction) — re-export from the checkpoint that produced the experiment's config. |
training-time values.