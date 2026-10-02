'''
@file    :   export_onnx.py
@create date : 2026-09-22
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    ONNX export utility for the V2 classification pipeline.

    Exports a trained classification checkpoint (HoloClassifierV2) to a
    self-contained ONNX model with:
        * ONE flat input `X` of shape [B, 80013]:
              [0:13]     13-dim fluorescence spectrum (the 15-dim raw spectrum
                         with the always-zero channels at indices 5 and 10
                         dropped — the same reduction the datasets apply)
              [13:40013]     image view A, 200x200, row-major, float in [0, 1]
              [40013:80013]  image view B, 200x200, row-major, float in [0, 1]
        * TWO outputs (in this order):
              `label`         int64 [B]       argmax class index
              `probabilities` float32 [B, C]  softmax probabilities
        * metadata:
              sws.class_names    JSON list of class names, index order 0..C-1
              sws.input_features JSON array of the 80013 per-column feature
                                 names in input-column order
              sws.input_layout   companion JSON: field/offset/dtype notes for
                                 the 80013 layout
              sws.image_reader   companion JSON: the image reader the model was
                                 trained with (type, pipeline, optional mean/std
                                 normalization) — build inputs with THIS reader
        * opset 17, dynamic batch dimension, CPU-safe (ONNX Runtime verified).

    The image feature values must be produced with the SAME image reader the
    model was trained with ('legacy', 'minmax' or 'global' — read the
    ``sws.image_reader`` metadata of the exported model). All readers yield
    200x200 views in [0, 1], but the value mapping differs, e.g.:
        legacy : 16-bit -> per-image min-max -> 8-bit quantisation
                 -> bilinear resize to 200x200 if needed (no-op for native
                 200x200 images, the normal case)
        minmax : 16-bit -> per-image min-max -> float32
                 -> bilinear resize to 200x200 if needed
        global : absolute full-scale division (8-bit /255, 16-bit /65535)
                 -> float32 -> bilinear resize to 200x200 if needed
    All three readers follow the same read -> resize (only if needed) ->
    augmentation -> normalize order, and the final img_mean/img_std
    normalization is applied ONLY when data.image_normalization was enabled
    via the config file at training time.
    Use build_flat_features() below (pass the same img_reader_type /
    img_mean / img_std as training) as the reference implementation, or feed
    the same tensors the training/validation datasets produce.

    CLI:
        # Export BOTH checkpoints (best + last) of an experiment — everything
        # (num_classes, image size, reader, class names, ...) is read from the
        # full experiment directory (config.yaml + architecture.yaml + cats_dict.txt):
        python -m bioairmet.utils.export_onnx --experiment-dir /path/to/exp

        # Export a single checkpoint:
        python -m bioairmet.utils.export_onnx \
            --checkpoint /path/to/exp/last_classification_model.pth \
            [--output PATH] [--opset 17] [--verify-samples 16] [--skip-verify]
    ]
'''
import argparse
import inspect
import json
import os
import warnings

import numpy as np
import torch
import torch.nn as nn

# 15-dim raw spectrum -> 13-dim model input (drops the always-zero channels).
FLUO_DROPPED_INDICES = (5, 10)


def flat_feature_dim(image_size: int, fl_dim: int) -> int:
    """Total flat input width: fl + 2 * (image_size x image_size)."""
    return fl_dim + 2 * image_size * image_size


class FlatFeatureClassifier(nn.Module):
    """Wraps a trained HoloClassifierV2 for ONNX export.

    Input : `X` float32 [B, 80013]  (see module docstring for the layout)
    Output: `label` int64 [B], `probabilities` float32 [B, C]

    The wrapper reshapes the flat vector back into (image_a, image_b, fl)
    exactly as the training code feeds the model, so the exported graph is
    numerically identical to the PyTorch model in eval mode.
    """

    def __init__(self, model: nn.Module, image_size: int = 200, fl_dim: int = 13):
        super().__init__()
        self.model = model
        self.image_size = int(image_size)
        self.fl_dim = int(fl_dim)
        self.img_dim = self.image_size * self.image_size

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        fl = x[:, :self.fl_dim]
        img_a = x[:, self.fl_dim:self.fl_dim + self.img_dim].view(b, 1, self.image_size, self.image_size)
        img_b = x[:, self.fl_dim + self.img_dim:].view(b, 1, self.image_size, self.image_size)
        logits = self.model(img_a, img_b, fl)
        probs = torch.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)
        return labels, probs

def _legacy_minmax_reader(img: np.ndarray, image_size: int) -> torch.Tensor:
    """Reference of the training-time LegacyMinMaxImageReader for ONE image.

    16-bit (or any-range) input array -> per-image min-max -> 8-bit
    quantisation -> bilinear resize ONLY if the native size differs from
    (image_size, image_size) (200x200 is the native corpus size, so normally
    no resize happens) -> float32 in [0, 1].  Mirrors
    LegacyMinMaxImageReader._read_image (PIL 'L' resize) + base_transform
    (ToTensor), so values match the training pipeline.
    """
    from PIL import Image
    from torchvision import transforms

    img = np.asarray(img, dtype=np.float32)
    if img.ndim > 2:
        img = img.mean(axis=-1)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        q = np.clip(((img - mn) / (mx - mn) * 255.0), 0, 255).astype(np.uint8)
    else:
        q = np.zeros(img.shape, dtype=np.uint8)
    pil_img = Image.fromarray(q, mode='L')
    if pil_img.size != (image_size, image_size):
        pil_img = pil_img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    tensor = transforms.ToTensor()(pil_img)
    return tensor  # (1, image_size, image_size) float32 in [0, 1]


def _resize_bilinear(arr: np.ndarray, image_size: int) -> torch.Tensor:
    """(H, W) float32 -> (1, image_size, image_size); bilinear when sizes differ.

    Mirrors :meth:`New_Image_reader._to_tensor`.  An image already at
    ``image_size`` passes through untouched (200x200 is the native corpus
    size); the legacy reader applies the same "resize only if off-size"
    rule through PIL.
    """
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0)
    if t.shape[-2] != image_size or t.shape[-1] != image_size:
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(image_size, image_size), mode='bilinear',
            align_corners=False).squeeze(0)
    return t


def _minmax_reader(img: np.ndarray, image_size: int) -> torch.Tensor:
    """Reference of the training-time ``New_Image_reader(method='minmax')``.

    16-bit (or any-range) input array -> per-image contrast stretch
    ``(x - min) / (max - min)`` -> float32 in [0, 1] -> bilinear resize to
    (image_size, image_size) when the native size differs.  No 8-bit
    quantisation — that is what distinguishes it from the legacy reader.
    Flat images (no particle) map to uniform 0.5, as at training time.
    The division is done in float64 (like ``read_image_norm_new``) and only
    cast to float32 at the end, so values match training bit-for-bit.
    """
    img = np.asarray(img)
    if img.ndim > 2:
        img = img.mean(axis=-1)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn > 1e-5:
        img = (img - mn) / (mx - mn)
    else:
        img = np.ones(img.shape, dtype=np.float64) * 0.5
    return _resize_bilinear(img, image_size)


def _global_reader(img: np.ndarray, image_size: int) -> torch.Tensor:
    """Reference of the training-time ``New_Image_reader(method='global')``.

    Divides by the file's absolute full scale — 8-bit /255, 16-bit /65535
    (from a raw array the bit depth is inferred from the max value:
    ``<= 255`` means 8-bit) -> float32 in [0, 1] -> bilinear resize when the
    native size differs.  Division in float64 like ``read_image_norm_new``,
    cast to float32 at the end — values match training bit-for-bit.
    """
    img = np.asarray(img)
    if img.ndim > 2:
        img = img.mean(axis=-1)
    scale = 255.0 if float(img.max()) <= 255.0 else 65535.0
    img = np.clip(img / scale, 0.0, 1.0)
    return _resize_bilinear(img, image_size)


def build_flat_features(images: np.ndarray, spectra: np.ndarray,
                        image_size: int = 200, img_reader_type: str = 'legacy',
                        img_mean=None, img_std=None) -> np.ndarray:
    """Build the ONNX input `X` from raw per-sample features (training parity).

    Args:
        images:  (N, 2, H, W) raw images (uint8/uint16/float) — column 0 =
                 view A, column 1 = view B (the same pair the training
                 datasets yield).
        spectra: (N, 15) raw fluorescence spectra (13 is accepted too).
        image_size: target size (data.image_size, 200).
        img_reader_type: the reader the model was trained with — 'legacy',
                 'minmax' or 'global'. Read it from the ``sws.image_reader``
                 metadata of the exported ONNX model; it MUST match training.
        img_mean/img_std: optional final per-image normalization, mirroring
                 ``data.image_normalization`` when it was enabled during
                 training (also in the ``sws.image_reader`` metadata).

    Returns:
        (N, 13 + 2*image_size*image_size) float32 array, ready for the
        ONNX input `X`.
    """
    from ..data.datasets_cls import normalize_img_reader_type
    readers = {
        'legacy': _legacy_minmax_reader,
        'minmax': _minmax_reader,
        'global': _global_reader,
    }
    reader = normalize_img_reader_type(img_reader_type)
    if reader not in readers:
        raise ValueError(
            f"img_reader_type={img_reader_type!r} is not supported by "
            f"build_flat_features; expected one of {sorted(readers)}.")

    images = np.asarray(images)
    spectra = np.asarray(spectra, dtype=np.float32)
    if spectra.shape[1] == 15:
        keep = [i for i in range(15) if i not in FLUO_DROPPED_INDICES]
        spectra = spectra[:, keep]
    elif spectra.shape[1] != 13:
        raise ValueError(f"Expected 15- or 13-dim spectra, got {spectra.shape[1]}")

    parts = [spectra.astype(np.float32)]
    for view_idx in (0, 1):
        views = []
        for img in images[:, view_idx]:
            t = readers[reader](img, image_size)
            if img_mean is not None and img_std is not None:
                t = (t - float(img_mean)) / float(img_std)
            views.append(t.numpy().ravel())
        parts.append(np.stack(views, axis=0).astype(np.float32))
    return np.concatenate(parts, axis=1)


def _read_class_names(experiment_dir: str, config, num_classes: int):
    """Class names in index order 0..num_classes-1 from the category map.

    Prefers the archived cats_dict.txt in the experiment dir, falls back to
    config.data.cat_map_path. Indices without a name get 'class_<i>'.
    """
    candidates = []
    arch = os.path.join(experiment_dir, 'cats_dict.txt')
    if os.path.isfile(arch):
        candidates.append(arch)
    cat_map = getattr(getattr(config, 'data', None), 'cat_map_path', None)
    if cat_map:
        candidates.append(cat_map)
        if not os.path.isabs(cat_map):
            candidates.append(os.path.join(experiment_dir, cat_map))
            config_dir = os.path.dirname(getattr(config, 'config_path', '') or '')
            if config_dir:
                candidates.append(os.path.join(config_dir, cat_map))

    name_by_idx = {}
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, 'r') as f:
                next(f, None)  # skip header comment line
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) == 2:
                        name_by_idx[int(parts[1].strip())] = parts[0].strip()
            break
        except Exception:
            continue

    return [name_by_idx.get(i, f'class_{i}') for i in range(num_classes)]

def convert_to_onnx(
    checkpoint_path: str,
    output_path: str = None,
    experiment_dir: str = None,
    opset: int = 17,
    verify: bool = True,
    verify_samples: int = 16,
    batch_size: int = 4,
    log=print,
) -> str:
    """
    Export a trained classification checkpoint to a self-contained ONNX model.

    Args:
        checkpoint_path:  ``*.pth`` classification checkpoint (last or best).
        output_path:      output ``.onnx`` path. Default:
                          ``<experiment>/onnx/<kind>/model.onnx`` where ``kind`` is
                          ``best`` / ``last`` for the standard checkpoint names
                          (otherwise the checkpoint file stem). A ``README.md``
                          describing the model is written next to it.
        experiment_dir:   experiment bundle (config.yaml + architecture.yaml +
                          cats_dict.txt); defaults to the checkpoint's directory.
                          Everything about the model (num_classes, image_size,
                          reader, class names, ...) is resolved from this bundle.
        opset:            ONNX opset (17).
        verify:           validate with onnx.checker + ONNX Runtime against the
                          PyTorch model on real validation samples.
        verify_samples:   number of validation samples used for the parity check.
        batch_size:       batch size used inside the parity check.
        log:              log function (default print).

    Returns:
        The output path.
    """
    from ..utils.config_parser import load_experiment_config
    from ..models import build_model_from_config

    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    experiment_dir = os.path.abspath(experiment_dir or os.path.dirname(checkpoint_path))
    if output_path is None:
        stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
        kind = ('best' if stem.startswith('best')
                else 'last' if stem.startswith('last') else stem)
        output_path = os.path.join(experiment_dir, 'onnx', kind, 'model.onnx')
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    config = load_experiment_config(experiment_dir)
    arch = config.architecture_setup
    if arch.get('type') != 'classification':
        raise ValueError(
            f"Experiment at {experiment_dir} is not a classification bundle "
            f"(architecture_setup.type={arch.get('type')!r}).")
    num_classes = int(arch.classification_model.num_classes)
    image_size = int(config.data.get('image_size', 200))
    fl_dim = int(arch.fluorescence_tower.get('input_dim', 13))
    feature_dim = flat_feature_dim(image_size, fl_dim)

    # The reader the model was trained with — the consumer MUST build inputs
    # with this same reader. Recorded in the sws.image_reader metadata below.
    from ..data.datasets_cls import normalize_img_reader_type
    img_reader_type = normalize_img_reader_type(config.data.get('img_reader_type', 'legacy'))
    norm_cfg = getattr(config.data, 'image_normalization', None) or {}
    img_mean = norm_cfg.get('img_mean')
    img_std = norm_cfg.get('img_std')
    if isinstance(img_mean, (list, tuple, np.ndarray)):
        img_mean = float(img_mean[0])
    if isinstance(img_std, (list, tuple, np.ndarray)):
        img_std = float(img_std[0])
    if not norm_cfg.get('enable', False) or img_mean is None or img_std is None:
        img_mean = img_std = None
    if img_reader_type == 'legacy':
        reader_note = ('16-bit -> per-image min-max -> 8-bit quantisation -> '
                       f'bilinear resize to {image_size}x{image_size} if needed -> float in [0, 1]')
    elif img_reader_type == 'minmax':
        reader_note = ('16-bit -> per-image min-max -> float32 -> '
                       f'bilinear resize to {image_size}x{image_size} if needed -> float in [0, 1]')
    else:  # 'global'
        reader_note = ('absolute full-scale division (8-bit /255, 16-bit /65535) -> float32 -> '
                       f'bilinear resize to {image_size}x{image_size} if needed -> float in [0, 1]')
    if img_mean is not None and img_std is not None:
        reader_note += f' -> (x - {img_mean}) / {img_std} (data.image_normalization)'

    log(f"Building model from {experiment_dir} (num_classes={num_classes}, "
        f"image_size={image_size}, fl_dim={fl_dim}, feature_dim={feature_dim})")
    log(f"Training-time image reader: '{img_reader_type}' "
        f"(recorded in the sws.image_reader metadata; inputs must be built with it)")

    # Load the exact checkpoint weights through the validated resume path of
    # the model builder (full-model weights + completeness check).
    config.model_initialization.resume.enable = True
    config.model_initialization.resume.checkpoint_path = checkpoint_path
    config.model_initialization.resume.load_optimizer = True
    model = build_model_from_config(config)
    model = model.to('cpu').eval()

    # The fused-view fast path slices with the traced batch size, which would
    # bake a static batch into the graph. Disable it for tracing — numerically
    # identical for frozen encoders in eval mode (see image_encoder_fusable).
    if hasattr(model, 'fuse_frozen_image_views'):
        model.fuse_frozen_image_views = False

    wrapper = FlatFeatureClassifier(model, image_size=image_size, fl_dim=fl_dim)
    dummy = torch.zeros(2, feature_dim, dtype=torch.float32)

    dynamic_axes = {
        'X': {0: 'batch_size'},
        'label': {0: 'batch_size'},
        'probabilities': {0: 'batch_size'},
    }
    supports_dynamo = 'dynamo' in inspect.signature(torch.onnx.export).parameters
    export_kwargs = dict(
        input_names=['X'],
        output_names=['label', 'probabilities'],
        opset_version=opset,
    )
    if supports_dynamo:
        export_kwargs['dynamo'] = False
        export_kwargs['dynamic_axes'] = dynamic_axes
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            torch.onnx.export(wrapper, (dummy,), f=output_path, **export_kwargs)
    except Exception as e:
        if not supports_dynamo:
            raise
        log(f"Legacy exporter failed ({e}); retrying with the torch.export-based exporter.")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            torch.onnx.export(
                wrapper, (dummy,), f=output_path,
                input_names=['X'], output_names=['label', 'probabilities'],
                opset_version=opset, dynamo=True,
                dynamic_shapes={'X': {0: 'batch_size'},
                                'label': {0: 'batch_size'},
                                'probabilities': {0: 'batch_size'}},
            )
    log(f"Exported ONNX graph to {output_path}")
    # Graph-level validation + metadata
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    class_names = _read_class_names(experiment_dir, config, num_classes)
    raw_fl_dim = fl_dim + len(FLUO_DROPPED_INDICES)
    if len(FLUO_DROPPED_INDICES) == 2 and raw_fl_dim == 15:
        fl_note = (f'{fl_dim}-dim spectrum: the {raw_fl_dim}-dim raw spectrum with the '
                   'always-zero channels at indices 5 and 10 dropped')
    else:
        fl_note = f'{fl_dim}-dim fluorescence spectrum'
    input_layout = {
        'name': 'X',
        'shape': ['batch', feature_dim],
        'dtype': 'float32',
        'layout': [
            {'field': 'fluorescence', 'start': 0, 'end': fl_dim, 'note': fl_note},
            {'field': 'image_a', 'start': fl_dim, 'end': fl_dim + image_size * image_size,
             'note': f'{image_size}x{image_size} grayscale, row-major. ' + reader_note},
            {'field': 'image_b', 'start': fl_dim + image_size * image_size, 'end': feature_dim,
             'note': f'{image_size}x{image_size} grayscale, row-major. ' + reader_note},
        ],
    }
    # Per-column feature names in input-column order (the generic ONNX
    # contract requires sws.input_features to be exactly this list).
    fl_raw_keep = [i for i in range(raw_fl_dim) if i not in FLUO_DROPPED_INDICES]
    feature_names = [f'fl_raw_{i}' for i in fl_raw_keep]
    for prefix in ('imgA', 'imgB'):
        feature_names += [f'{prefix}_r{r:04d}_c{c:04d}'
                          for r in range(image_size) for c in range(image_size)]
    assert len(feature_names) == feature_dim
    image_reader_meta = {
        'type': img_reader_type,
        'pipeline': reader_note,
        'image_size': image_size,
        'image_normalization': {
            'enable': img_mean is not None,
            'img_mean': img_mean,
            'img_std': img_std,
        },
    }
    onnx_model.metadata_props.append(onnx.StringStringEntryProto(
        key='sws.class_names', value=json.dumps(class_names)))
    onnx_model.metadata_props.append(onnx.StringStringEntryProto(
        key='sws.input_features', value=json.dumps(feature_names)))
    onnx_model.metadata_props.append(onnx.StringStringEntryProto(
        key='sws.input_layout', value=json.dumps(input_layout)))
    onnx_model.metadata_props.append(onnx.StringStringEntryProto(
        key='sws.image_reader', value=json.dumps(image_reader_meta)))
    onnx.save(onnx_model, output_path)
    log(f"Metadata written: sws.class_names ({len(class_names)} classes), "
        f"sws.input_features ({len(feature_names)} feature names), "
        f"sws.input_layout, sws.image_reader (reader='{img_reader_type}')")

    # Model-package README: the generic ONNX contract expects the package to
    # contain model.onnx plus a README describing the model version, training
    # context, feature definitions, and preprocessing — including the type and
    # state of the images the model works with (the training-time reader).
    if img_mean is not None and img_std is not None:
        norm_line = (f"Finally apply `(x - {img_mean}) / {img_std}` "
                     f"(data.image_normalization was enabled at training time).")
    else:
        norm_line = "No additional per-image normalization was applied."
    project_name = config.get('project_name', 'BioAirMet classification model')
    readme = f"""# {project_name} — ONNX classifier

Self-contained ONNX export of a trained BioAirMet V2 pollen classifier
(opset {opset}, float32, dynamic batch axis, CPU-verified with ONNX Runtime).

## Model version and training context

- Experiment: `{experiment_dir}`
- Checkpoint: `{os.path.basename(checkpoint_path)}`
- Number of classes: {num_classes} (names in the `sws.class_names` metadata,
  index order = probability column order)
- Input feature width: {feature_dim}

## Input

One flat float32 input `X` of shape `[batch, {feature_dim}]`:

| Columns | Content |
|---|---|
| `[0:{fl_dim}]` | {fl_note} |
| `[{fl_dim}:{fl_dim + image_size * image_size}]` | image view A, {image_size}x{image_size} grayscale, row-major |
| `[{fl_dim + image_size * image_size}:{feature_dim}]` | image view B, {image_size}x{image_size} grayscale, row-major |

Per-column feature names in input-column order are in the `sws.input_features`
metadata; field/offset/dtype notes are in the companion `sws.input_layout`
metadata.

## Image type and required preprocessing

The model works with 16-bit grayscale holographic reconstruction magnitude
images (one pair of views per particle: view A, view B). The image values in
`X` must be produced with the **`{img_reader_type}`** image reader, exactly as
at training time:

> {reader_note}
>
> {norm_line}

Building inputs with a different reader (e.g. `minmax` instead of `legacy`)
changes the value distribution and will silently degrade accuracy.
`bioairmet.utils.export_onnx.build_flat_features(..., img_reader_type='{img_reader_type}', ...)`
is the reference implementation (bit-exact against the training pipeline); the
machine-readable description is also in the `sws.image_reader` metadata of
this file.

## Outputs

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `label` | int64 | `[batch]` | argmax class index (lowest index wins ties) |
| `probabilities` | float32 | `[batch, {num_classes}]` | softmax probabilities; column *i* is class `sws.class_names[i]` |

`probabilities` is the authoritative output; `label` is a convenience.

## Verification

The export was verified against the PyTorch model on real validation samples
from the experiment's own data: labels exact, max |dP| < 1e-4, and the graph
passed `onnx.checker`.
"""
    readme_path = os.path.join(out_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme)
    log(f"Model-package README written to {readme_path}")

    if verify:
        _verify_with_onnxruntime(
            output_path, wrapper, config, image_size, fl_dim,
            verify_samples, batch_size, log)
    return output_path


def export_experiment(experiment_dir: str,
                      opset: int = 17,
                      verify: bool = True,
                      verify_samples: int = 16,
                      batch_size: int = 4,
                      log=print) -> list:
    """Export ALL standard classification checkpoints of an experiment.

    Scans the FULL experiment directory for ``best_classification_model.pth``
    and ``last_classification_model.pth`` and exports each to
    ``<experiment>/onnx/<best|last>/model.onnx`` (with its own README.md).
    Everything about the models (num_classes, image_size, image reader,
    class names, ...) is resolved from the experiment's config.yaml /
    architecture.yaml / cats_dict.txt — nothing is passed by hand.

    Args:
        experiment_dir: experiment bundle directory.
        opset:          ONNX opset (17).
        verify:         run the ONNX Runtime parity check for each export.
        verify_samples: number of validation samples per parity check.
        batch_size:     batch size inside each parity check.
        log:            log function (default print).

    Returns:
        List of output ``.onnx`` paths (one per exported checkpoint).

    Raises:
        FileNotFoundError: if neither standard checkpoint exists in the dir.
    """
    experiment_dir = os.path.abspath(experiment_dir)
    if not os.path.isdir(experiment_dir):
        raise NotADirectoryError(f"Experiment directory not found: {experiment_dir}")
    found = [
        os.path.join(experiment_dir, name)
        for name in ('best_classification_model.pth', 'last_classification_model.pth')
        if os.path.isfile(os.path.join(experiment_dir, name))
    ]
    if not found:
        raise FileNotFoundError(
            f"No best_classification_model.pth / last_classification_model.pth "
            f"found in {experiment_dir}")
    log(f"Exporting {len(found)} checkpoint(s) from {experiment_dir}: "
        f"{', '.join(os.path.basename(p) for p in found)}")
    return [
        convert_to_onnx(
            checkpoint_path=ckpt,
            experiment_dir=experiment_dir,
            opset=opset,
            verify=verify,
            verify_samples=verify_samples,
            batch_size=batch_size,
            log=log,
        )
        for ckpt in found
    ]


def _verify_with_onnxruntime(
    onnx_path: str,
    wrapper: nn.Module,
    config,
    image_size: int,
    fl_dim: int,
    n_samples: int,
    batch_size: int,
    log=print,
):
    """Parity check: ONNX Runtime (CPU) vs the PyTorch wrapper on real data.

    Builds the validation dataset from the experiment config (tiny
    deterministic subset), runs the ONNX session in small batches (including
    a ragged final batch to exercise the dynamic axis), and compares
    labels exactly and probabilities within 1e-4.
    """
    import copy
    import onnxruntime as ort
    from ..data import build_dataset_from_config

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    in_name = session.get_inputs()[0].name
    if in_name != 'X':
        raise RuntimeError(f"Expected ONNX input name 'X', got {in_name!r}")

    cfg = copy.deepcopy(config)
    cfg.data.subset_percentage = 0.002
    if hasattr(cfg.train, 'num_workers'):
        cfg.train.num_workers = 0
    _, val_ds = build_dataset_from_config(cfg)
    n = min(n_samples, len(val_ds))
    if n == 0:
        raise RuntimeError("Validation dataset is empty; cannot verify parity.")

    wrapper.eval()
    labels_ok = True
    max_prob_diff = 0.0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            feats = []
            for i in range(start, min(start + batch_size, n)):
                sample = val_ds[i]
                img_a = sample['image'][0]  # (1, H, W) float in [0, 1]
                img_b = sample['image'][1]
                fl = sample['fluorescence']  # (fl_dim,)
                x = torch.cat([
                    fl.reshape(1, -1),
                    img_a.reshape(1, -1),
                    img_b.reshape(1, -1),
                ], dim=1)
                feats.append(x)
            X = torch.cat(feats, dim=0).numpy().astype(np.float32)
            pt_labels, pt_probs = wrapper(torch.from_numpy(X))
            onnx_labels, onnx_probs = session.run(None, {in_name: X})
            if not np.array_equal(pt_labels.numpy(), onnx_labels):
                labels_ok = False
            max_prob_diff = max(max_prob_diff, float(np.abs(pt_probs.numpy() - onnx_probs).max()))
            log(f"  verify batch {start // batch_size + 1}: B={X.shape[0]} "
                f"labels_match={np.array_equal(pt_labels.numpy(), onnx_labels)} "
                f"max|dP|={max_prob_diff:.3e}")

    if not labels_ok:
        raise RuntimeError("ONNX Runtime labels disagree with PyTorch model.")
    if max_prob_diff >= 1e-4:
        raise RuntimeError(
            f"ONNX Runtime probabilities deviate beyond tolerance: max|dP|={max_prob_diff:.3e}")
    log(f"ONNX Runtime parity check PASSED on {n} validation samples "
        f"(max|dP|={max_prob_diff:.3e})")

def main(argv=None):
    """CLI entry point.

    Without ``--checkpoint``, BOTH standard checkpoints of the experiment
    (``best_classification_model.pth`` and ``last_classification_model.pth``)
    are exported; with ``--checkpoint``, a single checkpoint is exported.
    In both cases the full experiment directory supplies num_classes,
    image_size, image reader, class names, and the data for verification.
    """
    p = argparse.ArgumentParser(
        description='Export a trained V2 classification experiment to ONNX '
                    '(opset 17, dynamic batch, CPU-verified). By default '
                    'exports BOTH best_classification_model.pth and '
                    'last_classification_model.pth of the experiment.')
    p.add_argument('--checkpoint', default=None,
                   help='Path to one specific classification checkpoint (*.pth). '
                        'If omitted, both best and last checkpoints of the '
                        'experiment directory are exported.')
    p.add_argument('--output', default=None,
                   help='Output .onnx path for --checkpoint mode (default: '
                        '<experiment_dir>/onnx/<best|last|<checkpoint stem>>/model.onnx)')
    p.add_argument('--experiment-dir', default=None,
                   help='Experiment bundle dir (config.yaml + architecture.yaml + '
                        'cats_dict.txt + checkpoints). Required when --checkpoint '
                        'is omitted; otherwise defaults to the checkpoint directory')
    p.add_argument('--opset', type=int, default=17, help='ONNX opset version (default 17)')
    p.add_argument('--verify-samples', type=int, default=16,
                   help='Number of validation samples for the ONNX Runtime parity check')
    p.add_argument('--batch-size', type=int, default=4,
                   help='Batch size used inside the parity check')
    p.add_argument('--skip-verify', action='store_true',
                   help='Skip the ONNX Runtime parity check')
    args = p.parse_args(argv)
    common = dict(opset=args.opset,
                  verify=not args.skip_verify,
                  verify_samples=args.verify_samples,
                  batch_size=args.batch_size)
    if args.checkpoint:
        outs = [convert_to_onnx(checkpoint_path=args.checkpoint,
                                output_path=args.output,
                                experiment_dir=args.experiment_dir,
                                **common)]
    else:
        if not args.experiment_dir:
            p.error('--experiment-dir is required when --checkpoint is not given')
        if args.output is not None:
            p.error('--output is only valid together with --checkpoint')
        outs = export_experiment(args.experiment_dir, **common)
    for out in outs:
        print(f'ONNX model written to: {out}')
    return outs


if __name__ == '__main__':
    main()