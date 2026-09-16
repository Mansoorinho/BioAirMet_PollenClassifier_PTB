'''
@file    :   _common.py
@create date : 2026-08-25
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    Shared model-building helpers: checkpoint loading (DDP-prefix aware) and
    weight loading with clear reporting of missing / unexpected keys.
    Used by both the SSL and the classification builders.
    ]
'''
import os
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional


import torch

logger = logging.getLogger(__name__)


def strip_ddp_prefix(state_dict: dict) -> dict:
    """Remove the 'module.' prefix added by DistributedDataParallel saves.

    The state dict's ``_metadata`` attribute (per-module version info written
    by ``torch.nn.Module.state_dict()``) is preserved: torchvision models
    such as MNASNet read ``_metadata[<module>]["version"]`` when loading
    weights and raise ``ValueError: version should be set to 1 or 2 instead
    of None`` if it is missing.
    """
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict  # same object -> the _metadata attribute survives

    stripped = OrderedDict(
        (k[7:] if k.startswith("module.") else k, v) for k, v in state_dict.items()
    )
    metadata = getattr(state_dict, "_metadata", None)
    if isinstance(metadata, dict):
        stripped._metadata = OrderedDict(
            (k[7:] if k.startswith("module.") else k, v) for k, v in metadata.items()
        )
    return stripped


def ensure_state_dict_metadata(state_dict: dict, model: torch.nn.Module) -> "OrderedDict":
    """Make sure ``state_dict`` carries per-module ``_metadata`` for ``model``.

    ``torch.nn.Module.state_dict()`` stores, in ``state_dict._metadata``, a
    mapping ``<dotted module name> -> {"version": module._version, ...}``;
    ``load_state_dict()`` passes each module's entry back to it as
    ``local_metadata``. torchvision models rely on that version
    (``MNASNet._load_from_state_dict`` requires ``version in [1, 2]`` and
    raises ``ValueError: version should be set to 1 or 2 instead of None``
    otherwise).

    Checkpoints saved before this helper existed — or state dicts re-created
    with plain dict comprehensions (e.g. ``{k: v.cpu() for ...}`` when saving
    checkpoints) — have lost ``_metadata`` entirely. For those legacy
    checkpoints the metadata is synthesised from the *target* model's
    modules, which is correct because the weights being loaded must match
    the target's architecture (a MNASNet built from the same torchvision
    factory carries ``_version = 2``, or 1 for v1-pretrained builds —
    exactly the version its own weights expect).

    Existing metadata entries are never overwritten; only missing modules
    are filled in. Returns the state dict (an ``OrderedDict``) ready for
    ``model.load_state_dict()``.
    """
    if not isinstance(state_dict, OrderedDict):
        state_dict = OrderedDict(state_dict)

    metadata = getattr(state_dict, "_metadata", None)
    if not isinstance(metadata, dict):
        metadata = OrderedDict()

    for name, module in model.named_modules():
        if name not in metadata:
            metadata[name] = {"version": getattr(module, "_version", 1)}

    state_dict._metadata = metadata
    return state_dict

def get_classifier_info(state_dict):
    """Infer final classifier dimensions from a state dict."""

    candidates = []

    for key, tensor in state_dict.items():
        if key.endswith(".weight") and tensor.ndim == 2:
            candidates.append((key, tensor))

    if not candidates:
        raise ValueError("No 2D weight tensors found in state_dict.")

    # If the classifier is explicitly named, prefer it.
    classifier_candidates = [
        (key, tensor)
        for key, tensor in candidates
        if "classifier" in key.lower()
        or "head" in key.lower()
    ]

    if not classifier_candidates:
        raise ValueError(
            "Could not identify a classifier/head in state_dict."
        )

    # Take the final classifier/head Linear layer.
    key, weight = classifier_candidates[-1]

    # Linear weight shape = [out_features, in_features]
    num_classes = weight.shape[0]
    feature_dim = weight.shape[1]

    return num_classes, feature_dim, key


def load_state_dict(weights_path: str, map_location: str = "cpu") -> dict:
    """Load a checkpoint file and return its state dict.

    Handles both our save format ({'model_state_dict': ..., ...}) and raw
    state-dict files, and strips a DDP 'module.' prefix if present.

    Args:
        weights_path: path to the checkpoint file.
        map_location: torch.load device mapping (default 'cpu').

    Returns:
        dict: the (prefix-stripped) state dict.
    """
    if not weights_path or not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Checkpoint file not found: {weights_path!r}")

    checkpoint = torch.load(weights_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected a checkpoint dict (or plain state_dict), "
            f"got {type(checkpoint).__name__}: {weights_path}"
        )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    return strip_ddp_prefix(state_dict)

@dataclass
class WeightLoadResult:
    """Outcome of :func:`load_weights_into`.

    Carries the same key lists as the ``NamedTuple`` returned by
    ``model.load_state_dict()``, plus ``classifier_skipped``: True when the
    checkpoint's classifier output dim differed from the requested
    ``num_classes`` and its weights were dropped on purpose (the model's
    head stays randomly initialized).
    """
    missing_keys: list
    unexpected_keys: list
    classifier_skipped: bool = False


def load_weights_into(
    model: torch.nn.Module,
    weights_path: str,
    num_classes: Optional[int] = None,
    strict: bool = False,
    label: Optional[str] = None,
) -> object:
    """Load checkpoint weights into `model`.

    If the checkpoint contains a classifier, its output dimension is inferred
    automatically. When `num_classes` is supplied, it is compared against the
    checkpoint's classifier dimension. If they differ, classifier weights are
    excluded and only compatible encoder weights are loaded.

    Args:
        model: Target module.
        weights_path: Checkpoint file to load.
        num_classes: Expected number of classes for the target model. If None,
            no classifier-dimension check is performed.
        strict: Passed to `load_state_dict`.
        label: Optional tag prefixed to log lines.

    Returns:
        WeightLoadResult with the ``load_state_dict()`` missing/unexpected key
        lists plus ``classifier_skipped`` (True when the checkpoint's
        classifier was dropped because its output dim differs from
        ``num_classes``).
    """

    tag = f"[{label}] " if label else ""

    state_dict = load_state_dict(weights_path)
    classifier_skipped = False

    # Try to infer the classifier dimensions from the checkpoint.
    checkpoint_num_classes = None
    checkpoint_feature_dim = None
    classifier_key = None

    try:
        (
            checkpoint_num_classes,
            checkpoint_feature_dim,
            classifier_key,
        ) = get_classifier_info(state_dict)

        logger.info(
            "%sCheckpoint classifier: %s (%d features -> %d classes)",
            tag,
            classifier_key,
            checkpoint_feature_dim,
            checkpoint_num_classes,
        )

    except ValueError:
        logger.info(
            "%sNo classifier found in checkpoint; treating it as encoder-only.",
            tag,
        )

    # ------------------------------------------------------------------
    # Check classifier dimensions if the caller supplied num_classes.
    # ------------------------------------------------------------------
    if num_classes is not None and checkpoint_num_classes is not None:
        if checkpoint_num_classes != num_classes:
            logger.warning(
                "%sClassifier output dimension mismatch: "
                "checkpoint has %d classes, model expects %d classes.",
                tag,
                checkpoint_num_classes,
                num_classes,
            )

            # Remove classifier weights from the checkpoint. This allows
            # the encoder/backbone to load while leaving the model's
            # classifier randomly initialized.
            classifier_prefix = classifier_key.rsplit(".", 2)[0]

            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith(classifier_prefix + ".")
            }

            logger.info(
                "%sRemoved classifier weights (%s); "
                "loading encoder weights only.",
                tag,
                classifier_prefix,
            )
            classifier_skipped = True


    # ------------------------------------------------------------------
    # Load the weights.
    # ------------------------------------------------------------------
    # Legacy checkpoints lost the per-module `_metadata` that torchvision
    # models (e.g. MNASNet) need when loading; restore it if it is missing.
    state_dict = ensure_state_dict_metadata(state_dict, model)
    result = model.load_state_dict(
        state_dict,
        strict=strict,
    )

    if result.missing_keys:
        logger.info(
            "%s%d missing keys.",
            tag,
            len(result.missing_keys),
        )

        logger.info(
            "%sMissing keys: %s",
            tag,
            result.missing_keys[:10],
        )

    if result.unexpected_keys:
        logger.info(
            "%s%d unexpected keys.",
            tag,
            len(result.unexpected_keys),
        )

        logger.info(
            "%sUnexpected keys: %s",
            tag,
            result.unexpected_keys[:10],
        )

    return WeightLoadResult(
        missing_keys=result.missing_keys,
        unexpected_keys=result.unexpected_keys,
        classifier_skipped=classifier_skipped,
    )



def _owns_params(module: torch.nn.Module) -> bool:
    """True when ``module`` (or one of its descendants) owns a parameter."""
    return next(module.parameters(), None) is not None


def _select_layer_units(module: torch.nn.Module, min_units: int = 2,
                        max_depth: int = 8) -> list:
    """Flatten wrapper modules until the units are real layer granularity.

    ``GrayscaleBackbone`` holds ``self.model`` (an ``EfficientNet``), whose own
    children are ``[features, avgpool, classifier]`` - only ``features`` owns
    parameters.  Stopping one level down would therefore make "unfreeze the
    last N layers" mean "unfreeze the whole backbone" for every ``N``.  So the
    deepest single parameter-bearing container is expanded in place until the
    unit list holds at least ``min_units`` parameter-bearing units (blocks of
    ``features`` / layers of an MLP) or no further expansion is possible.
    """
    units = list(module.children())
    for _ in range(max_depth):
        param_pos = [i for i, unit in enumerate(units) if _owns_params(unit)]
        if len(param_pos) >= min_units:
            break
        expanded = False
        for i in reversed(param_pos):        # expand the deepest container first
            kids = list(units[i].children())
            if not kids:
                continue
            # Descending is worth it when it adds granularity: either several
            # children (blocks of a backbone, layers of an MLP) or a single
            # wrapper child that still carries the parameters.
            if len(kids) > 1 or _owns_params(kids[0]):
                units = units[:i] + kids + units[i + 1:]
                expanded = True
                break
        if not expanded:
            break
    return units


def select_last_n_layer_modules(module: torch.nn.Module, n: int) -> list:
    """Select the last N parameter-bearing "layer units" of an encoder module.

    Used for partial fine-tuning ("unfreeze the last N layers").

    Rules:
        1. Wrapper modules are flattened until the unit list has real layer
           granularity, i.e. until at least two children own parameters (this
           unwraps single-container wrappers such as ``img_encoder.model``
           backbones and ``fl_encoder.encoder`` MLPs and, more importantly,
           descends into the ``features`` container of an EfficientNet instead
           of treating it as a single layer).
        2. The selection is the suffix of the unit list that starts at the
           Nth parameter-bearing unit counted from the output side.  Units
           without parameters (activations, dropout, pooling) ride along when
           they fall inside the suffix, so e.g. a final
           ``Linear -> Act -> Dropout`` tail is kept intact.
        3. If ``n`` is >= the number of parameter-bearing units, all units
           are returned (i.e. the whole encoder).

    Args:
        module: the encoder module (e.g. HoloClassifierV2.img_encoder).
        n: number of trailing parameter-bearing layer units (>= 1).

    Returns:
        list[nn.Module]: the selected trailing layer units (may be empty if
        the module has no parameters at all).
    """
    if n is None or n <= 0:
        return []
    units = _select_layer_units(module)

    param_unit_indices = [i for i, unit in enumerate(units) if list(unit.parameters())]
    if not param_unit_indices:
        return []
    if n >= len(param_unit_indices):
        return list(units)
    start = param_unit_indices[-n]
    return list(units[start:])


__all__ = [
    "get_classifier_info",
    "WeightLoadResult",
    "strip_ddp_prefix", "load_state_dict", "load_weights_into",
    "ensure_state_dict_metadata",
    "select_last_n_layer_modules",
]