'''
@file    :   backbones.py
@create date : 2025-07-01 09:15:35
@modify date 2026-08-25
@author  :   Mansoor Nabawi
@version :   2.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    Image backbones adapted for single-channel (grayscale) holographic input.

    Every backbone follows the same three-step recipe, which keeps the code
    modular and easy to extend:

        1. SELECT   _select_<family>(variant, pretrained)
                    build the stock (RGB) model for a variant
        2. ADAPT    _adapt_<family>(model, in_channels)
                    convert the first conv layer(s) to the desired number of
                    input channels and replace the classification head with a
                    pure feature output
        3. WRAP     a thin nn.Module exposing forward / get_output_dim /
                    freeze_batchnorm

    The public class names (EfficientNet_grayscale, fastViT_grayscale, ...)
    and the wrapper attribute `model` are part of every saved state_dict and
    must therefore NOT be renamed.

    Adding a new backbone family only needs: one _select_ function, one
    _adapt_ function and one entry in BACKBONE_REGISTRY. Families that do not
    follow the select/adapt recipe (e.g. a hand-written CNN) can be added from
    any module with register_backbone(build=...); see models/custom_models.py
    for a worked template.
    ]
'''

import logging

import torch
import torch.nn as nn
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

import timm
from functools import partial
from torchvision import models
from torchvision.models.mobilenetv3 import (
    InvertedResidualConfig,
    _mobilenet_v3_conf,
    MobileNetV3,
)
from torchvision.models.efficientnet import EfficientNet

# EfficientNet config classes were renamed between torchvision 0.24 and 0.25:
#   0.23 / 0.24 : EfficientNetBlockConfig / FusedEfficientNetBlockConfig
#   0.25+       : MBConvConfig            / FusedMBConvConfig
try:  # torchvision >= 0.25
    from torchvision.models.efficientnet import MBConvConfig as _MBConv
    from torchvision.models.efficientnet import FusedMBConvConfig as _FusedMBConv
except ImportError:  # torchvision 0.23 - 0.24
    from torchvision.models.efficientnet import EfficientNetBlockConfig as _MBConv
    from torchvision.models.efficientnet import FusedEfficientNetBlockConfig as _FusedMBConv


# ---------------------------------------------------------------------------
# Custom (ultra-tiny) EfficientNet variants
# ---------------------------------------------------------------------------
#
# Hand-sized block topologies that fit strict parameter budgets (the 100k / 250k
# custom variants). They are built with the exact same EfficientNet building
# blocks as the stock models (MBConv / FusedMBConv + SE), so they are compatible
# with the standard _adapt_efficientnet / _get_output_dim helpers.
#
# All four variants have NO stock ImageNet weights (they are random-init).
#
#   v1_100k  – <= 100k  params – EfficientNetV1 blocks (MBConv)                78,680 params
#   v1_250k  – <= 250k  params – EfficientNetV1 blocks (MBConv)               193,408 params
#   v2_100k  – <= 100k  params – EfficientNetV2 blocks (FusedMBConv + MBConv)  91,140 params
#   v2_250k  – <= 250k  params – EfficientNetV2 blocks (FusedMBConv + MBConv) 207,784 params


_CUSTOM_EFFNET_CONFIGS = {
    "v1_100k": (
        [
            _MBConv(1, 3, 2, 32, 48, 1),
            _MBConv(2, 3, 2, 48, 64, 1),
            _MBConv(3, 3, 2, 64, 80, 1),
            _MBConv(1, 3, 1, 80, 96, 1),
        ],
        128,
    ),
    "v1_250k": (
        [
            _MBConv(1, 3, 2, 32, 48, 1),
            _MBConv(3, 3, 2, 48, 80, 1),
            _MBConv(4, 3, 2, 80, 96, 1),
            _MBConv(2, 3, 1, 96, 128, 1),
        ],
        288,
    ),
    "v2_100k": (
        [
            _FusedMBConv(1, 3, 2, 32, 48, 1),
            _FusedMBConv(1, 3, 2, 48, 64, 1),
            _MBConv(2, 3, 2, 64, 80, 1),
            _MBConv(1, 3, 1, 80, 96, 1),
        ],
        128,
    ),
    "v2_250k": (
        [
            _FusedMBConv(1, 3, 2, 32, 48, 1),
            _FusedMBConv(2, 3, 2, 48, 64, 1),
            _MBConv(4, 3, 2, 64, 96, 1),
            _MBConv(2, 3, 1, 96, 128, 1),
        ],
        288,
    ),
}

def _build_custom_efficientnet(variant: str) -> nn.Module:
    """Build a custom (ultra-tiny) EfficientNet variant.

    Args:
        variant: one of 'v1_100k', 'v1_250k', 'v2_100k', 'v2_250k'.

    Returns:
        An EfficientNet instance with a custom block topology and a small
        output channel count. The classifier will be replaced by Identity
        during the ADAPT step.
    """
    if variant not in _CUSTOM_EFFNET_CONFIGS:
        raise ValueError(
            f"Unknown custom EfficientNet variant: '{variant}'. "
            f"Supported: {', '.join(sorted(_CUSTOM_EFFNET_CONFIGS))}"
        )
    setting, last_channel = _CUSTOM_EFFNET_CONFIGS[variant]
    return EfficientNet(
        inverted_residual_setting=setting,
        dropout=0.2,
        stochastic_depth_prob=0.0,
        num_classes=1,  # replaced by Identity in _adapt_efficientnet
        last_channel=last_channel,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_weights(pretrained):
    """Map a `pretrained` flag to torchvision's `weights` argument.

    Accepts True/False (legacy boolean) or a torchvision weights string such
    as "DEFAULT" / "IMAGENET1K_V1".
    """
    if isinstance(pretrained, bool):
        return "DEFAULT" if pretrained else None
    return pretrained


def _timm_pretrained(pretrained):
    """Map a `pretrained` flag to timm's `pretrained` argument (bool or cfg str)."""
    if pretrained is None:
        return False
    return pretrained


def _make_adapted_conv(original_conv: nn.Conv2d, in_channels: int = 1) -> nn.Conv2d:
    """Create a replacement Conv2d with `in_channels` input channels.

    When converting 3 colour channels to 1 grayscale channel the new weights
    are the channel-mean of the original (RGB) weights, so the grayscale conv
    reproduces the average behaviour of the pretrained colour conv (this is
    the exact weight initialisation used by the legacy code).

    Args:
        original_conv: the (colour) conv layer to replace.
        in_channels: number of input channels for the new layer (1 = grayscale).

    Returns:
        nn.Conv2d with identical spatial parameters (kernel/stride/padding, no bias).
    """
    new_conv = nn.Conv2d(
        in_channels,
        original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False,
    )
    if original_conv.weight is not None:
        with torch.no_grad():
            if original_conv.in_channels == in_channels:
                new_conv.weight = nn.Parameter(original_conv.weight.detach().clone())
            else:
                new_conv.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
    return new_conv


def freeze_batchnorm(model: nn.Module) -> int:
    """Freeze all BatchNorm layers in `model`.

    A "frozen" BatchNorm is put in eval() mode while *keeping*
    ``track_running_stats`` enabled, so it normalises with its pretrained
    running statistics (deterministic during both training and inference).
    eval() mode already blocks any update to ``running_mean`` / ``running_var``,
    so the pretrained statistics cannot drift while the encoder is frozen.

    Do NOT disable tracking here: PyTorch picks the normalisation statistics via
    ``training = self.training or not self.track_running_stats``, so a frozen BN
    with tracking disabled would fall back to *batch* statistics (batch-size
    dependent and wrong at inference). To re-estimate the running statistics on
    new data instead, keep BN in train() mode via the fine-tuning flag
    ``update_batchnorm_stats_*`` (AdaBN).

    Returns:
        int: number of BatchNorm layers that were frozen.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            module.track_running_stats = True
            count += 1
    return count


def unfreeze_batchnorm(*modules: nn.Module) -> int:
    """Restore running-stat tracking on all BatchNorm layers in `modules`.

    Inverse of :func:`freeze_batchnorm`: re-enables ``track_running_stats``
    so that an encoder being (partially) fine-tuned can update its
    statistics again.  The affine parameters of the unfrozen modules are
    also re-enabled (``requires_grad=True``) — callers that froze them
    during full-encoder freezing must not rely on this for other modules.

    Returns:
        int: number of BatchNorm layers that were unfrozen.
    """
    count = 0
    for root in modules:
        for module in root.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.track_running_stats = True
                for param in module.parameters(recurse=False):
                    param.requires_grad = True
                count += 1
    return count


def _get_output_dim(family: str, model: nn.Module) -> int:
    """Feature dimension of a backbone, read from the actual module structure."""
    if family in ("efficientnet", "mobilenet_v3"):
        return model.features[-1][0].out_channels
    if family == "fastvit":
        return model.final_conv.conv_scale.conv.out_channels
    if family == "shufflenet_v2":
        return model.conv5[0].out_channels
    if family == "mnasnet":
        for layer in reversed(model.layers):
            if hasattr(layer, "out_channels"):
                return layer.out_channels
        raise ValueError("Could not determine MNASNet output dimension.")
    raise ValueError(
        f"Unknown backbone family: '{family}'. "
        f"Supported: {', '.join(sorted(BACKBONE_REGISTRY))}"
    )


def _resolve_output_dim(family: str, info: Dict[str, Any], model: nn.Module) -> int:
    """Feature dimension of a directly-built backbone (register_backbone).

    Uses the registry entry's ``get_output_dim`` callable when present,
    otherwise the built module's own ``get_output_dim()`` method.
    """
    getter = info.get("get_output_dim")
    if getter is not None:
        return int(getter(model))
    module_getter = getattr(model, "get_output_dim", None)
    if callable(module_getter):
        return int(module_getter())
    raise ValueError(
        f"Backbone family '{family}': the module returned by the custom "
        "builder has no get_output_dim(). Either give the module a "
        "get_output_dim() method or pass get_output_dim= to register_backbone()."
    )


# ---------------------------------------------------------------------------
# Variant selection (step 1) and grayscale adaptation (step 2)
# ---------------------------------------------------------------------------

def _normalize_variant(family: str, variant: Optional[str]) -> Optional[str]:
    """Normalise a variant to its bare form (e.g. 'fastvit_t8' -> 't8').

    Returns None when `variant` is None so the caller can apply a default.
    """
    if variant is None:
        return None
    variant = str(variant).strip().lower()
    for prefix in BACKBONE_REGISTRY[family]["name_prefixes"]:
        if variant.startswith(prefix):
            return variant[len(prefix):]
    return variant


def _validate_variant(family: str, variant: str) -> None:
    """Fail fast when a backbone family or variant is unknown."""
    if family not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone family: '{family}'. "
            f"Supported: {', '.join(sorted(BACKBONE_REGISTRY))}"
        )
    variants = BACKBONE_REGISTRY[family]["variants"]
    if variant not in variants:
        raise ValueError(
            f"Unknown {family} variant: '{variant}'. "
            f"Supported variants: {', '.join(variants)}"
        )


def _custom_mobilenet_v3_conf(
    arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False, **kwargs: Any
    ):
    """
    Internal helper function to generate the configuration for our custom MobileNetV3.
    """
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_tiny_v2":
        width_mult = 0.5
        bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
        adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)
        
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),
        ]
        
        last_channel = 512
    else:
        raise ValueError(f"Unsupported custom model type {arch}")

    return inverted_residual_setting, last_channel


def _select_efficientnet(variant: str, pretrained) -> nn.Module:
    """Step 1 (EfficientNet): build the stock or custom model for a variant.

    Stock variants (b0-b7, v2_s/m/l) use torchvision's pretrained builders.
    Custom variants (v1_100k, v1_250k, v2_100k, v2_250k) have no pretrained
    weights and are always randomly initialised.
    """
    builders = {
        "b0": models.efficientnet_b0,
        "b1": models.efficientnet_b1,
        "b2": models.efficientnet_b2,
        "b3": models.efficientnet_b3,
        "b4": models.efficientnet_b4,
        "b5": models.efficientnet_b5,
        "b6": models.efficientnet_b6,
        "b7": models.efficientnet_b7,
        "v2_s": models.efficientnet_v2_s,
        "v2_m": models.efficientnet_v2_m,
        "v2_l": models.efficientnet_v2_l,
    }
    if variant in builders:
        return builders[variant](weights=_normalize_weights(pretrained))
    # Custom (ultra-tiny) variant – no pretrained weights available.
    if variant in _CUSTOM_EFFNET_CONFIGS:
        if pretrained:
            logger.warning(
                "Custom EfficientNet variant '%s' has no stock ImageNet weights; "
                "ignoring pretrained=%r and randomly initialising.",
                variant, pretrained,
            )
        return _build_custom_efficientnet(variant)
    raise ValueError(
        f"Unknown EfficientNet variant: '{variant}'. "
        f"Supported stock: b0-b7, v2_s, v2_m, v2_l. "
        f"Supported custom: v1_100k, v1_250k, v2_100k, v2_250k."
    )


def _select_mobilenet_v3(variant: str, pretrained) -> nn.Module:
    """Step 1 (MobileNetV3): build the stock model for a variant.

    NOTE: 'tiny' uses a custom "tiny v2" topology that has no stock ImageNet
    weights, so it is always randomly initialised.
    """
    if variant == "tiny":
        setting, last_channel = _custom_mobilenet_v3_conf("mobilenet_v3_tiny_v2")
        return MobileNetV3(setting, last_channel)
    if variant == "small":
        return models.mobilenet_v3_small(weights=_normalize_weights(pretrained))
    return models.mobilenet_v3_large(weights=_normalize_weights(pretrained))


def _select_fastvit(variant: str, pretrained) -> nn.Module:
    """Step 1 (FastViT): build the stock timm model for a variant."""
    return timm.create_model(f"fastvit_{variant}", pretrained=_timm_pretrained(pretrained))


def _select_shufflenet_v2(variant: str, pretrained) -> nn.Module:
    """Step 1 (ShuffleNetV2): build the stock model for a variant."""
    builders = {
        "x0_5": models.shufflenet_v2_x0_5,
        "x1_0": models.shufflenet_v2_x1_0,
        "x1_5": models.shufflenet_v2_x1_5,
        "x2_0": models.shufflenet_v2_x2_0,
    }
    return builders[variant](weights=_normalize_weights(pretrained))


def _select_mnasnet(variant: str, pretrained) -> nn.Module:
    """Step 1 (MNASNet): build the stock model for a variant."""
    builders = {
        "0_5": models.mnasnet0_5,
        "0_75": models.mnasnet0_75,
        "1_0": models.mnasnet1_0,
        "1_3": models.mnasnet1_3,
    }
    return builders[variant](weights=_normalize_weights(pretrained))


def _adapt_efficientnet(model: nn.Module, in_channels: int = 1) -> nn.Module:
    """Step 2 (EfficientNet): grayscale first conv + feature-output head."""
    model.features[0][0] = _make_adapted_conv(model.features[0][0], in_channels)
    model.classifier = nn.Identity()
    return model


def _adapt_mobilenet_v3(model: nn.Module, in_channels: int = 1) -> nn.Module:
    """Step 2 (MobileNetV3): grayscale first conv + feature-output head."""
    model.features[0][0] = _make_adapted_conv(model.features[0][0], in_channels)
    model.classifier = nn.Identity()
    return model


def _adapt_fastvit(model: nn.Module, in_channels: int = 1) -> nn.Module:
    """Step 2 (FastViT): grayscale both stem convs + feature-output head."""
    model.stem[0].conv_kxk[0].conv = _make_adapted_conv(model.stem[0].conv_kxk[0].conv, in_channels)
    model.stem[0].conv_scale.conv = _make_adapted_conv(model.stem[0].conv_scale.conv, in_channels)
    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
    )
    return model


def _adapt_shufflenet_v2(model: nn.Module, in_channels: int = 1) -> nn.Module:
    """Step 2 (ShuffleNetV2): grayscale first conv + feature-output head.

    `conv1` and `conv5` are nn.Sequential containers whose first module is
    the Conv2d (conv1: stem, conv5: final feature conv before the classifier).
    """
    model.conv1[0] = _make_adapted_conv(model.conv1[0], in_channels)
    model.fc = nn.Identity()
    return model


def _adapt_mnasnet(model: nn.Module, in_channels: int = 1) -> nn.Module:
    """Step 2 (MNASNet): grayscale first conv + feature-output head.

    `model.layers` is a flat list; `layers[0]` is the first Conv2d. The
    classifier (Dropout + Linear) is replaced by a pure feature output.
    """
    model.layers[0] = _make_adapted_conv(model.layers[0], in_channels)
    model.classifier = nn.Identity()
    return model


# ---------------------------------------------------------------------------
# Registry — single source of truth for families, variants and steps
# ---------------------------------------------------------------------------

BACKBONE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "efficientnet": {
        "variants": (
            "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7",
            "v2_s", "v2_m", "v2_l",
            # custom (ultra-tiny, random-init) variants:
            "v1_100k", "v1_250k", "v2_100k", "v2_250k",
        ),
        "default_variant": "v2_s",
        "name_prefixes": ("efficientnet_",),
        "select": _select_efficientnet,
        "adapt": _adapt_efficientnet,
    },
    "mobilenet_v3": {
        "variants": ("tiny", "small", "large"),
        "default_variant": "large",
        "name_prefixes": ("mobilenet_v3_", "mobilenet_"),
        "select": _select_mobilenet_v3,
        "adapt": _adapt_mobilenet_v3,
    },
    "fastvit": {
        "variants": ("t8", "t12", "s12"),
        "default_variant": "t8",
        "name_prefixes": ("fastvit_",),
        "select": _select_fastvit,
        "adapt": _adapt_fastvit,
    },
    "shufflenet_v2": {
        "variants": ("x0_5", "x1_0", "x1_5", "x2_0"),
        "default_variant": "x1_0",
        "name_prefixes": ("shufflenet_v2_", "shufflenetv2_"),
        "select": _select_shufflenet_v2,
        "adapt": _adapt_shufflenet_v2,
    },
    "mnasnet": {
        "variants": ("0_5", "0_75", "1_0", "1_3"),
        "default_variant": "1_0",
        "name_prefixes": ("mnasnet_", "mnasnet"),
        "select": _select_mnasnet,
        "adapt": _adapt_mnasnet,
    },
}


def register_backbone(family: str, *, build=None, select=None, adapt=None,
                      get_output_dim=None, variants=("default",),
                      default_variant=None, name_prefixes=None,
                      override: bool = False) -> dict:
    """Register a backbone family so configs can select it by name.

    Two flavours are supported:

      * **Recipe** (same as the built-in families): pass ``select`` and
        ``adapt`` — the wrapper selects a stock model and adapts it in place.
      * **Direct builder**: pass ``build(variant, pretrained, in_channels)``
        returning a finished feature extractor that exposes
        ``get_output_dim()`` (or pass ``get_output_dim=`` alongside the
        builder). This is the easy route for hand-written networks — see
        ``models/custom_models.py`` for a complete template.

    Once registered, ``image_tower.model_name: <family>_<variant>`` resolves
    through ``GrayscaleBackbone`` exactly like the built-in families (the
    config name is matched against the registry's ``name_prefixes``).

    Args:
        family: registry key, e.g. ``'mycnn'`` for ``mycnn_small``.
        build: factory ``(variant, pretrained, in_channels) -> nn.Module``.
        select: recipe selector ``(variant, pretrained) -> nn.Module``.
        adapt: recipe adapter ``(model, in_channels) -> nn.Module``.
        get_output_dim: optional ``(model) -> int``; required only when the
            built module has no ``get_output_dim()`` method.
        variants: accepted variant names.
        default_variant: variant used when none is given; defaults to the
            first entry of ``variants``.
        name_prefixes: config-name prefixes stripped to recover the variant;
            defaults to ``(family + "_",)``.
        override: replace an existing family (default: refuse).

    Returns:
        dict: the created registry entry (also stored in BACKBONE_REGISTRY).

    Raises:
        ValueError: on invalid arguments or an already-registered family
            without ``override=True``.
    """
    family = str(family).strip().lower()
    if not family:
        raise ValueError("register_backbone: 'family' must be a non-empty string.")
    if family in BACKBONE_REGISTRY and not override:
        raise ValueError(
            f"Backbone family '{family}' is already registered. "
            "Pass override=True to replace it."
        )
    if build is None and (select is None or adapt is None):
        raise ValueError(
            "register_backbone: provide 'build', or both 'select' and 'adapt'."
        )
    variant_names = tuple(str(v).strip().lower() for v in variants)
    if not variant_names or not all(variant_names):
        raise ValueError("register_backbone: 'variants' must be a non-empty "
                         "sequence of non-empty names.")
    if default_variant is None:
        default_variant = variant_names[0]
    if default_variant not in variant_names:
        raise ValueError(
            f"register_backbone: default_variant '{default_variant}' is not in "
            f"variants {variant_names}."
        )

    entry = {
        "variants": variant_names,
        "default_variant": default_variant,
        "name_prefixes": tuple(name_prefixes) if name_prefixes else (family + "_",),
    }
    if build is not None:
        entry["build"] = build
    else:
        entry["select"] = select
        entry["adapt"] = adapt
    if get_output_dim is not None:
        entry["get_output_dim"] = get_output_dim

    BACKBONE_REGISTRY[family] = entry
    logger.info("Registered backbone family '%s' (variants: %s)",
                family, ", ".join(variant_names))
    return entry


# ---------------------------------------------------------------------------
# Step 3: thin wrappers
# ---------------------------------------------------------------------------

class _GrayscaleBackboneBase(nn.Module):
    """Shared behaviour for all backbone wrappers (select -> adapt -> expose).

    The wrapper attribute is always named `model`: that name is baked into
    every saved state_dict (e.g. `img_encoder.model.features.0.0.weight`) and
    must not be changed.
    """
    family: Optional[str] = None  # set by subclasses; key into BACKBONE_REGISTRY

    def __init__(self, variant=None, pretrained=False, freeze_bn=False, in_channels: int = 1):
        super().__init__()
        if self.family is None:
            raise TypeError(f"{type(self).__name__} must set the `family` attribute.")
        if self.family not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone family: '{self.family}'. "
                f"Available families: {', '.join(sorted(BACKBONE_REGISTRY))}."
            )
        info = BACKBONE_REGISTRY[self.family]
        variant = _normalize_variant(self.family, variant) or info["default_variant"]
        _validate_variant(self.family, variant)
        if "build" in info:
            # Direct-builder entry (custom family added with register_backbone):
            # the factory returns the finished feature extractor.
            self.model = info["build"](variant=variant, pretrained=pretrained,
                                       in_channels=in_channels)
            self.feature_outsize = _resolve_output_dim(self.family, info, self.model)
        else:
            self.model = info["adapt"](info["select"](variant, pretrained), in_channels)
            self.feature_outsize = _get_output_dim(self.family, self.model)
        if freeze_bn:
            self.freeze_batchnorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_output_dim(self) -> int:
        """Feature dimension of the backbone (as consumed by the encoders)."""
        return self.feature_outsize

    def freeze_batchnorm(self) -> int:
        """Freeze all BatchNorm layers; returns the number of layers frozen."""
        return freeze_batchnorm(self.model)


class GrayscaleBackbone(_GrayscaleBackboneBase):
    """Family-agnostic backbone constructor.

    Example:
        GrayscaleBackbone("efficientnet", "b0", pretrained=True)
    """

    def __init__(self, family, variant=None, pretrained=False, freeze_bn=False, in_channels: int = 1):
        if family not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone family: '{family}'. "
                f"Supported: {', '.join(sorted(BACKBONE_REGISTRY))}"
            )
        self.family = family
        super().__init__(variant=variant, pretrained=pretrained,
                         freeze_bn=freeze_bn, in_channels=in_channels)


def build_grayscale_backbone(family, variant=None, pretrained=False,
                             freeze_bn=False, in_channels: int = 1) -> nn.Module:
    """Convenience function: build a grayscale backbone by family + variant."""
    return GrayscaleBackbone(family, variant, pretrained=pretrained,
                             freeze_bn=freeze_bn, in_channels=in_channels)


# ---------------------------------------------------------------------------
# Public per-family classes (stable names: they appear in experiment docs and
# are imported by ssl_models.py / user code).
# ---------------------------------------------------------------------------

class EfficientNet_grayscale(_GrayscaleBackboneBase):
    """EfficientNet backbone adapted for single-channel (grayscale) input.

    Args:
        model_type: one of 'b0', 'b1', 'v2_s', 'v2_m' (default 'v2_s').
            `model_type` is the legacy name of the `variant` argument.
        pretrained: True/False or a torchvision weights string ("DEFAULT",
            "IMAGENET1K_V1", ...).
        freeze_bn: freeze BatchNorm layers (eval + no running-stats updates).
        in_channels: input channels (1 = grayscale, 3 = RGB).
        variant: alias of model_type (new-style name).
    """
    family = "efficientnet"

    def __init__(self, model_type="v2_s", pretrained=False, freeze_bn=False,
                 in_channels=1, variant=None):
        super().__init__(variant=variant or model_type, pretrained=pretrained,
                         freeze_bn=freeze_bn, in_channels=in_channels)


class MobileNetV3_grayscale(_GrayscaleBackboneBase):
    """MobileNetV3 backbone adapted for single-channel (grayscale) input.

    Args:
        pretrained: True/False or a torchvision weights string.
            NOTE: the custom 'tiny' variant has no stock weights and is
            randomly initialised.
        model_type: one of 'tiny', 'small', 'large' (default 'large').
            `model_type` is the legacy name of the `variant` argument.
        freeze_bn: freeze BatchNorm layers (eval + no running-stats updates).
        in_channels: input channels (1 = grayscale, 3 = RGB).
        variant: alias of model_type (new-style name).
    """
    family = "mobilenet_v3"

    def __init__(self, pretrained=False, model_type="large", freeze_bn=False,
                 in_channels=1, variant=None):
        super().__init__(variant=variant or model_type, pretrained=pretrained,
                         freeze_bn=freeze_bn, in_channels=in_channels)


class fastViT_grayscale(_GrayscaleBackboneBase):
    """FastViT backbone adapted for single-channel (grayscale) input.

    Args:
        model_name: one of 'fastvit_t8', 'fastvit_t12', 'fastvit_s8',
            'fastvit_s12' — or just the variant ('t8', ...) (default 't8').
        pretrained: True/False or a timm pretrained-cfg string.
        freeze_bn: freeze BatchNorm layers (eval + no running-stats updates).
        in_channels: input channels (1 = grayscale, 3 = RGB).
        variant: alias of model_name (new-style name).
    """
    family = "fastvit"

    def __init__(self, model_name="fastvit_t8", pretrained=False, freeze_bn=False,
                 in_channels=1, variant=None):
        super().__init__(variant=variant or model_name, pretrained=pretrained,
                         freeze_bn=freeze_bn, in_channels=in_channels)


class ShuffleNetV2_grayscale(_GrayscaleBackboneBase):
    """ShuffleNetV2 backbone adapted for single-channel (grayscale) input.

    Args:
        model_name: one of 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
            'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0' — or just the
            variant ('x1_0', ...) (default 'x1_0').
        pretrained: True/False or a torchvision weights string.
        freeze_bn: freeze BatchNorm layers (eval + no running-stats updates).
        in_channels: input channels (1 = grayscale, 3 = RGB).
        variant: alias of model_name (new-style name).
    """
    family = "shufflenet_v2"

    def __init__(self, model_name="shufflenet_v2_x1_0", pretrained=False,
                 freeze_bn=False, in_channels=1, variant=None):
        super().__init__(variant=variant or model_name, pretrained=pretrained,
                         freeze_bn=freeze_bn, in_channels=in_channels)


class MNASNet_grayscale(_GrayscaleBackboneBase):
    """MNASNet backbone adapted for single-channel (grayscale) input.

    Args:
        model_name: one of 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0',
            'mnasnet1_3' — or just the variant ('1_0', ...) (default '1_0').
        pretrained: True/False or a torchvision weights string.
        freeze_bn: freeze BatchNorm layers (eval + no running-stats updates).
        in_channels: input channels (1 = grayscale, 3 = RGB).
        variant: alias of model_name (new-style name).
    """
    family = "mnasnet"

    def __init__(self, model_name="mnasnet1_0", pretrained=False,
                 freeze_bn=False, in_channels=1, variant=None):
        super().__init__(variant=variant or model_name, pretrained=pretrained,
                         freeze_bn=freeze_bn, in_channels=in_channels)


__all__ = [
    # Shared helpers
    "_make_adapted_conv",
    "freeze_batchnorm",
    # Registry
    "BACKBONE_REGISTRY",
    "register_backbone",
    # Generic
    "GrayscaleBackbone",
    "build_grayscale_backbone",
    # Public per-family wrappers (stable names — part of saved state_dicts)
    "EfficientNet_grayscale",
    "MobileNetV3_grayscale",
    "fastViT_grayscale",
    "ShuffleNetV2_grayscale",
    "MNASNet_grayscale",
]