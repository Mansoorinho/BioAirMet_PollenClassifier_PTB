'''
@file    :   custom_models.py
@author  :   Mansoor Nabawi
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    Templates for custom encoders — copy one, modify it, select it from the
    config. This module ships two small but complete reference encoders and
    registers them with the model factories:

      * SimpleCnn_grayscale  ->  image_tower.model_name: mycnn_small | mycnn_wide
      * FluorescenceCNN      ->  fluorescence_tower.model_name: FluorescenceCNN

    Both work in BOTH training stages (Stage 1 trains the encoders
    contrastively, Stage 2 reuses them under the classification head).

    The encoder contract (everything the trainers and builders require):

      1. An nn.Module whose forward(x) returns a (B, D) float tensor, where
         x is (B, 1, H, W) for image towers and (B, input_dim) for
         fluorescence towers, and D is fixed for a given instance.
      2. A get_output_dim() method returning D — SSLModel_SingleIMG and
         HoloClassifierV2 size their fusion layer from it.
      3. Nothing else: freezing, "unfreeze last N layers" and the
         BatchNorm switches walk the module tree generically, so plain
         PyTorch modules just work.

    Checkpoint note: class names and module paths are stored in saved
    state_dicts. Renaming a custom class or moving its file makes existing
    checkpoints unloadable — pick a name and keep it.

    To add your own encoder:
      * image backbone:  define the module, write a builder
        (variant, pretrained, in_channels) -> module, call register_backbone()
        (see the bottom of this file), and reference it as
        image_tower.model_name: <family>_<variant>.
      * fluorescence encoder: define the module, write a builder
        (fluorescence_tower config) -> module, call
        register_fluorescence_encoder(), and reference it as
        fluorescence_tower.model_name: <name>.

    This module is imported automatically by bioairmet.models, so everything
    registered here (or added to this file in an editable install) is
    available to the CLI entry points without extra wiring.
    Full guide: docs/custom_models.md.
    ]
'''

import logging
from typing import Sequence

import torch
import torch.nn as nn

from .backbones import register_backbone
from .fluorescence_encoders import register_fluorescence_encoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image-domain template: a small fully-convolutional grayscale backbone
# ---------------------------------------------------------------------------

class SimpleCnn_grayscale(nn.Module):
    """Small Conv -> BatchNorm -> ReLU stack for grayscale holographic images.

    Each block halves the spatial resolution (stride 2) except the last one;
    a global average pool produces the (B, channels[-1]) feature vector. The
    point of this template is the interface, not the accuracy — replace the
    block, the depth or the pooling with whatever your data needs, and keep
    forward()/get_output_dim() as the contract.

    Args:
        channels: channel widths of the conv blocks (one block per entry).
        in_channels: input channels (1 = grayscale, 3 = RGB).
    """

    def __init__(self, channels: Sequence[int] = (16, 32, 64, 128),
                 in_channels: int = 1):
        super().__init__()
        if len(channels) < 2:
            raise ValueError(
                f"SimpleCnn_grayscale needs at least two channel entries, got {channels}"
            )
        blocks = []
        prev = in_channels
        last = len(channels) - 1
        for i, ch in enumerate(channels):
            blocks += [
                nn.Conv2d(prev, ch, kernel_size=3,
                          stride=1 if i == last else 2, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ]
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self._output_dim = int(channels[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Grayscale images (B, 1, H, W) -> features (B, output_dim)."""
        return self.pool(self.features(x)).flatten(1)

    def get_output_dim(self) -> int:
        """Feature dimension (fixed for a given instance)."""
        return self._output_dim


# Variant table: config name suffix -> channel widths.
_MYCNN_VARIANTS = {
    "small": (16, 32, 64, 128),
    "wide": (32, 64, 128, 256),
}


def _build_mycnn(variant: str, pretrained, in_channels: int) -> nn.Module:
    """Registry builder for the 'mycnn' family (template for your own)."""
    if variant not in _MYCNN_VARIANTS:
        raise ValueError(
            f"Unknown mycnn variant: '{variant}'. Supported: {', '.join(_MYCNN_VARIANTS)}"
        )
    if pretrained:
        logger.warning("SimpleCnn_grayscale has no pretrained weights; "
                       "starting from a random initialisation.")
    return SimpleCnn_grayscale(channels=_MYCNN_VARIANTS[variant],
                               in_channels=in_channels)


register_backbone(
    "mycnn",
    build=_build_mycnn,
    variants=tuple(_MYCNN_VARIANTS),
    default_variant="small",
    name_prefixes=("mycnn_",),
)


# ---------------------------------------------------------------------------
# Fluorescence-domain template: a 1-D convolutional spectrum encoder
# ---------------------------------------------------------------------------

class FluorescenceCNN(nn.Module):
    """1-D convolutional encoder for fluorescence spectra.

    The spectrum (B, input_dim) is treated as a single-channel 1-D signal and
    passed through Conv1d -> BatchNorm1d -> ReLU blocks, a global average pool
    and one linear layer. Compared to the MLP encoders this shares weights
    across neighbouring wavelengths, which suits inputs where the local
    spectral shape (peaks, shoulders) carries the signal.

    Args:
        input_dim: number of spectral features (accepted for config parity;
            the conv stack itself is length-agnostic thanks to the global
            pool).
        output_dim: embedding dimension.
        channels: channel widths of the conv blocks.
        kernel_size: conv kernel width in spectral bins.
        dropout: dropout probability before the output projection.
    """

    def __init__(self, input_dim: int, output_dim: int = 256,
                 channels: Sequence[int] = (32, 64), kernel_size: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.output_dim = int(output_dim)

        blocks = []
        prev = 1
        for ch in channels:
            blocks += [
                nn.Conv1d(prev, ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
            ]
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(prev, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fluorescence features (B, F) -> embedding (B, output_dim)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B, F) -> (B, 1, F)
        return self.head(self.features(x))

    def get_output_dim(self) -> int:
        """Embedding dimension."""
        return self.output_dim


def _build_fluorescence_cnn(fluo) -> nn.Module:
    """Registry builder reading the fluorescence_tower config section.

    Optional keys (beyond the standard input_dim / output_dim):
        channels (list), kernel_size (int), dropout (float).
    """
    return FluorescenceCNN(
        input_dim=int(fluo.input_dim),
        output_dim=int(fluo.get("output_dim", 256)),
        channels=tuple(fluo.get("channels", (32, 64))),
        kernel_size=int(fluo.get("kernel_size", 5)),
        dropout=float(fluo.get("dropout", 0.1)),
    )


register_fluorescence_encoder("FluorescenceCNN", _build_fluorescence_cnn)


__all__ = [
    "SimpleCnn_grayscale",
    "FluorescenceCNN",
]