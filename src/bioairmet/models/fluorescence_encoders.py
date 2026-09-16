'''
@file    :   fluorescence_encoders.py
@create date : 2025-06-05 10:26:08
@modify date 2026-08-25
@author  :   Mansoor Nabawi
@version :   2.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    Fluorescence-spectra encoders (MLP) for the BioAirMet project.

    FluorescenceMLP is the single configurable encoder: hidden-layer list,
    dropout, activation ('relu' | 'gelu' | 'silu') and optional normalization
    ('bn' | 'ln' | None). FluorescenceMLP_Simple is a thin alias that keeps
    the legacy "lightweight" behaviour (ReLU only, no norm, no dropout).
    Both expose get_output_dim() for use in multimodal fusion.

    Custom encoders are added with register_fluorescence_encoder(name, builder)
    — the builder receives the fluorescence_tower config section and returns a
    module with forward() and get_output_dim(); see models/custom_models.py
    for a worked template.
    ]
'''
import logging
from typing import Any, Callable, Dict

import torch.nn as nn
import torch

logger = logging.getLogger(__name__)

# Supported activation / normalization lookups (single place to extend).
_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}

_NORMS = {
    "ln": nn.LayerNorm,
    "bn": nn.BatchNorm1d,
}


def _get_activation(activation: str) -> nn.Module:
    """Return an activation module by name; fails fast with the supported list."""
    try:
        return _ACTIVATIONS[activation]()
    except KeyError:
        raise ValueError(
            f"Unknown activation: '{activation}'. "
            f"Supported: {', '.join(_ACTIVATIONS)}"
        ) from None


def _get_norm(norm: str):
    """Return the normalization class by name; fails fast with the supported list."""
    try:
        return _NORMS[norm]
    except KeyError:
        raise ValueError(
            f"Unknown norm: '{norm}'. "
            f"Supported: {', '.join(_NORMS)} (or None to disable)"
        ) from None


class FluorescenceMLP(nn.Module):
    """Multi-Layer Perceptron encoder for fluorescence spectra.

    Structure (per hidden layer):  Linear -> [Norm (first layer only)] -> Act -> Dropout
    Final layer:                  Linear -> Act -> Dropout

    Signature and defaults match the legacy class exactly, so existing
    configs/checkpoints keep working:
        (input_dim, output_dim, dropout, hidden_dims, activation, norm)

    Args:
        input_dim: number of input fluorescence features.
        output_dim: embedding dimension of the encoder.
        dropout: dropout probability (default 0.2).
        hidden_dims: list of hidden-layer widths (default [128]).
        activation: 'relu' | 'gelu' | 'silu' (default 'relu').
        norm: 'bn' | 'ln' (default 'ln') — normalization on the first hidden
            layer (pass None to disable).
        use_dropout: add Dropout modules after each hidden layer (default
            True; False keeps the legacy 'Simple' layout exactly).
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2,
                 hidden_dims=(128,), activation: str = "relu", norm="ln",
                 use_dropout: bool = True):
        super().__init__()
        self.output_dim = output_dim

        act = _get_activation(activation)
        norm_cls = _get_norm(norm) if norm is not None else None

        layers = []
        current_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, h_dim))
            if i == 0 and norm_cls is not None:
                layers.append(norm_cls(h_dim))
            layers.append(act)
            if use_dropout:
                layers.append(nn.Dropout(dropout))
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(act)
        if use_dropout:
            layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fluorescence features (B, F) -> embedding (B, output_dim)."""
        return self.encoder(x)

    def get_output_dim(self) -> int:
        """Output feature dimension of the encoder."""
        return self.output_dim


class FluorescenceMLP_Simple(FluorescenceMLP):
    """Lightweight MLP: ReLU activations, no normalization, no dropout.

    Kept as a separate class so configs written for the legacy name
    ('FluorescenceMLP_Simple') keep working unchanged.
    """

    def __init__(self, input_dim: int, output_dim: int = 64, hidden_dims=(32,)):
        super().__init__(
            input_dim,
            output_dim=output_dim,
            dropout=0.0,
            hidden_dims=list(hidden_dims),
            activation="relu",
            norm=None,
            use_dropout=False,
        )


# ---------------------------------------------------------------------------
# Encoder registry — maps fluorescence_tower.model_name to a builder
# ---------------------------------------------------------------------------

FLUORESCENCE_ENCODER_REGISTRY: Dict[str, Callable[[Any], nn.Module]] = {}


def register_fluorescence_encoder(name: str, builder: Callable[[Any], nn.Module],
                                  override: bool = False) -> None:
    """Register a fluorescence encoder so configs can select it by name.

    The builder is called with the ``architecture_setup.fluorescence_tower``
    config section and must return an ``nn.Module`` with ``forward(x)`` for
    (B, input_dim) spectra and ``get_output_dim()`` (the fusion layer sizes
    itself from it). Select the encoder with
    ``fluorescence_tower.model_name: <name>`` — see ``models/custom_models.py``
    for a complete template.

    Args:
        name: config name of the encoder, e.g. 'FluorescenceCNN'.
        builder: callable (fluorescence_tower config) -> nn.Module.
        override: replace an existing name (default: refuse).

    Raises:
        ValueError: on a non-callable builder, an empty name, or an
            already-registered name without override=True.
    """
    name = str(name).strip()
    if not name:
        raise ValueError("register_fluorescence_encoder: 'name' must be non-empty.")
    if not callable(builder):
        raise ValueError(
            f"register_fluorescence_encoder: 'builder' must be callable, got {type(builder).__name__}."
        )
    if name in FLUORESCENCE_ENCODER_REGISTRY and not override:
        raise ValueError(
            f"Fluorescence encoder '{name}' is already registered. "
            "Pass override=True to replace it."
        )
    FLUORESCENCE_ENCODER_REGISTRY[name] = builder
    logger.info("Registered fluorescence encoder '%s'", name)


def _build_fluorescence_mlp(fluo) -> nn.Module:
    """Built-in builder for the configurable FluorescenceMLP."""
    return FluorescenceMLP(
        input_dim=fluo.input_dim,
        output_dim=fluo.output_dim,
        dropout=fluo.get('dropout', 0.2),
        hidden_dims=fluo.hidden_dim,
        activation=fluo.get('activation', 'relu'),
        norm=fluo.get('norm', 'bn'),
    )


def _build_fluorescence_mlp_simple(fluo) -> nn.Module:
    """Built-in builder for the legacy lightweight FluorescenceMLP_Simple."""
    return FluorescenceMLP_Simple(
        input_dim=fluo.input_dim,
        output_dim=fluo.output_dim,
        hidden_dims=fluo.hidden_dim,
    )


register_fluorescence_encoder("FluorescenceMLP", _build_fluorescence_mlp)
register_fluorescence_encoder("FluorescenceMLP_Simple", _build_fluorescence_mlp_simple)


__all__ = [
    "FluorescenceMLP",
    "FluorescenceMLP_Simple",
    "FLUORESCENCE_ENCODER_REGISTRY",
    "register_fluorescence_encoder",
]