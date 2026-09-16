'''
@file    :   __init__.py
@desc    :   Public API of the models package.

    build_model_from_config(config)
        Top-level dispatcher: builds the SSL model (architecture_setup.type
        == 'ssl') or the classification model (type == 'classification')
        from the merged architecture config.

    build_ssl_model_from_config(config)
        SSL stage: SSLModel_SingleIMG (+ optional pretrained checkpoint).

    build_classification_model_from_config_v2(config)
        Classification stage: HoloClassifierV2 on top of the SSL encoders,
        with model_initialization-driven weight loading.

    build_img_encoder_from_config / build_fl_encoder_from_config
        Individual encoder builders (used internally and handy in scripts).

    register_backbone / register_fluorescence_encoder
        Public extension points for custom encoders (see custom_models.py
        and docs/custom_models.md).

    GrayscaleBackbone / EfficientNet_grayscale / ... 
        The grayscale image backbones and their registry.
'''
import os
from typing import Optional

import torch

from .backbones import (
    BACKBONE_REGISTRY,
    GrayscaleBackbone,
    build_grayscale_backbone,
    register_backbone,
    EfficientNet_grayscale,
    MobileNetV3_grayscale,
    fastViT_grayscale,
    ShuffleNetV2_grayscale,
    MNASNet_grayscale,
)
from .fluorescence_encoders import (
    FluorescenceMLP,
    FluorescenceMLP_Simple,
    FLUORESCENCE_ENCODER_REGISTRY,
    register_fluorescence_encoder,
)
from .ssl_models import (
    SSLModel_SingleIMG,
    Projector_Flexible,
    build_ssl_model_from_config,
    build_img_encoder_from_config,
    build_fl_encoder_from_config,
)
from .classification_models_v2 import (
    HoloClassifierV2,
    build_classification_model_from_config_v2,
)
from ._common import load_weights_into, load_state_dict, strip_ddp_prefix
from . import custom_models  # noqa: F401  (imports register the template encoders)
from .custom_models import SimpleCnn_grayscale, FluorescenceCNN


def build_model_from_config(config):
    """
    Builds a model instance (SSL or classification) from the merged config.

    Dispatches on `architecture_setup.type`:
        'ssl'            -> build_ssl_model_from_config(config)
        'classification' -> build_classification_model_from_config_v2(config)

    Args:
        config (EasyDict): merged configuration object.

    Returns:
        torch.nn.Module: the built model.
    """
    model_type = config.architecture_setup.get('type')

    if model_type == 'ssl':
        return build_ssl_model_from_config(config)

    if model_type == 'classification':
        classification_conf = config.architecture_setup.classification_model
        model_name = classification_conf.get('name', '')
        if model_name not in ('HoloClassifierV2',):
            raise ValueError(
                f"Unknown classification model name: '{model_name}'. "
                "Supported: HoloClassifierV2 (legacy V1 models were removed)."
            )
        return build_classification_model_from_config_v2(config)

    raise ValueError(
        f"Unknown architecture_setup.type: '{model_type}'. "
        "Expected 'ssl' or 'classification'."
    )


def load_model(config, checkpoint_path: Optional[str] = None,
               device: str = "cpu") -> "torch.nn.Module":
    """Build a model from ``config`` and (optionally) load checkpoint weights.

    This is the single documented entry point for loading a model:

        model = load_model(config)                          # fresh (random) weights
        model = load_model(config, "exp/last_model.pth")    # + load checkpoint
        model = load_model(config, ckpt, device="cuda:0")   # on a specific device

    The model type is chosen from ``architecture_setup.type`` (``'ssl'`` ->
    ``SSLModel_SingleIMG``, ``'classification'`` -> ``HoloClassifierV2``). If
    ``checkpoint_path`` is given, its weights are loaded with ``strict=False``
    (so an encoder-only checkpoint can be loaded into a full model) and any
    missing/unexpected keys are logged. The returned model is moved to ``device``
    and set to ``eval()``.

    Args:
        config: merged configuration object (EasyDict or dict).
        checkpoint_path: optional path to a checkpoint file.
        device: target device (default ``'cpu'``).

    Returns:
        torch.nn.Module: the model on ``device`` in ``eval()`` mode.

    Raises:
        FileNotFoundError: if ``checkpoint_path`` is given but does not exist.
    """
    model = build_model_from_config(config).to(device)

    if checkpoint_path is not None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path!r}")
        load_weights_into(model, checkpoint_path, strict=False, label="load_model")

    model.eval()
    return model


__all__ = [
    # Top-level
    'build_model_from_config',
    'load_model',
    # Checkpoint / weight loading
    'load_weights_into',
    'load_state_dict',
    'strip_ddp_prefix',
    # SSL stage
    'build_ssl_model_from_config',
    'SSLModel_SingleIMG',
    'Projector_Flexible',
    # Classification stage
    'build_classification_model_from_config_v2',
    'HoloClassifierV2',
    # Encoders
    'build_img_encoder_from_config',
    'build_fl_encoder_from_config',
    'FluorescenceMLP',
    'FluorescenceMLP_Simple',
    # Custom-model extension points
    'register_backbone',
    'FLUORESCENCE_ENCODER_REGISTRY',
    'register_fluorescence_encoder',
    'custom_models',
    'SimpleCnn_grayscale',
    'FluorescenceCNN',
    # Backbones
    'BACKBONE_REGISTRY',
    'GrayscaleBackbone',
    'build_grayscale_backbone',
    'EfficientNet_grayscale',
    'MobileNetV3_grayscale',
    'fastViT_grayscale',
    'ShuffleNetV2_grayscale',
    'MNASNet_grayscale',
]