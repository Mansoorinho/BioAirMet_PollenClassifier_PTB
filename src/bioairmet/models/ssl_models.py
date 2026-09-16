'''
@file    :   ssl_models.py
@create date : 2025-02-10 13:36:08
@modify date 2026-08-25
@author  :   Mansoor Nabawi
@version :   2.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    Self-supervised learning (SSL) models for the BioAirMet project.

    SSLModel_SingleIMG: contrastive SSL model for one holographic image view
    + fluorescence features. Each modality has its own encoder (image backbone
    / fluorescence MLP) and its own projection head; a CLIP-style temperature
    (static or learnable) scales the logits.

    Factory functions turn the merged architecture config into instances:
        build_ssl_model_from_config     -> SSLModel_SingleIMG
        build_img_encoder_from_config   -> grayscale image backbone
        build_fl_encoder_from_config    -> fluorescence encoder (registry)
    ]
'''
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from ._common import load_state_dict, ensure_state_dict_metadata
from .backbones import BACKBONE_REGISTRY, GrayscaleBackbone
from .fluorescence_encoders import FLUORESCENCE_ENCODER_REGISTRY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

PROJECTOR_TYPES = ("big", "small_norm", "small")


class Projector_Flexible(nn.Module):
    """Projection head for SSL models (maps an encoder embedding to the shared space).

    Supported types:
        'big'        2-layer MLP: Linear -> Norm -> Act -> Linear.
                     Acts as a computational shield for the encoder (recommended).
        'small_norm' Linear -> Norm.
        'small'      single Linear (weakest; not recommended for robust SSL).

    Args:
        input_dim: encoder output dimensionality.
        projection_dim: shared latent dimensionality (default 256).
        projector_type: 'big' | 'small_norm' | 'small' (default 'big').
        norm: 'bn' (BatchNorm1d) or 'ln' (LayerNorm) (default 'bn').
        bias: whether to use bias in the linear layers (default True).
        activation: 'relu' or 'gelu' (default 'relu').
        type: (legacy) alias of projector_type.
    """

    def __init__(self, input_dim: int, projection_dim: int = 256,
                 projector_type: str = "big", norm: str = "bn",
                 bias: bool = True, activation: str = "relu", type: Optional[str] = None):
        super().__init__()
        if type is not None:  # legacy kwarg name
            projector_type = type
        if projector_type not in PROJECTOR_TYPES:
            raise ValueError(
                f"Unknown projector type: '{projector_type}'. "
                f"Supported: {', '.join(PROJECTOR_TYPES)}"
            )

        def _norm(dim: int) -> nn.Module:
            return nn.BatchNorm1d(dim) if norm == "bn" else nn.LayerNorm(dim)

        act = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()

        if projector_type == "big":
            layers = [
                nn.Linear(input_dim, input_dim, bias=bias),
                _norm(input_dim),
                act,
                nn.Linear(input_dim, projection_dim, bias=bias),
            ]
        elif projector_type == "small_norm":
            layers = [
                nn.Linear(input_dim, projection_dim, bias=bias),
                _norm(projection_dim),
            ]
        else:  # 'small'
            layers = [nn.Linear(input_dim, projection_dim, bias=bias)]

        self.projector = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


# ---------------------------------------------------------------------------
# SSL model
# ---------------------------------------------------------------------------

class SSLModel_SingleIMG(nn.Module):
    """Contrastive SSL model for one image view + fluorescence features.

    Both modalities are encoded, projected into a shared latent space, L2
    normalised, and returned together with a logit scale (temperature).

    Args:
        img_base_model: image encoder (any module with get_output_dim()).
        fl_base_model: fluorescence encoder (any module with get_output_dim()).
        projection_dim: shared latent dim (default 256).
        projector_type: 'big' | 'small_norm' | 'small' (default 'small').
        norm_type: 'bn' | 'ln' for the projector heads (default 'bn').
        projection_activation: 'relu' | 'gelu' (default 'relu').
        projection_bias: bias flag for the projector linear layers (default True).
        freeze_bn: freeze BatchNorm layers at construction (default False).
        learnable_temperature: CLIP-style learnable logit scale (default False).
        init_temperature: initial temperature (default 0.1).
        config: (legacy) full config dict; used to fill any argument left as
            None from `config['architecture_setup']['ssl_model']`.
    """

    def __init__(self,
                 img_base_model: nn.Module,
                 fl_base_model: nn.Module,
                 projection_dim: Optional[int] = None,
                 projector_type: Optional[str] = None,
                 norm_type: Optional[str] = None,
                 projection_activation: Optional[str] = None,
                 projection_bias: Optional[bool] = None,
                 freeze_bn: bool = False,
                 learnable_temperature: bool = False,
                 init_temperature: float = 0.1,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__()

        # Legacy support: read the ssl_model section when an explicit value
        # was not given.
        ssl_conf: Dict[str, Any] = {}
        if config is not None:
            try:
                ssl_conf = config["architecture_setup"]["ssl_model"] or {}
            except (KeyError, TypeError):
                ssl_conf = {}

        def _resolve(key: str, default: Any, value: Any) -> Any:
            """Explicit kwarg wins, then config, then the built-in default."""
            if value is not None:
                return value
            return ssl_conf.get(key, default)

        projection_dim = _resolve("projection_dim", 256, projection_dim)
        projector_type = _resolve("projector_type", "small", projector_type)
        norm_type = _resolve("projector_norm", "bn", norm_type)
        projection_activation = _resolve("projector_activation", "relu", projection_activation)
        projection_bias = _resolve("projector_bias", True, projection_bias)
        freeze_bn = bool(ssl_conf.get("freeze_bn", False) or freeze_bn)

        # --- 1. Encoders + projection heads (shared latent space) ---
        self.img_encoder = img_base_model
        self.fl_encoder = fl_base_model
        img_feature_dim = self.img_encoder.get_output_dim()
        fl_feature_dim = self.fl_encoder.get_output_dim()

        self.img_projector = Projector_Flexible(
            img_feature_dim, projection_dim=projection_dim,
            projector_type=projector_type, norm=norm_type,
            bias=projection_bias, activation=projection_activation)
        self.fl_projector = Projector_Flexible(
            fl_feature_dim, projection_dim=projection_dim,
            projector_type=projector_type, norm=norm_type,
            bias=projection_bias, activation=projection_activation)

        # --- 2. CLIP-style temperature / logit scale ---
        self.learnable_temperature = learnable_temperature
        if self.learnable_temperature:
            # logit_scale = exp(logit_scale_param) = 1 / temperature
            init_logit_scale = torch.log(torch.tensor(1.0 / init_temperature))
            self.logit_scale = nn.Parameter(init_logit_scale)
        else:
            self.register_buffer("static_logit_scale", torch.tensor(1.0 / init_temperature))

        if freeze_bn:
            self.freeze_batchnorm()

        # Encoders that must stay frozen + in eval mode (set via freeze_encoder()).
        self._frozen_encoders: set = set()

        logger.info(
            "SSLModel_SingleIMG: img_dim=%s fl_dim=%s projection_dim=%s "
            "projector=%s/%s learnable_temperature=%s (init %s)",
            img_feature_dim, fl_feature_dim, projection_dim,
            projector_type, norm_type, learnable_temperature, init_temperature)

    def freeze_batchnorm(self, scope: Optional[nn.Module] = None) -> int:
        """Freeze BatchNorm layers: ``eval()`` mode, KEEPING ``track_running_stats``.

        ``eval()`` is what stops ``running_mean``/``running_var`` from updating, so
        tracking is deliberately left ON - the convention used across this project
        (``models.backbones.freeze_batchnorm`` and the classification stage), and
        checked by ``utils.bn_audit``.

        Setting ``track_running_stats=False`` instead is NOT equivalent: PyTorch
        then normalises with BATCH statistics whenever the layer is in train mode
        and with the stored buffers in eval mode - so a freshly built, never
        trained encoder whose buffers are still mean=0/var=1 would silently stop
        normalising at all, and any code path that calls ``.train()`` on it would
        change its normalisation without warning.

        Args:
            scope: restrict freezing to this sub-module (default: whole model).

        Returns:
            int: number of BatchNorm layers frozen.
        """
        root = scope if scope is not None else self
        count = 0
        for module in root.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.track_running_stats = True
                count += 1
        return count

    def freeze_encoder(self, name: str) -> None:
        """Freeze one encoder and keep it in eval mode with frozen BN stats.

        A frozen encoder has requires_grad=False on all parameters, is held
        in eval() mode (re-applied on every forward), and its BatchNorm
        running statistics never update — consistent with the classification
        stage's frozen-encoder semantics.

        Args:
            name: 'img_encoder' or 'fl_encoder'.
        """
        if name not in ('img_encoder', 'fl_encoder'):
            raise ValueError(
                f"freeze_encoder: expected 'img_encoder' or 'fl_encoder', got {name!r}")
        encoder = getattr(self, name)
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        self.freeze_batchnorm(scope=encoder)
        self._frozen_encoders.add(name)
        logger.info("SSL model: %s frozen (requires_grad=False, eval, BN stats frozen)", name)

    def forward(self, view_a: torch.Tensor, fl: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass: embed both modalities into the shared contrastive space.

        Returns:
            dict: with keys
                - ``image_embeddings`` (torch.Tensor): L2-normalised image projection.
                - ``fl_embeddings`` (torch.Tensor): L2-normalised fluorescence
                  projection (same shape as ``image_embeddings``).
                - ``logit_scale`` (torch.Tensor): the clamped temperature scale
                  (``exp(logit_scale)`` if learnable, else the static ``1/temperature``).
        """
        # Keep frozen encoders in eval mode even after model.train().
        for name in self._frozen_encoders:
            getattr(self, name).eval()

        # --- A. Feature extraction ---
        img_a_proj = self.img_encoder(view_a)
        fl_proj = self.fl_encoder(fl)

        # --- B. Projection to the shared space ---
        img_a_proj = self.img_projector(img_a_proj)
        fl_proj = self.fl_projector(fl_proj)

        img_a_proj = F.normalize(img_a_proj, p=2, dim=-1, eps=1e-8)
        fl_proj = F.normalize(fl_proj, p=2, dim=-1, eps=1e-8)

        # --- C. Logit scale (temperature) with consistent clamping ---
        # CRITICAL: clamp here so train/val behave identically.
        if self.learnable_temperature:
            logit_scale = self.logit_scale.exp()
            logit_scale = torch.clamp(logit_scale, min=1.0, max=100.0)
        else:
            logit_scale = self.static_logit_scale

        return {
            "image_embeddings": img_a_proj,
            "fl_embeddings": fl_proj,
            "logit_scale": logit_scale,
        }


# ---------------------------------------------------------------------------
# Factory functions (config -> model)
# ---------------------------------------------------------------------------

def _parse_backbone_spec(model_name: str) -> Tuple[str, str]:
    """Split a config `image_tower.model_name` into (family, variant).

    Matching is driven by the BACKBONE_REGISTRY name_prefixes; the longest
    prefix wins, so 'mobilenet_v3_small' resolves against the
    'mobilenet_v3_' prefix, not the bare 'mobilenet_'. Families added with
    register_backbone() are picked up here without further changes.

    Examples:
        'efficientnet_b0'     -> ('efficientnet', 'b0')
        'efficientnet_v2_l'   -> ('efficientnet', 'v2_l')
        'efficientnet_v1_100k' -> ('efficientnet', 'v1_100k')
        'mobilenet_v3_small'  -> ('mobilenet_v3', 'small')
        'fastvit_t8'          -> ('fastvit', 't8')
        'shufflenet_v2_x1_0'  -> ('shufflenet_v2', 'x1_0')
        'mnasnet1_0'          -> ('mnasnet', '1_0')
    """
    name = str(model_name).strip().lower()

    best_family, best_prefix = None, ""
    for family, info in BACKBONE_REGISTRY.items():
        for prefix in info["name_prefixes"]:
            if name.startswith(prefix) and len(prefix) > len(best_prefix):
                best_family, best_prefix = family, prefix
    if best_family is not None:
        return best_family, name[len(best_prefix):]

    supported = ", ".join(
        f"{family}_{{{','.join(info['variants'])}}}"
        for family, info in BACKBONE_REGISTRY.items()
    )
    raise ValueError(f"Unknown image encoder name: '{model_name}'. Supported: {supported}")


def build_img_encoder_from_config(config):
    """
    Builds a grayscale image encoder (backbone) from the merged architecture config.

    Reads `architecture_setup.image_tower`:
        model_name     backbone spec, e.g. 'efficientnet_b0' (required)
        pretrained     True/False or weights string (default False)
        freeze_bn      freeze BatchNorm layers at construction (default False)
        input_channels input channels: 1 = grayscale (default), 3 = RGB

    Args:
        config (EasyDict): merged configuration object.

    Returns:
        torch.nn.Module: backbone wrapper with forward() and get_output_dim().
    """
    img_tower = config.architecture_setup.image_tower
    model_name = img_tower.model_name
    pretrained = img_tower.get('pretrained', False)
    freeze_bn = bool(img_tower.get('freeze_bn', False))
    in_channels = int(img_tower.get('input_channels', 1))

    family, variant = _parse_backbone_spec(model_name)
    logger.info("Building image encoder: %s (variant=%s, pretrained=%s, in_channels=%d)",
                family, variant, pretrained, in_channels)
    return GrayscaleBackbone(family, variant, pretrained=pretrained,
                             freeze_bn=freeze_bn, in_channels=in_channels)


def build_fl_encoder_from_config(config):
    """
    Builds a fluorescence encoder from the merged architecture config.

    Reads `architecture_setup.fluorescence_tower`:
        model_name selects the encoder from FLUORESCENCE_ENCODER_REGISTRY
        (built in: 'FluorescenceMLP', 'FluorescenceMLP_Simple'); the selected
        builder receives the whole fluorescence_tower section
        (input_dim / output_dim / hidden_dim / dropout / activation / norm).
        Custom encoders are added with register_fluorescence_encoder() —
        see models/custom_models.py.

    Args:
        config (EasyDict): merged configuration object.

    Returns:
        torch.nn.Module: fluorescence encoder with forward() and get_output_dim().
    """
    fluo = config.architecture_setup.fluorescence_tower
    encoder_name = fluo.model_name

    builder = FLUORESCENCE_ENCODER_REGISTRY.get(encoder_name)
    if builder is None:
        raise ValueError(
            f"Unknown fluorescence encoder name: '{encoder_name}'. "
            f"Supported: {', '.join(sorted(FLUORESCENCE_ENCODER_REGISTRY))}. "
            "Register custom encoders with register_fluorescence_encoder()."
        )

    return builder(fluo)


_SSL_MODELS = {
    "SSLModel_SingleIMG": SSLModel_SingleIMG,
}


def build_ssl_model_from_config(config):
    """
    Builds the SSL model from the merged architecture config.

    Steps:
        1. build the image + fluorescence encoders
        2. build the SSL model (SSLModel_SingleIMG) around them
        3. if `architecture_setup.type == 'classification'` -> return as-is
           (weights are handled by the classification builder)
        4. otherwise, if `model_initialization.pretrained.enable` is set
           (legacy: `architecture_setup.pretrained`), load the SSL
           checkpoint it points to.

    Args:
        config (EasyDict): merged configuration object.

    Returns:
        torch.nn.Module: SSL model (SSLModel_SingleIMG).
    """
    img_base_model = build_img_encoder_from_config(config)
    fl_base_model = build_fl_encoder_from_config(config)

    ssl_conf = config.architecture_setup.ssl_model
    model_name = ssl_conf.get('model_name', 'SSLModel_SingleIMG')
    if model_name not in _SSL_MODELS:
        raise ValueError(
            f"Unknown SSL model name: '{model_name}'. "
            f"Supported: {', '.join(_SSL_MODELS)}"
        )

    # Temperature lives in the loss config (train.loss); read it defensively.
    train_conf = config.get('train') if hasattr(config, 'get') else None
    loss_conf = train_conf.get('loss') if train_conf is not None else None
    learnable_temp = bool(loss_conf.get('learnable_temp', False)) if loss_conf is not None else False
    temperature = float(loss_conf.get('temperature', 0.1)) if loss_conf is not None else 0.1

    ssl_model = _SSL_MODELS[model_name](
        img_base_model=img_base_model,
        fl_base_model=fl_base_model,
        freeze_bn=bool(ssl_conf.get('freeze_bn', False)),
        learnable_temperature=learnable_temp,
        init_temperature=temperature,
        config=config,  # ssl_model section fills the remaining defaults
    )

    # When called from the classification stage, weights are handled upstream.
    if config.architecture_setup.get('type') == 'classification':
        return ssl_model

    # SSL stage: optional pretrained checkpoint.
    # Unified location (both stages): model_initialization.pretrained;
    # legacy location: architecture_setup.pretrained (old configs / bundles).
    pre_trained = (config.get('model_initialization') or {}).get('pretrained') or {}
    if not pre_trained:
        pre_trained = config.architecture_setup.get('pretrained') or {}
    # Both key spellings are accepted during the transition:
    # `enable` (new) and `enabled` (legacy template).
    if not (pre_trained.get('enable') or pre_trained.get('enabled')):
        return ssl_model

    checkpoint_path = pre_trained.get('checkpoint_path')
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found at: {checkpoint_path}")

    state_dict = load_state_dict(checkpoint_path)
    state_dict = ensure_state_dict_metadata(state_dict, ssl_model)
    ssl_model.load_state_dict(state_dict)
    logger.info("SSL model loaded from checkpoint: %s", checkpoint_path)
    return ssl_model


__all__ = [
    "SSLModel_SingleIMG",
    "Projector_Flexible",
    "PROJECTOR_TYPES",
    "build_ssl_model_from_config",
    "build_img_encoder_from_config",
    "build_fl_encoder_from_config",
]
