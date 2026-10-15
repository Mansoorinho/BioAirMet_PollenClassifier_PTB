'''
@file    :   ssl_losses.py
@create date : 2025-06-10 13:36:08
@modify date 2026-10-12
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    This file defines the loss functions used for self-supervised learning (SSL) in the BioAirMet project.
    - ContrastiveLoss: Legacy (non-CLIP) symmetric cross-entropy loss over similarity matrices with log-space temperature, epsilon-normalized embeddings and logit clamping. Not reachable from train.loss.name: its forward() takes no logit_scale, so SSLModel_SingleIMG cannot drive it.
    - Clip_ContrastiveLoss: CLIP-style symmetric InfoNCE over the image / fluorescence embeddings, with optional cross-GPU gathering (emulate_old_loss / local_loss / gather_with_grad) and dtype promotion so an AMP image branch and an fp32 fluorescence branch can be compared.
    - negative_cosine_similarity: A utility function for computing cosine similarity based losses.
    - SSL_LOSS_REGISTRY / SUPPORTED_SSL_LOSSES: The single source of truth for the names accepted by train.loss.name.
    - resolve_ssl_temperature_keys: Shared resolution of the learnable_temp / learnable_temperature spellings (model factory, loss factory and optimizer grouping all read it).
    - build_ssl_loss_from_config: Factory that instantiates the loss named in the config and fails fast on an unknown name.
    ]
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import torch.distributed as dist

logger = logging.getLogger(__name__)

try:
    import torch.distributed.nn as dist_nn
    HAS_DIST_NN = True
except ImportError:
    HAS_DIST_NN = False
    
class ContrastiveLoss(nn.Module):
    """
    Improved ContrastiveLoss with numerical stability guarantees.
    Fixes the temperature scaling issue that causes NaN/Inf crashes.
    """
    def __init__(self, temperature=0.5, learnable_temperature=False):
        super().__init__()
        # Store temperature in log space for numerical stability
        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(max(temperature, 0.01))),
            requires_grad=learnable_temperature
        )
        
    @property
    def temperature(self):
        """Get temperature with clamping to prevent extreme values."""
        return torch.clamp(self.log_temperature.exp(), min=0.01, max=10.0).item()
        
    def forward(self, image_embeddings, fl_embeddings):
        """
        Forward pass with numerical stability improvements.
        
        Args:
            image_embeddings: (B, D) tensor
            fl_embeddings: (B, D) tensor
            
        Returns:
            scalar loss value
        """
        B = image_embeddings.shape[0]
        
        # Normalize embeddings with epsilon for numerical stability
        z_i = F.normalize(image_embeddings, p=2, dim=-1, eps=1e-8)
        z_j = F.normalize(fl_embeddings, p=2, dim=-1, eps=1e-8)
        
        # Get temperature with safety clamping
        temperature = torch.clamp(self.log_temperature.exp(), min=0.01, max=10.0)
        
        # Compute similarity matrix: (B, B)
        # similarity ∈ [-1, 1] for normalized vectors
        logits = torch.matmul(z_i, z_j.t())  # ← First compute similarity
        
        # THEN scale by temperature (this prevents extreme values)
        logits = logits / temperature
        
        # Clamp logits to prevent overflow in exp()
        # log(2^30) ≈ 20.7, so we clamp to [-50, 50] for safety
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        
        # Check for NaN/Inf before loss computation
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError(
                f"NaN/Inf detected in logits. "
                f"temp={temperature.item():.6f}, "
                f"logits: min={logits.min()}, max={logits.max()}"
            )
        
        # Create labels: each sample i should match with fl_embedding i
        labels = torch.arange(B, device=image_embeddings.device)
        
        # Cross-entropy loss
        # Image as query: logits[i, j] = sim(image_i, fl_j), we want i==j
        loss_i = F.cross_entropy(logits, labels)
        # FL as query: logits.t()[i, j] = sim(fl_i, image_j), we want i==j
        loss_j = F.cross_entropy(logits.t(), labels)
        
        loss = (loss_i + loss_j) / 2
        
        # Final NaN check
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(
                f"NaN/Inf loss detected. loss_i={loss_i}, loss_j={loss_j}, "
                f"temperature={temperature.item()}"
            )

        return loss
    

def negative_cosine_similarity(p, z):
    # z must be detached before calling this!
    return -F.cosine_similarity(p, z.detach(), dim=-1).mean()


class Clip_ContrastiveLoss(nn.Module):
    def __init__(
        self,
        emulate_old_loss: bool = False,
        local_loss: bool = False,
        gather_with_grad: bool = True,  # ✅ better default
    ):
        super().__init__()
        self.emulate_old_loss = emulate_old_loss
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad

    def _get_dist_info(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def gather_features(self, image_features, fl_features, rank, world_size):
        if world_size == 1:
            return image_features, fl_features
        
        # Detach to avoid gradient flow through all_gather
        # image_features = image_features.detach()
        # fl_features = fl_features.detach()

        if self.gather_with_grad:
            assert HAS_DIST_NN
            gathered_img = dist_nn.all_gather(image_features)
            gathered_fl = dist_nn.all_gather(fl_features)
            all_image_features = torch.cat(list(gathered_img), dim=0)
            all_fl_features = torch.cat(list(gathered_fl), dim=0)
        else:
            image_list = [torch.empty_like(image_features) for _ in range(world_size)]
            fl_list  = [torch.empty_like(fl_features) for _ in range(world_size)]

            dist.all_gather(image_list, image_features)
            dist.all_gather(fl_list, fl_features)

            image_list[rank] = image_features
            fl_list[rank] = fl_features

            all_image_features = torch.cat(image_list, dim=0)
            all_fl_features = torch.cat(fl_list, dim=0)

        return all_image_features, all_fl_features

    def forward(self, image_features, fl_features, logit_scale):
        rank, world_size = self._get_dist_info()

        # --- Mixed-precision safety ---
        # The image branch normally arrives in fp16 (AMP) while the fluorescence
        # branch is deliberately kept in fp32 (its spectral values are far too
        # small to survive fp16 rounding — see SSLModel_SingleIMG.fl_fp32).
        # A matmul between the two dtypes would raise, so both sides are
        # promoted to the wider one before the B x B similarity matrix and the
        # cross-entropy are computed. Gradients still flow back into the fp16
        # branch unchanged; this only avoids rounding inside the loss itself.
        compute_dtype = torch.result_type(image_features, fl_features)
        if image_features.dtype != compute_dtype:
            image_features = image_features.to(compute_dtype)
        if fl_features.dtype != compute_dtype:
            fl_features = fl_features.to(compute_dtype)
        if torch.is_tensor(logit_scale) and logit_scale.dtype != compute_dtype:
            logit_scale = logit_scale.to(compute_dtype)

        # --- Old loss ---
        if self.emulate_old_loss or world_size == 1:
            logits = logit_scale * (image_features @ fl_features.t())
            labels = torch.arange(logits.size(0), device=logits.device)

            loss_i = F.cross_entropy(logits, labels)
            loss_f = F.cross_entropy(logits.t(), labels)
            return 0.5 * (loss_i + loss_f)

        # --- Distributed ---
        all_image_features, all_fl_features = self.gather_features(
            image_features, fl_features, rank, world_size
        )

        if self.local_loss:
            logits_per_image = logit_scale * (image_features @ all_fl_features.t())
            logits_per_fl  = logit_scale * (fl_features @ all_image_features.t())

            batch_size = image_features.shape[0]
            labels = torch.arange(batch_size, device=image_features.device)
            labels = labels + batch_size * rank
        else:
            logits_per_image = logit_scale * (all_image_features @ all_fl_features.t())
            logits_per_fl  = logits_per_image.t()

            labels = torch.arange(logits_per_image.size(0), device=image_features.device)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_f = F.cross_entropy(logits_per_fl, labels)

        return 0.5 * (loss_i + loss_f)
    
# ---------------------------------------------------------------------------
# Loss registry — single source of truth for ``train.loss.name`` (SSL)
# ---------------------------------------------------------------------------

SSL_LOSS_REGISTRY = {
    # Symmetric CLIP InfoNCE over the two modality embeddings. Matches
    # open_clip.ClipLoss, and for world_size == 1 its math exactly.
    "Clip_ContrastiveLoss": Clip_ContrastiveLoss,
    # Historical name shipped by the SSL config templates. It always selected
    # the same symmetric InfoNCE objective, so it is kept as an explicit alias
    # rather than being silently ignored in favour of a hard-coded class.
    # The older non-CLIP ``ContrastiveLoss`` class above is deliberately NOT
    # registered: its forward() takes no logit_scale and therefore cannot be
    # driven by SSLModel_SingleIMG, which hands the model-owned temperature to
    # the loss through compute_ssl_loss().
    "ContrastiveLoss": Clip_ContrastiveLoss,
}

SUPPORTED_SSL_LOSSES = tuple(SSL_LOSS_REGISTRY)

# Values applied when a key is absent from train.loss. They mirror the values
# the SSL trainer hard-coded before the loss became config-routed, so existing
# configs keep behaving exactly as before. (``emulate_old_loss`` has no effect
# at world_size == 1: both branches compute the same symmetric InfoNCE.)
SSL_LOSS_DEFAULTS = {
    "name": "ContrastiveLoss",
    "emulate_old_loss": True,
    "local_loss": False,
    "gather_with_grad": False,
}


def resolve_ssl_temperature_keys(loss_conf, log=None):
    """Resolve ``(temperature, learnable_temp)`` from a ``train.loss`` section.

    Two spellings of the learnable-temperature key exist in this project:
    ``learnable_temp`` (the key the SSL templates ship and that
    ``build_ssl_model_from_config`` reads) and ``learnable_temperature`` (the
    key the loss classes take and the old factory documented). Reading only one
    of them silently loses the setting, so both are accepted here and this is
    the single place that resolves them — shared by the model factory, the loss
    factory and the optimizer parameter grouping.

    Args:
        loss_conf: the ``config.train.loss`` mapping (dict / EasyDict / None).
        log: optional logger for the disagreement warning.

    Returns:
        tuple: ``(temperature, learnable_temp)``. ``temperature`` is ``None``
        when unset; ``learnable_temp`` is ``False`` for a missing/empty section
        and ``None`` when the section exists but sets neither key, so callers
        can fall back to their own default.
    """
    if not loss_conf:
        return None, False

    log = log or logging.getLogger(__name__)
    temperature = loss_conf.get('temperature', None)

    has_primary = 'learnable_temp' in loss_conf
    has_alias = 'learnable_temperature' in loss_conf

    learnable_temp = None
    if has_primary:
        learnable_temp = bool(loss_conf.get('learnable_temp', False))
    elif has_alias:
        learnable_temp = bool(loss_conf.get('learnable_temperature', False))

    if has_primary and has_alias:
        alias_value = bool(loss_conf.get('learnable_temperature', False))
        if alias_value != learnable_temp:
            log.warning(
                "train.loss sets both 'learnable_temp'=%s and "
                "'learnable_temperature'=%s and they disagree; using "
                "'learnable_temp'=%s. Keep only one of the two keys.",
                learnable_temp, alias_value, learnable_temp)

    return temperature, learnable_temp


# Factory function to build SSL loss from config
def build_ssl_loss_from_config(config, log=None):
    """Build the SSL loss selected by ``train.loss.name``.

    The loss is resolved through :data:`SSL_LOSS_REGISTRY`; an unknown name is
    a hard error rather than a silent fallback, because training with a
    different objective than the one written in the config is exactly the kind
    of bug that costs a multi-day run.

    Config keys (all under ``train.loss``):
        name: ``Clip_ContrastiveLoss`` or ``ContrastiveLoss`` (an alias of the
            former). Defaults to ``SSL_LOSS_DEFAULTS['name']``.
        emulate_old_loss: compute the symmetric InfoNCE on this rank's features
            only, without gathering across GPUs (default True; identical math to
            the joint form when world_size == 1).
        local_loss: with emulate_old_loss False, use local query features
            against the gathered key features (default False).
        gather_with_grad: differentiable all-gather so every rank receives
            gradients through the gathered features (default False).
        temperature / learnable_temp / learnable_temperature: logged here; the
            logit scale itself is owned by ``SSLModel_SingleIMG`` (see
            ``build_ssl_model_from_config``), which reads the same keys through
            :func:`resolve_ssl_temperature_keys`.

    Args:
        config: full configuration (needs ``config.train.loss``).
        log: optional logger (defaults to this module's logger).

    Returns:
        torch.nn.Module: An instance of the specified SSL loss function.

    Raises:
        ValueError: if ``train.loss.name`` is not in the registry, or the
            ``train.loss`` section is missing.
    """
    logger_ = log or logging.getLogger(__name__)

    loss_conf = config.train.loss if (config and getattr(config, 'train', None)) else None
    if not loss_conf:
        raise ValueError(
            "build_ssl_loss_from_config requires config.train.loss with a "
            f"'name' key; supported: {', '.join(SUPPORTED_SSL_LOSSES)}.")

    loss_name = loss_conf.get('name', 'ContrastiveLoss')
    if loss_name not in SSL_LOSS_REGISTRY:
        raise ValueError(
            f"Unknown SSL loss: train.loss.name={loss_name!r}. "
            f"Supported: {', '.join(SUPPORTED_SSL_LOSSES)}. The loss is "
            "resolved through build_ssl_loss_from_config, so a typo fails at "
            "startup instead of being silently ignored.")

    temperature, learnable_temp = resolve_ssl_temperature_keys(loss_conf, logger_)
    loss_cls = SSL_LOSS_REGISTRY[loss_name]

    emulate_old = bool(loss_conf.get('emulate_old_loss',
                                     SSL_LOSS_DEFAULTS['emulate_old_loss']))
    local_loss = bool(loss_conf.get('local_loss',
                                    SSL_LOSS_DEFAULTS['local_loss']))
    gather_with_grad = bool(loss_conf.get('gather_with_grad',
                                          SSL_LOSS_DEFAULTS['gather_with_grad']))

    loss = loss_cls(emulate_old_loss=emulate_old,
                    local_loss=local_loss,
                    gather_with_grad=gather_with_grad)

    logger_.info(
        "SSL loss: train.loss.name=%s -> %s (emulate_old_loss=%s, "
        "local_loss=%s, gather_with_grad=%s)",
        loss_name, loss_cls.__name__, emulate_old, local_loss, gather_with_grad)
    if loss_name != "Clip_ContrastiveLoss":
        logger_.info("  ('%s' is an alias of 'Clip_ContrastiveLoss': symmetric "
                     "InfoNCE over the two modality embeddings)", loss_name)
    logger_.info(
        "  temperature=%s learnable_temp=%s (the logit scale is owned by "
        "SSLModel_SingleIMG / build_ssl_model_from_config)",
        temperature, learnable_temp)
    return loss