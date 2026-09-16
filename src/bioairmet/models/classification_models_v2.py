'''
@file    :   classification_models_v2.py
@create date : 2026-01-19 11:25:33
@modify date 2026-08-25
@author  :   Mansoor Nabawi
@version :   2.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    HoloClassifierV2: the supervised classification head on top of the SSL
    encoders.

    Key properties:
        - By default the encoders are fully frozen: requires_grad=False and
          eval() mode. Their BatchNorm layers keep track_running_stats enabled
          but stay in eval(), so they normalise with the pretrained running
          statistics and can never drift during fine-tuning (eval() blocks
          updates).
        - forward() re-enforces the per-encoder train/eval state every step,
          including partial fine-tuning ("unfreeze last N layers").
        - Optional L2 normalisation of the features before concatenation.
        - The fine-tuning policy (which encoder parts are trainable) is
          applied from `train.fine_tuning` by the builder, so the model's
          requires_grad / train-eval / BN-stat state always matches the
          optimizer parameter grouping.

    build_classification_model_from_config_v2: factory that
        1. resolves the initialisation mode from `model_initialization`
           ('none' | 'ssl_pretrain' | 'cls_finetune' | 'resume'),
        2. builds the SSL base model (fresh encoders),
        3. loads the right weights into the right model,
        4. wraps everything in HoloClassifierV2.
    ]
'''
import os
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from .backbones import freeze_batchnorm, unfreeze_batchnorm
from ._common import load_weights_into, select_last_n_layer_modules
from .ssl_models import build_ssl_model_from_config

logger = logging.getLogger(__name__)


class HoloClassifierV2(nn.Module):
    """Classification model: frozen SSL encoders + trainable head.

    Input:  (view_a, view_b, fl)
    Output: logits (B, num_classes)

    Config (architecture_setup.classification_model):
        feature_l2_normalization_enabled (bool, default True)
        classifier_type   'one_layer' (default) | 'two_layers'
        classifier_hidden_dim / classifier_dropout / classifier_layernorm

    Fine-tuning policy (train.fine_tuning, applied by the builder via
    _apply_fine_tuning_policy()):
        unfreeze_image_encoder / unfreeze_fluorescence_encoder (bool)
            -> full end-to-end fine-tuning of that encoder
        unfreeze_last_img_layers / unfreeze_last_fl_layers (int >= 1)
            -> only the last N layer units of that encoder are trainable
        BatchNorm state has two independent knobs (encoder-wide, composing with
        the full / last-N unfreeze above):
            update_batchnorm_stats_{img,fl}_encoder -> keep that encoder's BN in
                train() mode so its running statistics (running_mean/var, which
                are buffers, NOT parameters) update; trains nothing
            train_batchnorm_affine_{img,fl}_encoder -> make that encoder's BN
                affine weight/bias trainable and add them to the optimizer
        unfreeze_batchnorm_{img,fl}_encoder is a convenience "both" shortcut
        (running stats update + affine trainable).
        (default: both encoders fully frozen, only the head trains)
    """

    def __init__(self, trained_ssl_model: nn.Module, config: Dict[str, Any], num_classes: int):
        """
        Initialize the classifier with properly frozen encoders.

        Args:
            trained_ssl_model: SSL model exposing .img_encoder and .fl_encoder
            config: configuration dict (same format as the legacy classifier)
            num_classes: number of output classes
        """
        super(HoloClassifierV2, self).__init__()

        # --- Inherit encoders from the (pretrained) SSL model ---
        self.img_encoder = trained_ssl_model.img_encoder
        self.fl_encoder = trained_ssl_model.fl_encoder

        # --- Freeze encoders properly (see _freeze_encoder) ---
        self._freeze_encoder(self.img_encoder, "Image Encoder")
        self._freeze_encoder(self.fl_encoder, "Fluorescence Encoder")

        # Feature dimensions
        img_feature_dim = self.img_encoder.get_output_dim()
        fl_feature_dim = self.fl_encoder.get_output_dim()

        # L2 normalisation before concatenation
        self.l2_normalization = config['architecture_setup']['classification_model'].get(
            'feature_l2_normalization_enabled', True)

        # Concatenated feature dimension: img_a + img_b + fl
        concatenated_dim = img_feature_dim * 2 + fl_feature_dim

        self.classifier = self._build_classifier_head(
            config['architecture_setup'], concatenated_dim, num_classes)

        # --- Fine-tuning policy state (per encoder) ---
        # 'frozen'  : requires_grad=False + eval + BN running stats frozen
        # 'full'    : requires_grad=True + train mode + BN stats live
        # 'last_n'  : only the selected trailing layer units are trainable
        # (self._ft_units holds the selected modules for the 'last_n' case)
        self._ft_policy = {'img': 'frozen', 'fl': 'frozen'}
        self._ft_units = {'img': [], 'fl': []}
        # Independent per-encoder BatchNorm controls. BN has two kinds of state:
        # affine params (weight/bias: trainable Parameters) and running stats
        # (running_mean/var: buffers updated in the forward pass, never trained).
        #   _bn_stats[which]  -> BN kept in train() so running stats update.
        #   _bn_affine[which] -> BN affine weight/bias trainable + optimized.
        # Fed from update_batchnorm_stats_* / train_batchnorm_affine_* (and the
        # unfreeze_batchnorm_* "both" shortcut). get_fine_tuning_param_groups()
        # re-derives the same affine params, so model and optimizer stay in sync.
        self._bn_stats = {'img': False, 'fl': False}
        self._bn_affine = {'img': False, 'fl': False}

        # --- View fusion (speed) ---
        # When the image encoder is completely frozen (params + every submodule in
        # eval mode), encoding view_a and view_b in ONE batched forward is
        # numerically identical to two separate forwards: with BatchNorm/Dropout in
        # eval mode every sample is processed independently, so concatenating along
        # the batch dimension changes nothing but the kernel launches (roughly 1.3-1.6x
        # faster encoder pass for the tiny backbones, which are launch-bound).
        # It is skipped automatically as soon as any part of the encoder is trainable
        # or in train mode (BN updating / dropout drawing), because then the batch
        # composition DOES matter.
        self.fuse_frozen_image_views = bool(
            config['architecture_setup']['classification_model'].get(
                'fuse_frozen_image_views', True))
        self._fusion_logged = False

        logger.info("[HoloClassifierV2] img_dim=%s fl_dim=%s concatenated_dim=%s "
                    "l2_norm=%s encoders_frozen=True fuse_frozen_image_views=%s",
                    img_feature_dim, fl_feature_dim, concatenated_dim,
                    self.l2_normalization, self.fuse_frozen_image_views)

    def image_encoder_fusable(self) -> bool:
        """True when the two image views may be encoded in a single forward pass.

        Requires the image encoder to be frozen AND fully in eval mode. Any module
        still in train mode disqualifies fusion: a BatchNorm there would normalise
        across the concatenated batch, and dropout would draw a different number of
        mask values for a 2B batch than for two B batches.
        """
        if not self.fuse_frozen_image_views:
            return False
        if self._ft_policy.get('img', 'frozen') != 'frozen':
            return False
        if any(param.requires_grad for param in self.img_encoder.parameters()):
            return False
        return not any(module.training for module in self.img_encoder.modules())

    @staticmethod
    def _build_classifier_head(arch_setup: Dict[str, Any],
                               concatenated_dim: int, num_classes: int) -> nn.Sequential:
        """Build the trainable classification head from config.

        ``classifier_layernorm`` is authoritative: it toggles the LayerNorm on the
        concatenated feature input for BOTH the ``one_layer`` and ``two_layers``
        heads, so the config flag always matches the built architecture.
        """
        conf = arch_setup['classification_model']
        classifier_type = conf.get('classifier_type', 'one_layer')
        use_layernorm = bool(conf.get('classifier_layernorm', False))
        dropout = conf.get('classifier_dropout', 0.25)

        if classifier_type == "two_layers":
            hidden_dim = conf.get('classifier_hidden_dim', 256)
            head = []
            if use_layernorm:
                head.append(nn.LayerNorm(concatenated_dim))
            head += [
                nn.Linear(concatenated_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            ]
        else:  # 'one_layer'
            head = []
            if use_layernorm:
                head.append(nn.LayerNorm(concatenated_dim))
                head.append(nn.Dropout(dropout))
            head.append(nn.Linear(concatenated_dim, num_classes))

        return nn.Sequential(*head)

    def _freeze_encoder(self, encoder: nn.Module, name: str) -> None:
        """
        Properly freeze an encoder module.

        1. Set requires_grad=False for all parameters
        2. Put the entire encoder in eval() mode
        3. Freeze all BatchNorm layers (eval() + keep track_running_stats), so
           they normalise with the pretrained running statistics and cannot
           drift while the encoder is frozen (eval() blocks updates)
        """
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        bn_count = freeze_batchnorm(encoder)
        logger.info("[HoloClassifierV2] Frozen %s: requires_grad=False, eval mode, %d BatchNorm layer(s) frozen (pretrained running stats)",
                    name, bn_count)

    def _enforce_encoder_modes(self) -> None:
        """Enforce per-encoder train/eval state according to the policy.

        Only meaningful while the model itself is in training mode; during
        validation/inference the global eval() state is authoritative
        (deterministic dropout / stored BN statistics).

        Running-stat adaptation (``update_batchnorm_stats_*`` / the
        ``unfreeze_batchnorm_*`` shortcut) is honoured independently of the
        weight freeze policy: when enabled, the encoder's BN layers are forced
        back into train mode so their running statistics (buffers) update with
        the new distribution, while the (frozen) conv/linear weights stay in
        eval mode. ``train_batchnorm_affine_*`` alone leaves BN in eval mode
        (running stats fixed) — it only makes the affine params trainable, which
        the optimizer handles.
        """
        if not self.training:
            return
        for which in ('img', 'fl'):
            policy = self._ft_policy[which]
            encoder = self.img_encoder if which == 'img' else self.fl_encoder
            if policy == 'frozen':
                encoder.eval()
            elif policy == 'last_n':
                # Frozen prefix stays in eval; trainable suffix in train.
                encoder.eval()
                for unit in self._ft_units[which]:
                    unit.train()
            # 'full': model.train() already puts the encoder in train mode.
            if self._bn_stats.get(which, False):
                self._set_bn_train(encoder, True)

    @staticmethod
    def _set_bn_train(encoder: nn.Module, train: bool) -> None:
        """Set all BatchNorm layers of ``encoder`` to train/eval mode."""
        for module in encoder.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.train() if train else module.eval()

    @staticmethod
    def _set_bn_track_running(encoder: nn.Module, value: bool) -> None:
        """Enable/disable running-stat tracking on every BatchNorm layer.

        Frozen encoders already keep ``track_running_stats=True`` (see
        :func:`freeze_batchnorm`), so this is effectively idempotent for the
        enabled case. Tracking is required to *update* running stats (stats
        mode) and to normalise with live statistics when only the affine params
        are trained (affine mode keeps BN in eval() -> deterministic inference).
        """
        for module in encoder.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.track_running_stats = value

    @staticmethod
    def _set_bn_affine_trainable(encoder: nn.Module, value: bool) -> None:
        """Set ``requires_grad`` on every BatchNorm affine param (weight/bias)."""
        for module in encoder.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                for param in module.parameters(recurse=False):
                    param.requires_grad = value

    def enforce_frozen_modes(self) -> None:
        """Public entry point so trainers can re-apply the per-encoder state."""
        self._enforce_encoder_modes()

    @property
    def _encoders_frozen(self) -> bool:
        """Legacy flag: True when both encoders are fully frozen."""
        return (self._ft_policy['img'] == 'frozen'
                and self._ft_policy['fl'] == 'frozen')

    def forward(self, view_a: torch.Tensor, view_b: torch.Tensor, fl: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with enforced per-encoder train/eval state.

        Args:
            view_a: first image view (B, 1, H, W)
            view_b: second image view (B, 1, H, W)
            fl: fluorescence features (B, F)

        Returns:
            logits: class predictions (B, num_classes)
        """
        # Frozen encoders stay in eval even after model.train(); partially
        # unfrozen encoders keep their trainable suffix in train mode.
        self._enforce_encoder_modes()

        if self.image_encoder_fusable():
            # ONE batched encode for both views (identical result while the image
            # encoder is frozen + in eval mode; see image_encoder_fusable()).
            if not self._fusion_logged:
                logger.info("[HoloClassifierV2] fusing view_a+view_b into one encoder "
                            "forward (image encoder frozen and in eval mode)")
                self._fusion_logged = True
            both = self.img_encoder(torch.cat((view_a, view_b), dim=0))
            n_a = view_a.shape[0]
            feat_a, feat_b = both[:n_a], both[n_a:]
        else:
            feat_a = self.img_encoder(view_a)
            feat_b = self.img_encoder(view_b)
        feat_fl = self.fl_encoder(fl)

        if self.l2_normalization:
            feat_a = F.normalize(feat_a, p=2, dim=1)
            feat_b = F.normalize(feat_b, p=2, dim=1)
            feat_fl = F.normalize(feat_fl, p=2, dim=1)

        combined_features = torch.cat((feat_a, feat_b, feat_fl), dim=1)
        return self.classifier(combined_features)

    # ------------------------------------------------------------------
    # Fine-tuning policy
    # ------------------------------------------------------------------
    def _unfreeze(self, which: str) -> None:
        """Fully unfreeze one encoder (params + BN stats + train mode)."""
        encoder = self.img_encoder if which == 'img' else self.fl_encoder
        for param in encoder.parameters():
            param.requires_grad = True
        unfreeze_batchnorm(encoder)
        encoder.train()
        self._ft_policy[which] = 'full'
        self._ft_units[which] = []

    def _unfreeze_last_n(self, which: str, n: int) -> None:
        """Unfreeze only the last N layer units of one encoder."""
        encoder = self.img_encoder if which == 'img' else self.fl_encoder
        units = select_last_n_layer_modules(encoder, n)
        if not units:
            logger.warning(
                "[HoloClassifierV2] %s encoder has no parameter-bearing "
                "layers to partially unfreeze; leaving it fully frozen.",
                which)
            return
        for unit in units:
            for param in unit.parameters():
                param.requires_grad = True
        unfreeze_batchnorm(*units)
        for unit in units:
            unit.train()
        self._ft_policy[which] = 'last_n'
        self._ft_units[which] = units
        logger.info(
            "[HoloClassifierV2] %s encoder: unfroze last %d layer unit(s): %s",
            which, n, [type(u).__name__ for u in units])

    def unfreeze_encoders(self, image: bool = True, fl: bool = True) -> None:
        """
        Utility to unfreeze the encoders for end-to-end fine-tuning experiments.

        Args:
            image: unfreeze the image encoder (default True).
            fl: unfreeze the fluorescence encoder (default True).

        NOTE: not recommended for the standard workflow (it defeats the point
        of pretrained encoders); provided for advanced users.
        """
        if image:
            self._unfreeze('img')
        if fl:
            self._unfreeze('fl')
        if image or fl:
            logger.warning(
                "[HoloClassifierV2] Encoders unfrozen (image=%s, fl=%s). Use with caution.",
                image, fl)

    def _apply_fine_tuning_policy(self, config) -> None:
        """
        Apply the `train.fine_tuning` configuration to the model state.

        Must stay consistent with get_fine_tuning_param_groups() (classification
        mode): the same config keys select the same encoder parameters.
        """
        ft = {}
        try:
            ft = config.train.fine_tuning or {}
        except (AttributeError, KeyError, TypeError):
            ft = {}

        img_full = bool(ft.get('unfreeze_image_encoder', False))
        fl_full = bool(ft.get('unfreeze_fluorescence_encoder', False))
        img_last_n = _as_positive_int(ft.get('unfreeze_last_img_layers', False))
        fl_last_n = _as_positive_int(ft.get('unfreeze_last_fl_layers', False))
        # BatchNorm flags (encoder-wide). unfreeze_batchnorm_* is a "both"
        # shortcut; the granular keys split running-stat updates (buffers) from
        # trainable affine params.
        img_bn_both = bool(ft.get('unfreeze_batchnorm_img_encoder', False))
        fl_bn_both = bool(ft.get('unfreeze_batchnorm_fl_encoder', False))
        img_bn_stats = img_bn_both or bool(ft.get('update_batchnorm_stats_img_encoder', False))
        fl_bn_stats = fl_bn_both or bool(ft.get('update_batchnorm_stats_fl_encoder', False))
        img_bn_affine = img_bn_both or bool(ft.get('train_batchnorm_affine_img_encoder', False))
        fl_bn_affine = fl_bn_both or bool(ft.get('train_batchnorm_affine_fl_encoder', False))

        if img_full:
            self._unfreeze('img')
        elif img_last_n > 0:
            self._unfreeze_last_n('img', img_last_n)

        if fl_full:
            self._unfreeze('fl')
        elif fl_last_n > 0:
            self._unfreeze_last_n('fl', fl_last_n)

        # BatchNorm controls are independent of (and compose with) the weight-
        # freeze policy. They are only ever turned ON here: full/last-N unfreeze
        # already turned them on, and get_fine_tuning_param_groups() re-derives
        # the same affine params, so model state and optimizer grouping match.
        self._bn_stats = {'img': img_bn_stats, 'fl': fl_bn_stats}
        self._bn_affine = {'img': img_bn_affine, 'fl': fl_bn_affine}
        # track_running_stats is needed to update stats AND to have live running
        # stats to use when only the affine params train (BN then stays in eval).
        if img_bn_stats or img_bn_affine:
            self._set_bn_track_running(self.img_encoder, True)
        if fl_bn_stats or fl_bn_affine:
            self._set_bn_track_running(self.fl_encoder, True)
        if img_bn_affine:
            self._set_bn_affine_trainable(self.img_encoder, True)
        if fl_bn_affine:
            self._set_bn_affine_trainable(self.fl_encoder, True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_CLASSIFICATION_MODELS = {
    "HoloClassifierV2": HoloClassifierV2,
}


def _section(config, *keys) -> Dict[str, Any]:
    """Walk nested dict/EasyDict keys; returns {} when any level is missing."""
    node = config
    for key in keys:
        if node is None:
            return {}
        value = node.get(key) if hasattr(node, 'get') else node[key]
        node = value
    return node or {}


def _enable_all_trainable(config) -> None:
    """'Train-from-scratch' normalisation: make both encoders trainable.

    In ``'none'`` mode no pretrained weights exist to protect, so the sensible
    default is end-to-end training (encoders + head). We flip the two
    ``train.fine_tuning.unfreeze_*_encoder`` flags to ``True`` because that is
    the single source of truth read by BOTH
    ``HoloClassifierV2._apply_fine_tuning_policy`` and
    ``get_fine_tuning_param_groups()`` — so the model's ``requires_grad`` state
    and the optimizer parameter groups stay consistent.

    We deliberately do NOT override the LR multipliers (that respects the user's
    config); instead we emit a prominent warning, because a fine-tuning-style
    tiny multiplier (e.g. ``0.001``) would leave the encoders barely learning on
    a from-scratch model.
    """
    # NOTE: EasyDict copies a plain dict on __setitem__, so we must either
    # mutate the stored object in place or re-read it after assigning -- never
    # keep writing to the local dict we passed in.
    train = config.get('train') if hasattr(config, 'get') else None
    if train is None:
        config['train'] = {}
    train = config['train']  # re-fetch the authoritative stored object

    ft = train.get('fine_tuning')
    if not ft:
        train['fine_tuning'] = {
            'unfreeze_image_encoder': True,
            'unfreeze_fluorescence_encoder': True,
        }
    else:
        ft['unfreeze_image_encoder'] = True
        ft['unfreeze_fluorescence_encoder'] = True

    ft = train['fine_tuning']  # re-fetch the stored object (authoritative)
    img_mult = ft.get('image_encoder_lr_multiplier')
    fl_mult = ft.get('fluorescence_encoder_lr_multiplier')
    logger.warning(
        "[V2 Builder] TRAIN-FROM-SCRATCH mode (model_initialization.pretraining.enable: false): "
        "no pretrained weights were loaded, so both encoders + the classification head are "
        "trainable. Learning rate: for a from-scratch model the encoder LR should be comparable "
        "to the head LR (set image_encoder_lr_multiplier / fluorescence_encoder_lr_multiplier to "
        "~1.0). Current multipliers: image=%s, fluorescence=%s — a fine-tuning-style tiny value "
        "(e.g. 0.001) would leave the encoders barely learning.",
        img_mult, fl_mult,
    )


def _as_positive_int(value) -> int:
    """Coerce a config value to a non-negative int (0 when falsy/invalid).

    Handles YAML bools (False), ints, and numeric strings ('3').
    """
    if value is None or value is False or str(value).strip().lower() in ('false', 'none', ''):
        return 0
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return max(n, 0)


def _discover_weights(exp_dir: Optional[str], weights_type: str, suffix: str) -> Optional[str]:
    """Find a '{weights_type}{suffix}' checkpoint (e.g. 'last_ssl_model.pth') in exp_dir.

    Returns the full path of the first match (sorted), or None if not found.
    """
    if not exp_dir or not os.path.isdir(exp_dir):
        return None
    prefix = f"{weights_type}{suffix}"
    candidates = sorted(f for f in os.listdir(exp_dir) if f.startswith(prefix))
    if not candidates:
        return None
    return os.path.join(exp_dir, candidates[0])


def _checkpoint_provenance(weights_path: Optional[str]) -> Dict[str, Any]:
    """Identity of a checkpoint file: modification time, age, size, epoch, BN steps.

    ``model_initialization.pretraining.weights: last`` points at a file the source
    experiment rewrites at the end of EVERY epoch.  Two runs started a few minutes
    apart can therefore initialise from different weights, which silently
    invalidates any A/B comparison between them.  The BatchNorm
    ``num_batches_tracked`` buffer is an exact fingerprint of how much training the
    source run had done (it increments once per train-mode BatchNorm forward), so
    it is reported next to the file's modification time.

    Never raises: an unreadable checkpoint simply yields fewer fields.
    """
    info: Dict[str, Any] = {}
    if not weights_path or not os.path.isfile(weights_path):
        return info
    try:
        stat = os.stat(weights_path)
        info["mtime"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        info["age_seconds"] = max(0.0, time.time() - stat.st_mtime)
        info["size_mb"] = stat.st_size / (1024.0 * 1024.0)
    except OSError:
        return info
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            epoch = checkpoint.get("epoch")
            if isinstance(epoch, (int, float)) and not isinstance(epoch, bool):
                info["epoch"] = int(epoch)
            state = checkpoint.get("model_state_dict", checkpoint)
            if isinstance(state, dict):
                counters = [int(v.item()) for k, v in state.items()
                            if isinstance(v, torch.Tensor) and k.endswith("num_batches_tracked")]
                if counters:
                    info["bn_num_batches_tracked"] = max(counters)
    except Exception as exc:  # pragma: no cover - metadata only, never fatal
        logger.debug("[V2 Builder] Could not read checkpoint metadata from %r: %s",
                     weights_path, exc)
    return info


def _log_initialization_provenance(mode: str, weights_path: Optional[str],
                                   stale_after_seconds: float = 120.0) -> None:
    """Log which checkpoint the model is really initialised from (and warn if moving).

    ``resume`` only logs the provenance; the pretraining modes additionally warn
    when the file is younger than ``stale_after_seconds`` - a strong hint that the
    source experiment is still running and overwriting it.
    """
    info = _checkpoint_provenance(weights_path)
    if not info:
        return
    logger.info(
        "[V2 Builder] Initialising from checkpoint %s | modified %s (%.0f s ago), "
        "%.1f MB, source epoch=%s, source BN num_batches_tracked=%s",
        weights_path, info.get("mtime", "?"), info.get("age_seconds", -1.0),
        info.get("size_mb", -1.0), info.get("epoch", "?"),
        info.get("bn_num_batches_tracked", "?"))
    age = info.get("age_seconds")
    if mode in ("ssl_pretrain", "cls_finetune") and age is not None \
            and age < stale_after_seconds:
        logger.warning(
            "[V2 Builder][STALE-INIT] The initialisation checkpoint %s was written %.0f "
            "seconds ago: the experiment that saves it is very likely STILL RUNNING, and "
            "'weights: last' is overwritten at the end of every epoch. A second run started "
            "later would initialise from DIFFERENT weights, so comparing the two runs would "
            "mix your settings under test with a different initialisation. Wait for the "
            "source run to finish, or copy this file into a private directory and point "
            "model_initialization.pretraining.experiment_path at the copy "
            "(or use 'weights: best').",
            weights_path, age)


def _resolve_initialization(config) -> Tuple[str, Optional[str]]:
    """
    Decide how the classification model is initialised, from `model_initialization`.

    Returns:
        (mode, weights_path) with mode one of:
            'none'         - fresh build, no weights (training from scratch)
            'ssl_pretrain' - SSL weights into the base model
            'cls_finetune' - previous classification weights into the full model
            'resume'       - resume checkpoint into the full model (trainer-driven)
    """
    model_init = _section(config, 'model_initialization')
    if not model_init:
        return 'none', None

    # Resume: checkpoint of a previous classification run.
    resume = _section(model_init, 'resume')
    if resume.get('enable', False):
        weights_path = resume.get('checkpoint_path')
        if not weights_path:
            raise ValueError(
                "model_initialization.resume.enable is true but "
                "'model_initialization.resume.checkpoint_path' is empty."
            )
        return 'resume', weights_path

    # Pretraining section (SSL stage-1 weights or a previous classifier).
    pre = _section(model_init, 'pretraining')
    if not pre.get('enable', False):
        return 'none', None

    weights_type = pre.get('weights', 'last')
    exp_path = pre.get('experiment_path')

    if pre.get('mode', 'ssl_pretraining') == 'ssl_pretraining':
        suffix = '_ssl_model.pth'
        mode = 'ssl_pretrain'
        label = "SSL pretraining"
    else:
        suffix = '_classification_model.pth'
        mode = 'cls_finetune'
        label = "classification finetuning"

    weights_path = _discover_weights(exp_path, weights_type, suffix)
    if weights_path is None:
        raise ValueError(
            f"{label} is enabled but no '{weights_type}{suffix}' checkpoint was "
            f"found in: {exp_path!r}"
        )
    return mode, weights_path


def _ensure_loaded(result, required_prefixes, weights_path, label):
    """Fail loudly if a required key group was missing from the checkpoint.

    Loading is non-strict so the (expected) SSL projector / temperature keys may
    be absent. But the *encoders* (and, in the full-model modes, the classifier
    head) must always come from the checkpoint: if they are missing, the
    experiment bundle's architecture does not match its own weights, and we
    would silently train with random weights. Better to abort with a clear,
    actionable error.

    The ``classifier.`` group is exempt when ``load_weights_into`` skipped it
    on purpose (``result.classifier_skipped``): the checkpoint's num_classes
    differs from the target, so the head is meant to stay randomly
    initialized.
    """
    if getattr(result, "classifier_skipped", False):
        required_prefixes = tuple(
            prefix for prefix in required_prefixes
            if not prefix.startswith("classifier")
        )

    bad = {}

    for prefix in required_prefixes:
        missing = [k for k in (getattr(result, "missing_keys", None) or [])
                   if k.startswith(prefix)]
        if missing:
            bad[prefix] = missing
    if bad:
        detail = "; ".join(f"{p}: {len(v)} missing (e.g. {v[:2]})" for p, v in bad.items())
        raise RuntimeError(
            f"[V2 Builder] Could not fully load {label} weights from {weights_path!r} - "
            f"checkpoint is missing: {detail}. "
            f"This means the experiment bundle's architecture.yaml does not describe "
            f"the weights it holds. Point model_initialization at the matching "
            f"experiment directory."
        )


def build_classification_model_from_config_v2(config) -> nn.Module:
    """
    Build the classification model from the merged architecture config.

    Steps:
        1. resolve the initialisation mode ('none' | 'ssl_pretrain' |
           'cls_finetune' | 'resume') from `model_initialization`,
        2. build the SSL base model (fresh encoders),
        3. for 'ssl_pretrain': load the SSL weights into the base model,
        4. wrap in HoloClassifierV2 (frozen encoders + trainable head),
        5. for 'cls_finetune' / 'resume': load the full-model weights,
        6. apply the train.fine_tuning policy (encoder freeze / unfreeze).

    Args:
        config: merged configuration object (EasyDict).

    Returns:
        HoloClassifierV2 instance.
    """
    mode, weights_path = _resolve_initialization(config)
    logger.info("[V2 Builder] Initialisation mode: %s | weights: %s", mode, weights_path or "-")
    # Record WHICH checkpoint file this run really starts from (and warn when the
    # source experiment may still be overwriting it) - see _checkpoint_provenance.
    _log_initialization_provenance(mode, weights_path)

    # 1b. Train-from-scratch ('none'): no pretrained weights exist to protect,
    #     so normalise the fine-tuning flags to make both encoders trainable.
    #     Both the model policy and the optimizer grouping read these same
    #     config.train.fine_tuning keys, so they stay in sync automatically.
    if mode == 'none':
        _enable_all_trainable(config)

    # 2. Build the SSL base model (encoders without weights for now).
    ssl_base_model = build_ssl_model_from_config(config)

    # 3. SSL stage-1 weights go into the base model (before freezing).
    if mode == 'ssl_pretrain':
        result = load_weights_into(ssl_base_model, weights_path, strict=False, label="V2 Builder")
        _ensure_loaded(result, ("img_encoder.", "fl_encoder."), weights_path, "SSL")

    # 4. Build the classification model.
    classification_conf = config.architecture_setup.classification_model
    model_name = classification_conf.get('name', '')
    if model_name not in _CLASSIFICATION_MODELS:
        raise ValueError(
            f"Unknown classification model: '{model_name}'. "
            f"Supported: {', '.join(_CLASSIFICATION_MODELS)}"
        )
    num_classes = classification_conf.get('num_classes')
    if num_classes is None:
        raise ValueError(
            "architecture_setup.classification_model.num_classes is required "
            "to build the classification model."
        )
    classification_model = _CLASSIFICATION_MODELS[model_name](
        ssl_base_model, config, num_classes)

    # 5. Full-model weights (previous classifier / resume checkpoint).
    if mode in ('cls_finetune', 'resume'):
        result = load_weights_into(classification_model, weights_path, num_classes=num_classes, strict=False, label="V2 Builder")
        if getattr(result, "classifier_skipped", False):
            logger.warning(
                "[V2 Builder] Checkpoint classifier does not match num_classes=%s "
                "and was NOT loaded; the classification head is randomly initialized.",
                num_classes)
        _ensure_loaded(result, ("img_encoder.", "fl_encoder.", "classifier."),
                       weights_path, "classification")

    # 6. Apply the fine-tuning policy (encoder freeze / unfreeze state,
    #    including BatchNorm adaptation via unfreeze_batchnorm_*_encoder) so
    #    the model's requires_grad / train-eval / BN-stat state matches the
    #    optimizer grouping produced by get_fine_tuning_param_groups().
    if hasattr(classification_model, '_apply_fine_tuning_policy'):
        classification_model._apply_fine_tuning_policy(config)

    return classification_model


# Export both the classifier and the explicit V2 builder
__all__ = [
    'HoloClassifierV2',
    'build_classification_model_from_config_v2',
]
