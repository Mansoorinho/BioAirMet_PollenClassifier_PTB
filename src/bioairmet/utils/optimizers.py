'''
@file    :   optimizers.py
@create date : 2025-05-13 11:20:45
@modify date 2026-02-26 15:25:56
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    This utility script provides specialized functions for setting up optimization routines in the BioAirMet project.
    - create_split_param_groups: Implements weight decay exclusion for 1D parameters (biases and BatchNormalization weights) to improve regularization.
    - get_fine_tuning_param_groups: Manages complex parameter grouping logic for both SSL and Classification modes. It allows for selective freezing of encoders and the application of differentiated learning rate multipliers across different model components (e.g., image encoder vs. classifier head).
    - build_optimizer_from_config: A factory function to instantiate popular optimizers like AdamW, Adam, and SGD based on project configuration.
    - build_scheduler_from_config: A factory function to set up various learning rate schedules, including CosineAnnealing, OneCycleLR, and ReduceLROnPlateau.
    - step_scheduler: Steps an LR scheduler while silencing one known-harmless torch ordering warning (see its docstring).
    ]
'''
import torch.optim as optim
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      ReduceLROnPlateau,
                                      StepLR, OneCycleLR,
                                      CosineAnnealingWarmRestarts)
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

from ..models._common import select_last_n_layer_modules

def create_split_param_groups(params, lr, weight_decay, bn_bias_no_decay=False):
    """
    Splits parameters into decay and no-decay groups based on their dimensionality.
    Parameters with ndim < 2 (biases, normalization weights) get 0 weight decay.
    """
    param_groups = []

    for param in params:
        if param.ndim < 2 and bn_bias_no_decay:
            param_groups.append({'params': [param], 'lr': lr, 'weight_decay': 0.0})
        else:
            param_groups.append({'params': [param], 'lr': lr, 'weight_decay': weight_decay})

    return param_groups


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


# ---------------------------------------------------------------------------
# Scheduler stepping rules — single source of truth for the trainers
# ---------------------------------------------------------------------------
# "Step-based" schedulers compute their LR from the number of OPTIMIZER steps
# already taken, so their ``step()`` must be called exactly once per optimizer
# step (i.e. once per gradient-accumulation window) and NEVER per epoch or
# before the training loop.  Everything else (CosineAnnealingLR, StepLR,
# ReduceLROnPlateau) is epoch-based: stepped once after validation.
STEP_BASED_SCHEDULER_NAMES = frozenset({
    "OneCycleLR",
    "CosineAnnealingWarmRestarts",
    "CLIPCosineWarmupScheduler",
})

EPOCH_BASED_SCHEDULER_NAMES = frozenset({
    "CosineAnnealingLR",
    "StepLR",
    "ReduceLROnPlateau",
})


def is_step_based_scheduler(scheduler) -> bool:
    """True if ``scheduler`` must be stepped once per optimizer step.

    Matches on the class itself and, as a fallback, on the class name so that
    resumed / re-wrapped schedulers (checkpoint loading, DDP) are still
    classified correctly.  ``None`` returns False.
    """
    if scheduler is None:
        return False
    if isinstance(scheduler, (OneCycleLR, CosineAnnealingWarmRestarts, CLIPCosineWarmupScheduler)):
        return True
    return type(scheduler).__name__ in STEP_BASED_SCHEDULER_NAMES


def is_epoch_based_scheduler(scheduler) -> bool:
    """True for schedulers stepped once per epoch (after validation).

    ``ReduceLROnPlateau`` is epoch-based but needs a metric, so callers must
    keep handling it separately (see ``base_trainer_v2.train``).
    """
    if scheduler is None:
        return False
    return not is_step_based_scheduler(scheduler)


def _is_unset(value) -> bool:
    """True for YAML-ish "not provided" values (None, False, '', 'none')."""
    return value is None or value is False or str(value).strip().lower() in ('', 'none', 'null')


def resolve_total_steps(config, scheduler_config=None, steps_per_epoch: int = 0,
                        verbose: bool = True):
    """Resolve the number of OPTIMIZER steps the schedule must span.

    Step-based schedulers are advanced once per OPTIMIZER step (see
    ``is_step_based_scheduler``), so the schedule length must be counted in
    optimizer steps, NOT in micro-batches: with
    ``train.gradient_accumulation_steps = A`` the trainer performs
    ``ceil(steps_per_epoch / A)`` optimizer steps per epoch.

    Precedence:
      1. ``train.scheduler.total_steps`` — explicit override, taken as an
         absolute optimizer-step count (no division applied).
      2. ``ceil(train.epochs * steps_per_epoch / gradient_accumulation_steps)``.

    Returns:
        tuple: (total_steps, source_description)

    Raises:
        ValueError: when the resolved count is not a positive integer.
    """
    if scheduler_config is None:
        scheduler_config = config.train.get('scheduler', None)

    grad_accum = _as_positive_int(config.train.get('gradient_accumulation_steps', 1))
    grad_accum = grad_accum if grad_accum > 0 else 1

    override = None if scheduler_config is None else scheduler_config.get('total_steps', None)
    if not _is_unset(override):
        total_steps = int(override)
        source = "train.scheduler.total_steps (explicit override)"
    else:
        epochs = int(config.train.epochs)
        spe = int(steps_per_epoch)
        if epochs <= 0 or spe <= 0:
            raise ValueError(
                f"Cannot resolve scheduler total_steps: train.epochs={epochs}, "
                f"steps_per_epoch={spe}. Both must be positive; pass "
                f"steps_per_epoch=len(train_loader) to build_scheduler_from_config()."
            )
        opt_steps_per_epoch = max(1, math.ceil(spe / grad_accum))
        total_steps = epochs * opt_steps_per_epoch
        source = (
            f"train.epochs ({epochs}) * optimizer_steps_per_epoch ({opt_steps_per_epoch}"
            f" = ceil(steps_per_epoch {spe} / gradient_accumulation_steps {grad_accum}))"
        )

    if total_steps <= 0:
        raise ValueError(f"Resolved scheduler total_steps must be > 0, got {total_steps} ({source})")
    if verbose:
        print(f"[scheduler] total_steps={total_steps}  [{source}]")
    return total_steps, source


def get_fine_tuning_param_groups(model, loss_module, config, mode='ssl'):
    """
    Generates parameter groups for fine-tuning, allowing selective freezing and
    different learning rates for different parts of the model.

    Behaviour:
        - All parameters are frozen by default, then selectively enabled.
        - 'ssl': image/fluorescence encoders (unless frozen via
          `freeze_image_encoder` / `freeze_fluorescence_encoder`), the
          projection heads / other params, and — when `train.loss.learnable_temp`
          is set — the loss module's learnable parameters (temperature in a
          dedicated zero-weight-decay group).
        - 'classification': only the classifier head (+ legacy 'img_fuser')
          by default; `unfreeze_image_encoder` / `unfreeze_fluorescence_encoder`
          add the full encoder, or `unfreeze_last_img_layers` /
          `unfreeze_last_fl_layers` (int >= 1) add only the last N layer units
          of that encoder (same selection as HoloClassifierV2).
          `train_batchnorm_affine_{img,fl}_encoder` (or the combined
          `unfreeze_batchnorm_{img,fl}_encoder` shortcut) add that encoder's
          BatchNorm affine params (weight/bias) to the optimizer so they train
          even when the rest of the encoder is frozen, mirroring the model's
          _bn_affine state. Running-stat updates (`update_batchnorm_stats_*`)
          touch buffers only, so they add nothing to the optimizer here.
        - A strict integrity check raises RuntimeError when any trainable
          parameter is missing from / duplicated across the groups.
        - If no group is resolved at all, classification mode raises
          (refusing to silently train everything); SSL mode trains everything.

    Args:
        model (nn.Module): The model to fine-tune.
        loss_module (nn.Module | None): Optional loss module whose learnable
            parameters are grouped (SSL mode only).
        config (EasyDict): Configuration object containing fine-tuning settings.
        mode (str): 'ssl' for self-supervised pre-training fine-tuning,
                    'classification' for supervised classification fine-tuning.

    Returns:
        list: A list of parameter groups suitable for a torch.optim.Optimizer.
    """
    param_groups = []

    # Default learning rate from config
    base_lr = config.train.optimizer.lr
    base_weight_decay = config.train.optimizer.get('weight_decay', 0.0)
    # Whether to exclude BN and bias from weight decay
    bn_bias_no_decay = config.train.fine_tuning.get('bn_bias_no_decay', False)

    # Collect all parameters and their names
    all_params = list(model.named_parameters())
    train_loss_params = bool(
        config.train.loss.get('learnable_temp', config.train.loss.get('learnable_temperature', False))
    )

    def _strip_ddp_prefix(param_name):
        # DDP adds 'module.' prefix; normalize names for robust matching
        return param_name[7:] if param_name.startswith('module.') else param_name

    # Initialize all parameters to not require gradients by default, then enable selectively
    # This is safer for classification mode where most parameters are frozen
    # for name, param in all_params:
    #     param.requires_grad = False

    if mode == 'ssl':
        # Legacy-compatible SSL grouping:
        # 1) image encoder params
        # 2) fluorescence encoder params
        # 3) all remaining params (e.g., projectors, logit_scale, etc.)
        #
        # Intentionally avoids per-parameter split groups so scheduler/optimizer
        # dynamics match old training behavior.
        freeze_image_encoder = config.train.fine_tuning.get('freeze_image_encoder', False)
        freeze_fluorescence_encoder = config.train.fine_tuning.get('freeze_fluorescence_encoder', False)

        image_encoder_lr_multiplier = config.train.fine_tuning.get('image_encoder_lr_multiplier', 1.0)
        fluorescence_encoder_lr_multiplier = config.train.fine_tuning.get('fluorescence_encoder_lr_multiplier', 1.0)
        projection_head_lr_multiplier = config.train.fine_tuning.get('projection_head_lr_multiplier', 1.0)
        temperature_lr_multiplier = config.train.fine_tuning.get('temperature_lr_multiplier', 0.1)
        loss_lr_multiplier = config.train.fine_tuning.get('loss_lr_multiplier', temperature_lr_multiplier)

        image_encoder_params = []
        fluorescence_encoder_params = []
        other_params = []
        loss_params = []
        temperature_params = []

        seen_param_ids = set()

        def _append_unique(bucket, param):
            param_id = id(param)
            if param_id not in seen_param_ids:
                bucket.append(param)
                seen_param_ids.add(param_id)

        def _is_temperature_param(param_name):
            lname = param_name.lower()
            keys = ('logit_scale', 'logit_bias', 'log_temperature', 'temperature')
            return any(k in lname for k in keys)

        for raw_name, param in all_params:
            name = _strip_ddp_prefix(raw_name)

            # Keep learnable temperature / scale parameters in a dedicated
            # zero-weight-decay group for stability and legacy compatibility.
            if _is_temperature_param(name):
                param.requires_grad = True
                _append_unique(temperature_params, param)
                continue

            if 'img_encoder' in name:
                if not freeze_image_encoder:
                    param.requires_grad = True
                    _append_unique(image_encoder_params, param)
            elif 'fl_encoder' in name:
                if not freeze_fluorescence_encoder:
                    param.requires_grad = True
                    _append_unique(fluorescence_encoder_params, param)
            # projection heads
            elif ('img_projector' in name) or ('fl_projector' in name):
                param.requires_grad = True
                _append_unique(other_params, param)
                
            else:
                param.requires_grad = True
                _append_unique(other_params, param)

        if image_encoder_params:
            param_groups.append({
                'params': image_encoder_params,
                'lr': base_lr * image_encoder_lr_multiplier,
                'weight_decay': base_weight_decay,
            })

        if fluorescence_encoder_params:
            param_groups.append({
                'params': fluorescence_encoder_params,
                'lr': base_lr * fluorescence_encoder_lr_multiplier,
                'weight_decay': base_weight_decay,
            })

        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr*projection_head_lr_multiplier,
                'weight_decay': base_weight_decay,
            })

        # Optional learnable parameters from the loss module (e.g., temperature)
        # should use zero weight decay and their own LR multiplier.
        if train_loss_params and loss_module is not None:
            for loss_name, param in loss_module.named_parameters():
                if not param.requires_grad:
                    continue
                if _is_temperature_param(loss_name):
                    _append_unique(temperature_params, param)
                else:
                    _append_unique(loss_params, param)

        if loss_params:
            param_groups.append({
                'params': loss_params,
                'lr': base_lr * loss_lr_multiplier,
                'weight_decay': 0.0,
            })

        if temperature_params:
            param_groups.append({
                'params': temperature_params,
                'lr': base_lr * temperature_lr_multiplier,
                'weight_decay': 0.0,
            })
            
    elif mode == 'classification':
        for name, param in all_params:
            param.requires_grad = False
            
        # In classification mode, by default, image and fluorescence encoders are frozen.
        # Only the classifier head is trainable.
        # Keep the model's own state (requires_grad / train-eval / BN stats) in
        # sync with this grouping — idempotent, and works through a DDP wrapper.
        if hasattr(model, '_apply_fine_tuning_policy'):
            model._apply_fine_tuning_policy(config)
        unfreeze_image_encoder = config.train.fine_tuning.get(
            'unfreeze_image_encoder', False)
        unfreeze_fluorescence_encoder = config.train.fine_tuning.get(
            'unfreeze_fluorescence_encoder', False)
        # Partial fine-tuning: unfreeze only the last N layer units of an
        # encoder (0/False -> disabled). Uses the same selection rule as
        # HoloClassifierV2 (shared select_last_n_layer_modules helper).
        unfreeze_last_img_layers = _as_positive_int(
            config.train.fine_tuning.get('unfreeze_last_img_layers', False))
        unfreeze_last_fl_layers = _as_positive_int(
            config.train.fine_tuning.get('unfreeze_last_fl_layers', False))
        # BatchNorm affine training: when enabled, an encoder's BN affine params
        # (weight/bias) become trainable even if the rest of the encoder is
        # frozen. Mirrors HoloClassifierV2._apply_fine_tuning_policy's _bn_affine
        # state (unfreeze_batchnorm_* is a "both" shortcut). Running-stat updates
        # (update_batchnorm_stats_*) touch buffers only, so they are intentionally
        # not read here (they add no optimizer parameters).
        img_bn_affine = bool(
            config.train.fine_tuning.get('unfreeze_batchnorm_img_encoder', False)) or bool(
            config.train.fine_tuning.get('train_batchnorm_affine_img_encoder', False))
        fl_bn_affine = bool(
            config.train.fine_tuning.get('unfreeze_batchnorm_fl_encoder', False)) or bool(
            config.train.fine_tuning.get('train_batchnorm_affine_fl_encoder', False))

        def _find_module_by_name(root, target_name):
            for mod_name, mod in root.named_modules():
                stripped = mod_name[7:] if mod_name.startswith('module.') else mod_name
                if stripped == target_name:
                    return mod
            return None

        img_last_n_ids = set()
        if unfreeze_last_img_layers > 0:
            img_mod = _find_module_by_name(model, 'img_encoder')
            if img_mod is None:
                raise RuntimeError(
                    "unfreeze_last_img_layers is set, but no 'img_encoder' "
                    "module was found on the model.")
            for unit in select_last_n_layer_modules(img_mod, unfreeze_last_img_layers):
                img_last_n_ids.update(id(p) for p in unit.parameters())

        fl_last_n_ids = set()
        if unfreeze_last_fl_layers > 0:
            fl_mod = _find_module_by_name(model, 'fl_encoder')
            if fl_mod is None:
                raise RuntimeError(
                    "unfreeze_last_fl_layers is set, but no 'fl_encoder' "
                    "module was found on the model.")
            for unit in select_last_n_layer_modules(fl_mod, unfreeze_last_fl_layers):
                fl_last_n_ids.update(id(p) for p in unit.parameters())

        def _bn_param_ids(module):
            ids = set()
            if module is None:
                return ids
            for m in module.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    ids.update(id(p) for p in m.parameters())
            return ids

        img_bn_ids = _bn_param_ids(
            _find_module_by_name(model, 'img_encoder')
            if img_bn_affine else None)
        fl_bn_ids = _bn_param_ids(
            _find_module_by_name(model, 'fl_encoder')
            if fl_bn_affine else None)
        # bn_bias_no_decay = config.train.fine_tuning.get('bn_bias_no_decay', True) # Whether to exclude BN and bias from weight decay

        image_encoder_lr_multiplier = config.train.fine_tuning.get(
            'image_encoder_lr_multiplier', 1.0)
        fluorescence_encoder_lr_multiplier = config.train.fine_tuning.get(
            'fluorescence_encoder_lr_multiplier', 1.0)
        classifier_lr_multiplier = config.train.fine_tuning.get(
            'classifier_lr_multiplier', 1.0)
        img_fuser_lr_multiplier = config.train.fine_tuning.get(
            'img_fuser_lr_multiplier', 0.01)

        image_encoder_params = []
        fluorescence_encoder_params = []
        classifier_params = []
        img_fuser_params = []
        seen_param_ids = set()

        def _append_unique(bucket, param):
            param_id = id(param)
            if param_id not in seen_param_ids:
                bucket.append(param)
                seen_param_ids.add(param_id)

        for raw_name, param in all_params:
            name = _strip_ddp_prefix(raw_name)

            if 'img_encoder' in name:
                if (unfreeze_image_encoder
                        or (img_last_n_ids and id(param) in img_last_n_ids)
                        or (img_bn_ids and id(param) in img_bn_ids)):
                    param.requires_grad = True
                    _append_unique(image_encoder_params, param)

            if 'fl_encoder' in name:
                if (unfreeze_fluorescence_encoder
                        or (fl_last_n_ids and id(param) in fl_last_n_ids)
                        or (fl_bn_ids and id(param) in fl_bn_ids)):
                    param.requires_grad = True
                    _append_unique(fluorescence_encoder_params, param)

            if 'classifier' in name:
                param.requires_grad = True
                _append_unique(classifier_params, param)

            if 'img_fuser' in name:
                param.requires_grad = True
                _append_unique(img_fuser_params, param)

        if image_encoder_params:
            param_groups.extend(
                create_split_param_groups(image_encoder_params,
                                          base_lr * image_encoder_lr_multiplier,
                                          base_weight_decay,
                                          bn_bias_no_decay=bn_bias_no_decay))
        if fluorescence_encoder_params:
            param_groups.extend(
                create_split_param_groups(fluorescence_encoder_params,
                                          base_lr * fluorescence_encoder_lr_multiplier,
                                          base_weight_decay,
                                          bn_bias_no_decay=bn_bias_no_decay))
        if classifier_params:
            param_groups.extend(
                create_split_param_groups(classifier_params,
                                          base_lr * classifier_lr_multiplier,
                                          base_weight_decay,
                                          bn_bias_no_decay=bn_bias_no_decay))
        if img_fuser_params:
            param_groups.extend(
                create_split_param_groups(img_fuser_params,
                                          base_lr * img_fuser_lr_multiplier,
                                          base_weight_decay,
                                          bn_bias_no_decay=bn_bias_no_decay))

    else:
        raise ValueError(
            f"Unknown fine-tuning mode: {mode}. Expected 'ssl' or 'classification'.")

    # Ensure at least one parameter group is created, otherwise optimizer will fail
    if not param_groups:
        if mode == 'classification':
            # In classification mode an empty grouping means the model exposes
            # no known trainable head ('classifier' / 'img_fuser') — silently
            # training *everything* (e.g. randomly-initialised encoders) is far
            # worse than failing fast.
            raise RuntimeError(
                "No trainable parameter groups were resolved for classification "
                "mode: the model exposes no module whose name contains "
                "'classifier' or 'img_fuser'. Refusing to fall back to "
                "training all parameters (e.g. randomly-initialised encoders). "
                "Check the model architecture and the 'train.fine_tuning' config."
            )
        # SSL fallback: training everything is a safe default for pre-training.
        print("Warning: No specific parameter groups defined for fine-tuning. Training all parameters with base LR.")
        all_trainable = []
        fallback_ids = set()
        for _, param in all_params:
            if param.requires_grad and id(param) not in fallback_ids:
                all_trainable.append(param)
                fallback_ids.add(id(param))

        if mode == 'ssl' and train_loss_params and loss_module is not None:
            for param in loss_module.parameters():
                if param.requires_grad and id(param) not in fallback_ids:
                    all_trainable.append(param)
                    fallback_ids.add(id(param))

        param_groups.extend(create_split_param_groups(
            all_trainable, base_lr, base_weight_decay, bn_bias_no_decay=bn_bias_no_decay))

    # Safety check: ensure all expected trainable params are covered exactly once.
    grouped_param_ids = []
    for group in param_groups:
        grouped_param_ids.extend(id(param) for param in group.get('params', []))

    grouped_param_id_set = set(grouped_param_ids)
    duplicate_count = len(grouped_param_ids) - len(grouped_param_id_set)

    expected_trainable_ids = set()
    trainable_name_by_id = {}

    for name, param in all_params:
        if param.requires_grad:
            param_id = id(param)
            expected_trainable_ids.add(param_id)
            trainable_name_by_id.setdefault(param_id, f"model::{name}")

    if mode == 'ssl' and train_loss_params and loss_module is not None:
        for name, param in loss_module.named_parameters():
            if param.requires_grad:
                param_id = id(param)
                expected_trainable_ids.add(param_id)
                trainable_name_by_id.setdefault(param_id, f"loss::{name}")

    missing_trainable = expected_trainable_ids - grouped_param_id_set
    unexpected_grouped = grouped_param_id_set - expected_trainable_ids

    if missing_trainable or unexpected_grouped or duplicate_count > 0:
        missing_names = [trainable_name_by_id.get(pid, f"id={pid}")
                         for pid in list(missing_trainable)[:10]]
        unexpected_ids = [str(pid) for pid in list(unexpected_grouped)[:10]]

        print(
            "WARNING: Parameter grouping mismatch detected before optimizer build: "
            f"expected_trainable={len(expected_trainable_ids)}, "
            f"grouped_total={len(grouped_param_ids)}, "
            f"grouped_unique={len(grouped_param_id_set)}, "
            f"missing={len(missing_trainable)}, "
            f"unexpected={len(unexpected_grouped)}, "
            f"duplicates={duplicate_count}."
        )
        if missing_names:
            print(f"WARNING: Missing trainable parameters (first {len(missing_names)}): {missing_names}")
        if unexpected_ids:
            print(f"WARNING: Unexpected grouped parameter ids (first {len(unexpected_ids)}): {unexpected_ids}")

        raise RuntimeError(
            "Parameter grouping integrity check failed: "
            f"missing={len(missing_trainable)}, "
            f"unexpected={len(unexpected_grouped)}, "
            f"duplicates={duplicate_count}."
        )

    return param_groups

def build_optimizer_from_config(config, param_groups):
    """
    Builds an optimizer instance based on the provided configuration.

    Args:
        config (EasyDict): Configuration object.
        param_groups (list): A list of parameter groups, where each group is a dict
                             containing 'params' and optionally 'lr', 'weight_decay', etc.

    Returns:
        torch.optim.Optimizer: An instance of the specified optimizer.
    """
    optimizer_name = config.train.optimizer.name
    # Default LR and weight_decay will be applied if not specified in param_groups
    default_lr = config.train.optimizer.lr
    default_weight_decay = config.train.optimizer.get('weight_decay', 0.0)
    default_eps = config.train.optimizer.get('eps', 1e-8)  # Default epsilon for AdamW
    default_betas = config.train.optimizer.get('betas', [0.9, 0.999])  # Default betas for AdamW
    optimizer_name = optimizer_name.lower()
    # Apply default LR and weight_decay to param_groups if not already specified
    for group in param_groups:
        group.setdefault('lr', default_lr)
        group.setdefault('weight_decay', default_weight_decay)
        # custom eps set to 2.5e-4 for AdamW to improve stability in SSL fine-tuning, especially with small batch sizes
        # add only for adamw
        if optimizer_name == "adamw":
            group.setdefault('eps', default_eps)
            group.setdefault('betas', default_betas)

    if optimizer_name == "adamw":
        return optim.AdamW(param_groups)
    elif optimizer_name == "adam":
        return optim.Adam(param_groups)
    elif optimizer_name == "sgd":
        default_momentum = config.train.optimizer.get('momentum', 0.9)
        for group in param_groups:
            group.setdefault('momentum', default_momentum)
        return optim.SGD(param_groups)
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")


def step_scheduler(scheduler, epoch=None):
    """
    Step an LR scheduler, silencing one known-harmless torch warning.

    PyTorch emits "Detected call of `lr_scheduler.step()` before
    `optimizer.step()`" in two expected situations in this codebase:

    1. Step-based schedulers (notably OneCycleLR) perform an internal initial
       step during construction, before any optimizer step has happened.
    2. With AMP, ``GradScaler.step()`` may SKIP ``optimizer.step()`` when it
       detects inf/nan gradients (common on the very first update), while the
       per-batch scheduler still advances.

    PyTorch's own optim docs flag this warning as expected/ignorable for these
    patterns, and the schedule advancing one step on a skipped update is
    negligible for the long cycles we use. We suppress exactly this message —
    and only this message — so training logs stay clean without hiding real
    problems. (Do not use for ReduceLROnPlateau, whose step() needs metrics.)
    """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            category=UserWarning,
            message='.*Detected call of `lr_scheduler\\.step\\(\\)` before `optimizer\\.step\\(\\)`.*',
        )
        scheduler.step(epoch)


def build_scheduler_from_config(config, optimizer, steps_per_epoch, last_epoch=-1):
    """
    Builds an LR scheduler instance based on the provided configuration.

    Args:
        config (EasyDict): Configuration object.
        optimizer (torch.optim.Optimizer): The optimizer for which to build the scheduler.
        last_epoch (int): The epoch to resume training from. Defaults to -1 (start from epoch 0).

    Returns:
        torch.optim.lr_scheduler._LRScheduler: An instance of the specified LR scheduler, or None if no scheduler is specified.
    """
    scheduler_config = config.train.get('scheduler', None)
    if scheduler_config is None:
        return None

    scheduler_name = scheduler_config.name
    # name: "None" (or null) disables scheduling — this is the documented way
    # to turn the scheduler off from a config.
    if scheduler_name is None or str(scheduler_name).strip().lower() in ("", "none"):
        return None
    # ---------------------------------------------------------------------
    # Total number of OPTIMIZER STEPS the schedule must span.
    #
    # This used to be `config.train.epochs * steps_per_epoch` and was then
    # multiplied by `steps_per_epoch` AGAIN inside the CLIP branch, i.e. the
    # schedule was stretched by a factor of `steps_per_epoch` (the LR barely
    # moved over the whole run). `resolve_total_steps()` now owns that
    # arithmetic:
    #   train.scheduler.total_steps  -> explicit override (absolute step count)
    #   otherwise                    -> train.epochs * steps_per_epoch
    # ---------------------------------------------------------------------
    steps_per_epoch = int(steps_per_epoch)
    total_epochs = int(config.train.epochs)
    total_steps, total_steps_source = resolve_total_steps(
        config, scheduler_config, steps_per_epoch, verbose=False
    )
    print(
        f"[scheduler] {scheduler_name}: total_steps={total_steps} [{total_steps_source}], "
        f"steps_per_epoch={steps_per_epoch}, epochs={total_epochs}"
    )

    if scheduler_name == "CosineAnnealingLR":
        T_max = scheduler_config.T_max
        if int(T_max) <= 0:
            raise ValueError(f"CosineAnnealingLR requires T_max > 0, got {T_max}")
        eta_min = scheduler_config.get('eta_min', 0)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)
    
    elif scheduler_name == "CLIPCosineWarmupScheduler":
        warmup_steps = scheduler_config.warmup_steps
        min_lr = scheduler_config.get('min_lr', 5e-6)
        return CLIPCosineWarmupScheduler(optimizer, warmup_steps=warmup_steps, 
        total_steps=total_steps, min_lr=min_lr, last_epoch=last_epoch)
    
    elif scheduler_name == "ReduceLROnPlateau":
        mode = scheduler_config.get('mode', 'min')
        factor = scheduler_config.get('factor', 0.1)
        patience = scheduler_config.get('patience', 10)
        return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    elif scheduler_name == "StepLR":
        step_size = scheduler_config.step_size
        gamma = scheduler_config.get('gamma', 0.1)
        if int(step_size) <= 0:
            raise ValueError(f"StepLR requires step_size > 0, got {step_size}")
        return StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)
    elif scheduler_name == "OneCycleLR":
        max_lr = scheduler_config.max_lr
        anneal_strategy = scheduler_config.get('anneal_strategy', 'cos')
        pct_start = scheduler_config.get('pct_start', 0.1)
        div_factor = scheduler_config.get('div_factor', 25)
        final_div_factor = scheduler_config.get('final_div_factor', 10000)
        # OneCycleLR is stepped once per optimizer step, so its cycle length is
        # specified directly in steps. `train.scheduler.epochs` stays supported
        # as an explicit override of the cycle length (= epochs * steps_per_epoch).
        oc_epochs = scheduler_config.get('epochs', None)
        oc_total_steps = total_steps
        if not _is_unset(oc_epochs) and int(oc_epochs) != total_epochs:
            # Same unit as total_steps: OPTIMISER steps, not micro-batches, so the
            # cycle still ends at the last update when gradient accumulation is on.
            oc_grad_accum = _as_positive_int(config.train.get('gradient_accumulation_steps', 1)) or 1
            oc_opt_steps_per_epoch = max(1, math.ceil(steps_per_epoch / oc_grad_accum))
            oc_total_steps = int(oc_epochs) * oc_opt_steps_per_epoch
            print(
                f"[scheduler] NOTE: train.scheduler.epochs={int(oc_epochs)} overrides train.epochs="
                f"{total_epochs} -> OneCycleLR cycle spans {oc_total_steps} optimiser steps"
            )
        if oc_total_steps <= 0:
            raise ValueError(f"OneCycleLR requires a positive cycle length, got {oc_total_steps}")
        return OneCycleLR(optimizer, max_lr=max_lr, total_steps=oc_total_steps,
                          anneal_strategy=anneal_strategy, pct_start=pct_start,
                          div_factor=div_factor, final_div_factor=final_div_factor)
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        T_0 = scheduler_config.T_0
        T_mult = scheduler_config.get('T_mult', 1)
        eta_min = scheduler_config.get('eta_min', 0)
        return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch)
    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")


class CLIPCosineWarmupScheduler(_LRScheduler):
    """
    Exact replica of OpenAI CLIP's LR schedule (Appendix C.1):
    - linear warmup from 0 to initial_lr over `warmup_steps`
    - cosine decay from initial_lr to `min_lr` over the remaining steps

    STEPPING: this is a *step-based* scheduler — ``step()`` must be called
    exactly once per OPTIMIZER step (once per gradient-accumulation window),
    never per epoch and never before the training loop.  Use
    ``is_step_based_scheduler()`` in the trainers to decide where to step.

    ``total_steps`` is the number of optimizer steps in the whole run, as
    resolved by :func:`resolve_total_steps` (``train.epochs * steps_per_epoch``
    by default, or ``train.scheduler.total_steps`` when set explicitly).
    """
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 warmup_steps: int, total_steps: int, 
                 min_lr: float = 5e-6, last_epoch: int = -1):
        total_steps = int(total_steps)
        if total_steps <= 0:
            raise ValueError(
                f"CLIPCosineWarmupScheduler requires total_steps > 0, got {total_steps}. "
                "total_steps must be the number of optimizer steps in the run "
                "(train.epochs * steps_per_epoch, or train.scheduler.total_steps)."
            )
        self.total_steps = total_steps

        requested_warmup = _as_positive_int(warmup_steps)
        # Keep at least one decay step so `decay_steps` can never be <= 0 and the
        # LR always reaches `min_lr` at the end of the run.
        self.warmup_steps = min(requested_warmup, self.total_steps - 1)
        if self.warmup_steps != requested_warmup:
            print(
                f"[scheduler] CLIPCosineWarmupScheduler: warmup_steps={requested_warmup} "
                f">= total_steps={self.total_steps}; clamped to {self.warmup_steps} "
                f"to keep {self.total_steps - self.warmup_steps} decay step(s)."
            )
        self.decay_steps = max(self.total_steps - self.warmup_steps, 1)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < 0:
            return [0.0 for _ in self.base_lrs]

        if self.warmup_steps > 0 and step < self.warmup_steps:
            # Linear warmup: 0 → base_lr over `warmup_steps` steps
            return [base_lr * (step + 1) / self.warmup_steps for base_lr in self.base_lrs]

        # Cosine decay: base_lr → min_lr, reaching exactly min_lr at total_steps
        progress = min((step - self.warmup_steps) / self.decay_steps, 1.0)
        cos_val = math.cos(math.pi * progress)
        return [self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + cos_val) for base_lr in self.base_lrs]