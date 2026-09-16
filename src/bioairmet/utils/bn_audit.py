'''
@file    :   bn_audit.py
@author  :   Mansoor Nabawi
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    Observability for the freeze / BatchNorm policy.

    ``repr(model)`` cannot show whether a module is in ``train()`` or ``eval()``
    mode, whether its BatchNorm layers update their running statistics, or which
    parameters have ``requires_grad``.  That is exactly the state the
    ``train.fine_tuning.*`` and ``*_batchnorm_*`` switches control, so a config
    change could silently have no effect while the log looked unchanged.

    This module renders that state as tables and checks it against the config:

    ``bn_state_summary(model)``
        One row per top-level submodule: number of BatchNorm layers, how many are
        in train/eval mode, how many track running stats, how many actually
        UPDATE them, trainable affine parameters, trainable/total parameters.
    ``freeze_state_summary(model)``
        One row per top-level submodule: trainable vs total parameters, share,
        whether every parameter is frozen, and train/eval module counts.
    ``bn_freeze_warnings(model, config, optimizer, phase)``
        ``[BN-DIVERGENCE]`` / ``[FREEZE-DIVERGENCE]`` messages for requests that
        had no effect (e.g. update_batchnorm_stats_* on an encoder whose BN never
        updates, trainable parameters missing from the optimizer, or
        track_running_stats=False anywhere).  ``phase='train'`` means the policy
        is expected to be in force (call it from ``train_epoch``); while the model
        is in ``eval()`` a ``update_batchnorm_stats_*`` request cannot take effect
        yet and is therefore NOT reported as a divergence here.
    ``bn_pending_notes(model, config)``
        The informational counterpart: ``[BN-policy]`` lines for BN-stat requests
        that are pending because the model is still in ``eval()`` (the state the
        startup audit runs in).
    ``bn_state_line(model)``
        One compact line, suitable for logging at the start of every epoch after
        ``model.train()`` has clobbered the per-encoder modes.
    ``log_bn_and_freeze_state(model, logger, ...)``
        Logs the tables plus the warnings.

    Convention enforced here: ``track_running_stats`` stays ``True`` EVERYWHERE.
    A frozen BatchNorm is achieved with ``eval()`` (which blocks buffer updates)
    while still tracking - never with ``track_running_stats=False``, because a
    BatchNorm with tracking disabled normalises with BATCH statistics in train
    mode and with the (possibly never-updated) buffers in eval mode, i.e. it
    silently stops normalising at all on a freshly built encoder.
    ]
'''

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.SyncBatchNorm, nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model of a DDP/DataParallel wrapper."""
    return getattr(model, "module", model)


def _fmt(n: int) -> str:
    return f"{n:,}"


def _bn_layers(module: nn.Module) -> List[nn.Module]:
    return [m for m in module.modules() if isinstance(m, BN_TYPES)]


def _bn_stats(module: nn.Module) -> Dict[str, int]:
    """Aggregate the BatchNorm state of one sub-module."""
    bns = _bn_layers(module)
    stats = {
        "n_bn": len(bns),
        "n_train": sum(1 for m in bns if m.training),
        "n_eval": sum(1 for m in bns if not m.training),
        "n_track": sum(1 for m in bns if getattr(m, "track_running_stats", True)),
        # a BN layer updates running_mean/running_var only while it is in train
        # mode AND tracks them (PyTorch rule in F.batch_norm)
        "n_updating": sum(1 for m in bns
                          if m.training and getattr(m, "track_running_stats", True)),
        "n_no_buffer": sum(1 for m in bns
                           if not getattr(m, "track_running_stats", True)
                           and getattr(m, "running_mean", None) is not None
                           and float(m.running_var.abs().sum()) == float(m.num_features)),
        "affine_total": 0,
        "affine_trainable": 0,
    }
    for m in bns:
        for pname in ("weight", "bias"):
            p = getattr(m, pname, None)
            if isinstance(p, nn.Parameter):
                stats["affine_total"] += 1
                if p.requires_grad:
                    stats["affine_trainable"] += 1
    return stats


def _groups(model: nn.Module) -> Dict[str, nn.Module]:
    """Top-level submodules of ``model`` (the rows of the audit tables)."""
    model = unwrap_model(model)
    groups = {name: child for name, child in model.named_children()}
    # parameters that belong to the root itself (no owning child)
    child_param_ids = set()
    for child in groups.values():
        child_param_ids.update(id(p) for p in child.parameters())
    root_params = [p for p in model.parameters(recurse=False)]
    if root_params:
        groups["(root params)"] = model
    return groups


def bn_state_summary(model: nn.Module) -> List[str]:
    """BatchNorm state table, one row per top-level submodule."""
    model = unwrap_model(model)
    header = (f"{'module':<26}{'BN':>4}  {'mode(tr/ev)':<12}{'track':<7}"
              f"{'updating':<10}{'affine(tr/tot)':<16}{'tensors(tr/tot)'}")
    lines = [header, "-" * len(header)]
    if not _bn_layers(model):
        lines.append("(no BatchNorm layers in this model)")
        return lines

    for name, child in _groups(model).items():
        s = _bn_stats(child)
        if s["n_bn"] == 0:
            continue
        params = list(child.parameters())
        trainable = sum(1 for p in params if p.requires_grad)
        lines.append(
            f"{name:<26}{s['n_bn']:>4}  "
            f"{s['n_train']}/{s['n_eval']:<10}"
            f"{s['n_track']}/{s['n_bn']:<5}"
            f"{s['n_updating']}/{s['n_bn']:<8}"
            f"{s['affine_trainable']}/{s['affine_total']:<14}"
            f"{_fmt(trainable)}/{_fmt(len(params))}"
        )
    lines.append(
        "mode: train=normalises with batch stats & updates them | eval=uses stored running stats; "
        "'updating' = BN in train mode AND tracking (the only case buffers change)")
    return lines


def freeze_state_summary(model: nn.Module) -> List[str]:
    """Trainable-parameter table, one row per top-level submodule."""
    model = unwrap_model(model)
    header = (f"{'module':<26}{'trainable':>12}{'total':>12}{'%train':>8}  "
              f"{'grad':<14}{'modules(tr/ev)'}")
    lines = [header, "-" * len(header)]
    tot_train = tot_all = 0
    for name, child in _groups(model).items():
        params = list(child.parameters())
        if not params:
            continue
        trainable = sum(p.numel() for p in params if p.requires_grad)
        total = sum(p.numel() for p in params)
        tot_train += trainable
        tot_all += total
        if trainable == 0:
            grad = "all frozen"
        elif trainable == total:
            grad = "all trainable"
        else:
            grad = "mixed"
        mods = list(child.modules())
        m_train = sum(1 for m in mods if m.training)
        lines.append(
            f"{name:<26}{_fmt(trainable):>12}{_fmt(total):>12}"
            f"{100.0 * trainable / max(total, 1):>7.1f}%  {grad:<14}"
            f"{m_train}/{len(mods) - m_train}"
        )
    lines.append(
        f"{'TOTAL':<26}{_fmt(tot_train):>12}{_fmt(tot_all):>12}"
        f"{100.0 * tot_train / max(tot_all, 1):>7.1f}%")
    return lines


def bn_state_line(model: nn.Module) -> str:
    """One-line BN summary per top-level submodule (for the per-epoch log)."""
    model = unwrap_model(model)
    parts = []
    for name, child in _groups(model).items():
        s = _bn_stats(child)
        if s["n_bn"] == 0:
            continue
        parts.append(
            f"{name}[BN {s['n_bn']}: eval {s['n_eval']}, train {s['n_train']}; "
            f"updating {s['n_updating']}; affine-trainable {s['affine_trainable']}/{s['affine_total']}]")
    if not parts:
        return "no BatchNorm layers"
    return "; ".join(parts)


def _optimizer_param_ids(optimizer) -> set:
    if optimizer is None:
        return set()
    ids = set()
    for group in optimizer.param_groups:
        for p in group.get("params", []):
            ids.add(id(p))
    return ids


def bn_freeze_warnings(model: nn.Module, config: Any = None,
                       optimizer=None, phase: Optional[str] = None) -> List[str]:
    """Divergence warnings: a requested freeze/BN behaviour that is not in effect.

    Args:
        model: the model (DDP-wrapped or not).
        config: the full experiment config (optional); ``train.fine_tuning`` is
            read for the classification-stage switches
            (``unfreeze_image_encoder``, ``unfreeze_last_img_layers``,
            ``update_batchnorm_stats_*``, ``train_batchnorm_affine_*``,
            ``unfreeze_batchnorm_*``), and ``model.freeze_bn`` / the SSL stage
            flags for the SSL-side ones.
        optimizer: the optimizer, to check that every trainable parameter is
            actually optimized.
        phase: ``'train'`` when the freeze / BatchNorm policy is IN FORCE (call it
            from ``train_epoch``, after ``model.train()`` +
            ``enforce_frozen_modes()``), ``'eval'`` when it is not applied yet (the
            startup audit, a freshly built model, or validation), ``None`` to
            auto-detect from ``model.training``.  Auto-detection is only a
            heuristic: a freshly built HoloClassifierV2 has ``training=True`` on the
            root while its encoders were put in ``eval()`` by the freeze policy, so
            the trainers pass the phase explicitly.  Only in the ``'train'`` phase is
            a ``update_batchnorm_stats_*`` request that is not in effect a real
            divergence; otherwise it is reported by :func:`bn_pending_notes`.  Checks
            that do not depend on the train/eval mode (``requires_grad``
            freezes/unfreezes, BN affine parameters, optimizer coverage,
            ``track_running_stats``) run in both phases.
    """
    # A DDP/DataParallel wrapper hides the real model behind '.module'; remember it
    # before unwrapping, because BatchNorm buffers are NOT synchronised across ranks.
    ddp_wrapped = hasattr(model, "module")
    model = unwrap_model(model)
    warnings: List[str] = []
    # 'train' phase = the training loop is running, so a BN-stats request that is
    # not in effect is a real bug. In 'eval' phase no BatchNorm can update its
    # buffers at all, so the same observation is only a PENDING request
    # (bn_pending_notes), not a divergence.
    if phase is None:
        phase = "train" if model.training else "eval"
    stats_checks_active = (phase == "train")
    ft: Dict[str, Any] = {}
    if config is not None:
        try:
            ft = dict(config.train.get("fine_tuning", {}) or {})
        except AttributeError:
            ft = {}

    groups = _groups(model)

    # ---- global invariant: track_running_stats must stay True --------------
    no_track = [name for name, mod in model.named_modules() if isinstance(mod, BN_TYPES)
                and not getattr(mod, "track_running_stats", True)]
    if no_track:
        warnings.append(
            f"[BN-DIVERGENCE] {len(no_track)} BatchNorm layer(s) have "
            f"track_running_stats=False (e.g. '{no_track[0]}'). Frozen BatchNorm is "
            "expressed with eval() + track_running_stats=True in this project: with "
            "tracking disabled a layer normalises with BATCH statistics in train mode "
            "and with never-updated buffers in eval mode (i.e. no normalisation at all). "
            "Call freeze_batchnorm() from bioairmet.models.backbones instead.")

    # ---- per-encoder checks (classification stage) -------------------------
    policy = getattr(model, "_ft_policy", None) or {}
    for which, enc_names in (("img", ("img_encoder", "image_encoder")),
                             ("fl", ("fl_encoder", "fluorescence_encoder"))):
        enc_name = next((n for n in enc_names if n in groups), None)
        enc = groups.get(enc_name) if enc_name else None
        if enc is None:
            continue
        s = _bn_stats(enc)
        params = list(enc.parameters())
        trainable = sum(1 for p in params if p.requires_grad)

        full_key = "unfreeze_image_encoder" if which == "img" else "unfreeze_fluorescence_encoder"
        last_n_key = "unfreeze_last_img_layers" if which == "img" else "unfreeze_last_fl_layers"
        want_full = bool(ft.get(full_key, False))
        want_last_n = ft.get(last_n_key, False)
        want_last_n = 0 if want_last_n in (None, False) else int(want_last_n)
        want_bn_stats = bool(ft.get(f"update_batchnorm_stats_{which}_encoder", False)) or \
            bool(ft.get(f"unfreeze_batchnorm_{which}_encoder", False))
        want_bn_affine = bool(ft.get(f"train_batchnorm_affine_{which}_encoder", False)) or \
            bool(ft.get(f"unfreeze_batchnorm_{which}_encoder", False))

        if want_full and trainable == 0:
            warnings.append(
                f"[BN-DIVERGENCE] {full_key}: true but {enc_name} has 0 trainable "
                "parameters - the encoder is still frozen (check that the flag is set in "
                "train.fine_tuning and that the model was built after it).")
        if (not want_full) and want_last_n == 0 and trainable > 0 and policy.get(which) == "frozen":
            warnings.append(
                f"[BN-DIVERGENCE] {enc_name} is policy 'frozen' but has {trainable} "
                "trainable parameter(s).")
        if want_last_n > 0 and trainable == 0:
            warnings.append(
                f"[BN-DIVERGENCE] {last_n_key}: {want_last_n} requested but {enc_name} has "
                "0 trainable parameters - select_last_n_layer_modules() matched no unit "
                "(the backbone may expose fewer layer units than requested).")
        if want_bn_stats and s["n_bn"] > 0 and s["n_updating"] == 0 and stats_checks_active:
            warnings.append(
                f"[BN-DIVERGENCE] update_batchnorm_stats_{which}_encoder: true but none of "
                f"the {s['n_bn']} BatchNorm layers of {enc_name} update their running stats "
                f"while the model is in TRAIN mode (all {s['n_eval']} are in eval mode). The "
                "policy was not applied: check that the model exposes enforce_frozen_modes() "
                "and that the trainer calls it after model.train(), and that "
                "track_running_stats is True.")
        if want_bn_stats and s["n_updating"] > 0 and ddp_wrapped:
            warnings.append(
                f"[BN-DDP] update_batchnorm_stats_{which}_encoder: true on a DDP-wrapped "
                f"model: {s['n_updating']} of the {s['n_bn']} BatchNorm layers of {enc_name} "
                "update their running statistics, but DDP does NOT synchronise buffers across "
                "ranks - each rank estimates them from its own shard and only rank 0's are "
                "saved. Convert with nn.SyncBatchNorm.convert_sync_batchnorm(), or update the "
                "statistics in a single process, if the estimate matters for this run.")
        if want_bn_affine and s["affine_trainable"] == 0 and s["affine_total"] > 0:
            warnings.append(
                f"[BN-DIVERGENCE] train_batchnorm_affine_{which}_encoder: true but 0 of the "
                f"{s['affine_total']} BatchNorm affine parameters of {enc_name} have "
                "requires_grad=True (they must be trainable AND in the optimizer).")

    # ---- optimizer <-> requires_grad consistency --------------------------
    if optimizer is not None:
        opt_ids = _optimizer_param_ids(optimizer)
        trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
        missing = trainable_ids - opt_ids
        extra = opt_ids - trainable_ids
        if missing:
            names = [n for n, p in model.named_parameters() if p.requires_grad and id(p) in missing]
            warnings.append(
                f"[FREEZE-DIVERGENCE] {len(missing)} trainable parameter(s) are NOT in any "
                f"optimizer param group and will never be updated (first few: "
                f"{names[:5]}).")
        if extra:
            names = [n for n, p in model.named_parameters() if not p.requires_grad and id(p) in extra]
            warnings.append(
                f"[FREEZE-DIVERGENCE] {len(extra)} parameter(s) are in the optimizer while "
                f"requires_grad=False (frozen after the optimizer was built?) (first few: "
                f"{names[:5]}).")

    return warnings


def bn_pending_notes(model: nn.Module, config: Any = None,
                     phase: Optional[str] = None) -> List[str]:
    """Requests whose BatchNorm effect cannot be observed YET (policy not applied).

    ``update_batchnorm_stats_*`` puts BatchNorm layers in train mode, but a layer
    can only touch its running statistics while the policy is actually in force.
    The startup audit runs BEFORE the first ``train_epoch``: the freeze policy put
    the encoders in ``eval()`` at build time and ``enforce_frozen_modes()`` has not
    run yet, so "0 updating" at that point is the EXPECTED state, not a
    misconfiguration - reporting it as a warning is a false positive.  ``_apply_fine_tuning_policy``
    only stores the intent; ``train_epoch()`` makes it effective with
    ``model.train()`` + ``enforce_frozen_modes()``, and the per-epoch ``[BN-state]``
    line then reports whether the request actually became effective.

    ``phase`` follows :func:`bn_freeze_warnings`: ``'train'`` = the policy is in
    force, ``'eval'`` = not yet applied, ``None`` = auto-detect from
    ``model.training``.  Returns one informational line per encoder with a pending
    request (empty list in the ``'train'`` phase).
    """
    model = unwrap_model(model)
    if phase is None:
        phase = "train" if model.training else "eval"
    if phase == "train":
        return []
    ft: Dict[str, Any] = {}
    if config is not None:
        try:
            ft = dict(config.train.get("fine_tuning", {}) or {})
        except AttributeError:
            ft = {}
    if not ft:
        return []

    notes: List[str] = []
    groups = _groups(model)
    for which, enc_names in (("img", ("img_encoder", "image_encoder")),
                             ("fl", ("fl_encoder", "fluorescence_encoder"))):
        enc_name = next((n for n in enc_names if n in groups), None)
        if enc_name is None:
            continue
        key = f"update_batchnorm_stats_{which}_encoder"
        want = (bool(ft.get(key, False))
                or bool(ft.get(f"unfreeze_batchnorm_{which}_encoder", False)))
        s = _bn_stats(groups[enc_name])
        if want and s["n_bn"] > 0 and s["n_updating"] == 0:
            notes.append(
                f"[BN-policy] {key}: true and PENDING (not a problem) - the BatchNorm policy "
                f"is not in force yet, so none of the {s['n_bn']} BatchNorm layers of "
                f"{enc_name} (all {s['n_eval']} in eval mode) updates its running statistics. "
                "This is the state the startup audit sees; the policy becomes effective when "
                "train_epoch() calls model.train() + enforce_frozen_modes(). Verify it on the "
                f"[BN-state] epoch line: 'updating {s['n_bn']}/{s['n_bn']}' means the stats "
                "really update.")
    return notes


def log_bn_and_freeze_state(model: nn.Module, logger: Any, config: Any = None,
                            optimizer=None, header: str = "BatchNorm / Freeze audit",
                            warn: bool = True,
                            phase: Optional[str] = None) -> List[str]:
    """Log the BatchNorm and freeze tables plus any divergence warnings.

    ``phase`` is forwarded to :func:`bn_freeze_warnings` ('train' = the policy is
    expected to be in force, 'eval' = startup audit, None = auto-detect).  While
    the model is in eval mode, pending ``update_batchnorm_stats_*`` requests are
    logged as informational ``[BN-policy]`` lines instead of warnings.

    Returns the warnings (also printed via ``logger.warning`` when ``warn``).  A
    ``logger`` without ``warning`` (e.g. a plain logger) is handled gracefully.
    """
    msgs: List[str] = []
    block = [f"--- {header} ---"]
    block += ["  " + line for line in freeze_state_summary(model)]
    block += [""]
    block += ["  " + line for line in bn_state_summary(model)]
    text = "\n".join(block)
    if logger is not None:
        logger.info(text)
    else:  # pragma: no cover - convenience for ad-hoc use
        print(text)

    # Pending BN-stat requests (policy not in force yet): informational, NOT warnings.
    for note in bn_pending_notes(model, config=config, phase=phase):
        if logger is not None:
            logger.info(note)
        else:
            print(note)

    if warn:
        msgs = bn_freeze_warnings(model, config=config, optimizer=optimizer, phase=phase)
        for w in msgs:
            if logger is not None:
                log = getattr(logger, "warning", None) or logger.info
                log(w)
            else:
                print(w)
    return msgs
