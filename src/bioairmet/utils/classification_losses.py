'''
@file    :   classification_losses.py
@create date : 2025-05-18 13:05:36
@modify date 2026-02-19 10:13:46
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    This file defines the classification loss functions used in the BioAirMet project.
    It includes a wrapper for CrossEntropyLoss and a custom implementation of Focal Loss (FocalLoss_gt_Corrected), in the standard RetinaNet form: alpha_t * (1 - p_t)^gamma * CE.
    
    The factory function (build_classification_loss_from_config) instantiates the loss from config. For FocalLoss, the per-class alpha weights are either taken from config.train.loss.alpha (explicit list/scalar) or derived from the training set's class imbalance by resolve_focal_alpha() (inverse frequency, sum-normalised) when alpha is "auto" or absent.
    ]
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class CrossEntropyLoss(nn.Module):
    """
    Wrapper for PyTorch's CrossEntropyLoss.
    """
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, size_average=size_average, ignore_index=ignore_index,
            reduce=reduce, reduction=reduction, label_smoothing=label_smoothing
        )

    def forward(self, inputs, targets):
        return self.criterion(inputs, targets)

class FocalLoss_gt_Corrected(nn.Module):
    """
    Compute the focal loss.

    Attributes:
        gamma (float): Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
        alpha (torch.Tensor, optional): Weighting factor for class imbalance. 
                                         Can be a scalar or a 1D tensor of size num_classes.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    """

    def __init__(self, gamma: float = 2.0, alpha: Union[list, tuple, float, torch.Tensor, None] = None, reduction: str = "mean"):
        """
        Initialize the FocalLoss.

        Args:
            gamma (float): Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
            alpha (list, tuple, float, torch.Tensor, optional): Weighting factor for class imbalance. 
                If a list/tuple, it's converted to a Tensor. If None, no alpha weighting is applied.
                If a float, it's a scalar weight. For your case, provide the list of 35 alpha values.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss_gt_Corrected, self).__init__()
        self.gamma = gamma
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                # Convert list/tuple of alphas to a tensor
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, (float, int)):
                # If alpha is a scalar, store it as is (though for your specific list, this branch isn't taken)
                self.alpha = alpha 
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                raise TypeError("Alpha must be a list, tuple, float, int, or torch.Tensor")
        else:
            self.alpha = None # No alpha weighting

        self.reduction = reduction
        if self.reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the focal loss.

        Args:
            inputs (torch.Tensor): The model output (logits). Shape: (batch_size, num_classes)
            targets (torch.Tensor): The ground truth labels (class indices). Shape: (batch_size,)

        Returns:
            torch.Tensor: The focal loss.
        """
        # Calculate Cross Entropy loss for each sample without reduction
        # inputs are logits, targets are class indices
        ce_loss = F.cross_entropy(inputs, targets, reduction="none") # Shape: (batch_size,)
        
        # Calculate pt (probability of the true class)
        pt = torch.exp(-ce_loss) # Shape: (batch_size,)
        
        # Calculate the modulating factor
        modulating_factor = (1 - pt) ** self.gamma # Shape: (batch_size,)
        
        # Base focal loss term (without alpha)
        focal_loss_sample = modulating_factor * ce_loss # Shape: (batch_size,)

        # Apply alpha weighting if alpha is provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor) and self.alpha.ndim == 1: # Per-class alpha weights
                # Ensure alpha tensor is on the same device as inputs
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                
                # Gather the alpha values for each target class
                # self.alpha has shape (num_classes,); targets has shape (batch_size,)
                alpha_t = self.alpha[targets] # Shape: (batch_size,)
                focal_loss_sample = alpha_t * focal_loss_sample
            elif isinstance(self.alpha, (float, int)): # Scalar alpha
                 focal_loss_sample = self.alpha * focal_loss_sample
            # If self.alpha is a tensor but not 1D, it's an unexpected configuration for per-class weights.

        # Apply reduction
        if self.reduction == "mean":
            focal_loss = focal_loss_sample.mean()
        elif self.reduction == "sum":
            focal_loss = focal_loss_sample.sum()
        elif self.reduction == "none":
            focal_loss = focal_loss_sample
        else: # Should be caught by __init__
            focal_loss = focal_loss_sample # Default to none or raise error

        return focal_loss
    
# Factory function to build classification loss from config
def build_classification_loss_from_config(config):
    """
    Builds a classification loss function instance based on the provided configuration.

    Args:
        config (EasyDict): Configuration object.

    Returns:
        torch.nn.Module: An instance of the specified classification loss function.
    """
    loss_name = config.train.loss.get('name', 'CrossEntropyLoss') # Default to CrossEntropyLoss
    label_smoothing = config.train.loss.get('label_smoothing', 0.0)
    if loss_name == "CrossEntropyLoss":
        # Parameters for CrossEntropyLoss can be added here if needed
        return CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_name == "FocalLoss_gt_Corrected":
        # Per-class alpha weights: an explicit list/scalar in
        # config.train.loss.alpha is used as-is (manual control). "auto" (or an
        # absent key) means resolve_focal_alpha() derived them from the
        # training set's class imbalance and wrote them into the config — the
        # classification worker runs it before the loss is built. If they are
        # still missing here (e.g. standalone validation of an old bundle that
        # has no training-set counts), fall back to uniform weights with a note.
        alpha = config.train.loss.get('alpha', None)
        if alpha is None or (isinstance(alpha, str) and alpha.strip().lower() == 'auto'):
            print("FocalLoss: no per-class alpha weights in config — using uniform "
                  "class weights. (Training with alpha: 'auto' derives them from "
                  "the training set's class imbalance.)")
            alpha = None
        # Guard: a per-class alpha list must line up with the model head width,
        # otherwise FocalLoss will raise an obscure broadcasting error at train time.
        try:
            _num_classes = int(config.architecture_setup.classification_model.num_classes)
        except Exception:
            _num_classes = None
        if (isinstance(alpha, (list, tuple)) and _num_classes is not None
                and len(alpha) != _num_classes):
            raise ValueError(
                f"FocalLoss alpha has {len(alpha)} values but "
                f"architecture_setup.classification_model.num_classes={_num_classes}. "
                f"Use a model head with {len(alpha)} classes (set num_classes="
                f"{len(alpha)}) or provide a {len(alpha)}-element alpha list to match.")
        gamma = config.train.loss.get('gamma', 1.0)
        return FocalLoss_gt_Corrected(alpha=alpha, gamma=gamma)
    else:
        raise ValueError(
            f"Unknown classification loss function name: {loss_name}. "
            f"Supported: CrossEntropyLoss, FocalLoss_gt_Corrected.")


def resolve_focal_alpha(config, train_dataset, logger=None):
    """
    Derive per-class focal-loss alpha weights from the training set's class
    imbalance and store them in ``config.train.loss.alpha``.

    Called by the classification worker right after the training dataset is
    built (and before the loss is constructed). No-op unless
    ``train.loss.name == 'FocalLoss_gt_Corrected'`` and ``alpha`` is absent or
    ``'auto'`` — an explicit ``alpha`` list/scalar in the config always wins.

    Weights are inverse class frequency, normalised to sum to 1
    (alpha_c = (1/count_c) / sum_j (1/count_j)) — the same convention the
    focal loss historically shipped with. Classes absent from the training set
    get alpha 0. The result is written into the config object, so the
    experiment's config.yaml (copied to the log dir by the trainer) persists
    it and standalone validation of the experiment uses the identical weights.

    Args:
        config (EasyDict): Configuration object (mutated: train.loss.alpha).
        train_dataset: Training dataset; must expose ``class_counts()``.
        logger: Optional logger with .info()/.warning(); records the result.

    Returns:
        list[float] | None: The alpha weights written to the config, or None
        when no computation was needed/possible (explicit alpha, a different
        loss, or the dataset provides no class counts).
    """
    loss_conf = config.train.get('loss', None)
    if loss_conf is None or loss_conf.get('name', 'CrossEntropyLoss') != 'FocalLoss_gt_Corrected':
        return None

    alpha = loss_conf.get('alpha', None)
    if alpha is not None and not (isinstance(alpha, str) and alpha.strip().lower() == 'auto'):
        return None  # explicit manual alpha: keep it

    try:
        num_classes = int(config.architecture_setup.classification_model.num_classes)
    except Exception:
        num_classes = None

    counts = None
    if train_dataset is not None and hasattr(train_dataset, 'class_counts'):
        try:
            counts = train_dataset.class_counts(num_classes)
        except Exception as e:
            print(f"FocalLoss: could not read class counts from the training set ({e}) — "
                  "using uniform class weights.")

    if counts is None:
        print("FocalLoss: no class counts available — using uniform class weights "
              "(set alpha: 'auto' with a labelled training set to derive them).")
        return None

    counts = np.asarray(counts, dtype=np.float64)
    if num_classes is None:
        num_classes = int(counts.max()) + 1
    if len(counts) < num_classes:
        counts = np.pad(counts, (0, num_classes - len(counts)))
    else:
        counts = counts[:num_classes]

    absent = [int(c) for c in range(num_classes) if counts[c] <= 0]
    if absent:
        msg = (f"FocalLoss: class(es) {absent} absent from the training set — "
               "they get alpha 0.")
        print(msg)
        if logger is not None:
            logger.warning(msg)

    inv = np.zeros_like(counts)
    mask = counts > 0
    inv[mask] = 1.0 / counts[mask]
    total = inv.sum()
    if total <= 0:
        print("FocalLoss: no class has training samples — using uniform class weights.")
        return None
    alpha_list = (inv / total).tolist()

    config.train.loss.alpha = alpha_list
    if logger is not None:
        logger.info(
            "FocalLoss: computed per-class alpha from training-set class counts "
            "(inverse frequency, sum-normalised): "
            + ", ".join(f"c{c}={a:.4f}" for c, a in enumerate(alpha_list))
        )
    return alpha_list


def check_label_coverage(config, train_dataset, val_dataset=None, logger=None):
    """
    Cross-check the dataset class labels against the configured ``num_classes``
    and warn about any mismatch.

    The classification datasets exclude up front (see ``Stage2Dataset``)
    samples whose label is outside ``[0, num_classes)`` — so a "garbage"
    class 36 in the data does not crash a run configured with
    ``num_classes: 35`` but is silently skipped. This function makes that
    visible, and also reports the opposite case: configured classes that have
    no samples at all (their classifier heads would remain untrained and get
    focal alpha 0).

    Called by the classification worker right after the datasets are built.
    Warnings go through ``logger.warning`` when a logger is given, and are
    always printed (the worker passes a rank-0-only logger in multi-GPU mode).

    In addition, an INFO summary is always emitted for the training set
    (``logger.info`` when the logger supports it, else printed) reporting how
    many of the configured ``num_classes`` actually have samples in the
    training data and which classes are empty.

    Args:
        config (EasyDict): Configuration object (reads
            ``architecture_setup.classification_model.num_classes``).
        train_dataset: Training dataset; must expose ``class_counts()``.
        val_dataset: Optional validation dataset (same requirement).
        logger: Optional logger with ``.warning()``.

    Returns:
        dict: Per-dataset findings, keyed by "training"/"validation":
            ``{"excluded": {label: sample_count, ...},
               "uncovered": [label, ...]}``. ``excluded`` lists labels that
            exist in the file but were dropped by the category-map /
            num_classes pre-filter (both datasets); ``uncovered`` lists
            configured classes with zero training samples (training set only).
            Datasets with no readable HDF5 file are skipped.
    """
    try:
        num_classes = int(config.architecture_setup.classification_model.num_classes)
    except Exception:
        num_classes = None

    def _warn(msg):
        print(msg)
        if logger is not None:
            logger.warning(msg)

    def _info(msg):
        # Info-level summary: goes to the logger when it supports it (the
        # rank-0 model_logger in the worker), to the console when no logger
        # is given. Loggers without an .info method (unit-test stubs) are
        # skipped silently.
        info = getattr(logger, "info", None) if logger is not None else None
        if callable(info):
            info(msg)
        elif logger is None:
            print(msg)

    findings = {}
    for name, dataset in (("training", train_dataset), ("validation", val_dataset)):
        if dataset is None or not hasattr(dataset, 'class_counts'):
            continue
        if not getattr(dataset, 'hdf5_path', None):
            continue  # no HDF5 file behind this dataset — nothing to check

        try:
            raw_counts = dataset.class_counts(raw=True)
            eff_counts = dataset.class_counts()
        except Exception as e:
            _warn(
                f"Class-coverage: could not read label counts from the "
                f"{name} dataset ({e}) — skipping the coverage check for it."
            )
            continue

        raw_counts = np.asarray(raw_counts, dtype=np.int64)
        eff_counts = np.asarray(eff_counts, dtype=np.int64)
        n = max(len(raw_counts), len(eff_counts))

        excluded = {}
        for c in range(n):
            raw_c = int(raw_counts[c]) if c < len(raw_counts) else 0
            eff_c = int(eff_counts[c]) if c < len(eff_counts) else 0
            if raw_c > 0 and eff_c == 0:
                excluded[c] = raw_c

        uncovered = []
        if name == "training" and num_classes is not None:
            # classes already reported above as *excluded* (their samples
            # were filtered out) are not listed again as uncovered
            uncovered = [
                c for c in range(num_classes)
                if (c >= len(eff_counts) or eff_counts[c] <= 0)
                and c not in excluded
            ]

        # Always report (info) how many configured classes actually have
        # samples in the training data — i.e. which classifier heads will be
        # trained at all (empty ones stay untrained and get focal alpha 0).
        if name == "training" and num_classes is not None:
            present = [c for c in range(num_classes)
                       if c < len(eff_counts) and int(eff_counts[c]) > 0]
            total_samples = int(eff_counts.sum())
            missing = [c for c in range(num_classes) if c not in present]
            summary = (
                f"Class-coverage: training data contains "
                f"{len(present)}/{num_classes} configured classes "
                f"({total_samples} samples)"
                + (f"; classes with NO samples: {missing}" if missing
                   else "; every configured class has samples")
            )
            _info(summary)

        if not excluded and not uncovered:
            continue

        lines = []
        if excluded:
            details = []
            for c in sorted(excluded):
                if num_classes is not None and c >= num_classes:
                    reason = (f"label is outside the configured range "
                              f"(label >= num_classes={num_classes})")
                else:
                    reason = "not present in the category map"
                details.append(f"class {c}: {excluded[c]} sample(s) — {reason}")
            lines.append(
                "Class-coverage: the following classes are excluded from the "
                f"{name} dataset and will be skipped during training: "
                + "; ".join(details)
                + ". To include them, raise "
                "architecture_setup.classification_model.num_classes and/or "
                "adjust the category map."
            )
        if uncovered:
            lines.append(
                f"Class-coverage: num_classes={num_classes} but the training "
                f"set has no samples for class(es) {uncovered} — their "
                "classifier heads will remain untrained (focal alpha = 0). "
                "Reduce num_classes or provide samples for them."
            )
        for line in lines:
            _warn(line)
        findings[name] = {"excluded": excluded, "uncovered": uncovered}

    return findings

