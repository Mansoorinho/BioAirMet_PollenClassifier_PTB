'''
@file    :   base_trainer_v2.py
@create date : 2026-01-10 14:04:08
@modify date 2026-05-27 16:05:54
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    This file defines the base trainer class for V2 training workflows in the BioAirMet project.
    It provides common infrastructure for both SSL and classification training:
    - Logger setup (file + model + tensorboard)
    - Checkpoint management (save/load with atomic writes)
    - Device placement and DDP orchestration
    - Seed setting for reproducibility
    - Early stopping and Learning Rate scheduler management
    Subclasses must implement the train_epoch() and validate_epoch() methods, which define the core training and validation logic for each epoch.
    The main training loop is implemented in the train() method, which handles epoch iteration, logging, checkpointing, and early stopping.
    This design promotes code reuse and consistency across different training workflows while allowing flexibility for specific model architectures and training strategies.
    ]
'''

import os
import random
# import datetime
from collections import OrderedDict
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from ..utils import (
    setup_logger,
    setup_model_logger,
    TensorboardLogger,
    EarlyStoppingV2,
    log_metrics_to_file,
    step_scheduler,
    is_step_based_scheduler,
    log_bn_and_freeze_state,
    bn_state_line,
    bn_freeze_warnings,
)
from ..models._common import strip_ddp_prefix, ensure_state_dict_metadata


class BaseTrainerV2(ABC):
    """
    Abstract base class for all V2 trainers.
    
    Provides common functionality:
    - Logger setup (file + model + tensorboard)
    - Checkpoint management (save/load with atomic writes)
    - Device placement
    - Seed setting for reproducibility
    - DDP wrapper
    
    Subclasses must implement:
    - train_epoch()
    - validate_epoch()
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        rank: int,
        world_size: int,
        device_id: int,
        skip_seed: bool = False,
    ):
        """
        Initialize base trainer.
        
        Args:
            config: Configuration dict (EasyDict)
            rank: Process rank for DDP
            world_size: Total number of processes
            device_id: GPU device ID
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        # log_dir is optional — when missing it defaults to checkpoint_dir,
        # so configs without it (the shipped templates) work as-is.
        self.log_dir = config.logging.get('log_dir') or config.logging.checkpoint_dir
        self.checkpoint_dir = config.logging.checkpoint_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Setup logging
        log_subdir = os.path.join(self.log_dir, 'logs')
        plot_subdir = os.path.join(self.log_dir, 'plots')
        os.makedirs(log_subdir, exist_ok=True)
        os.makedirs(plot_subdir, exist_ok=True)
        
        self.file_logger = setup_logger(log_subdir, rank)
        self.model_logger = setup_model_logger(log_subdir, rank)
        self.tb_logger = None
        # tensorboard is optional — missing means off (old configs have no
        # such key; the shipped templates default it to False).
        if rank == 0 and config.logging.get('tensorboard', False):
            self.tb_logger = TensorboardLogger(log_subdir)
        
        # Set seed for reproducibility
        # skip_seed=True when the caller (e.g. train_ssl_worker_v2 wrapper) has
        # already set the seed prior to building datasets.  Re-setting it here
        # would shift the RNG state and produce different weight initialization
        # compared to legacy, which sets the seed only once.
        if not skip_seed:
            self._set_seed(config.seed + rank)
        
        # Optional throughput switches (train.cudnn_benchmark / train.tf32 /
        # train.channels_last).  All default to off => behaviour unchanged.
        self.use_channels_last = self._configure_performance()
        
        # Placeholders for model, optimizer, scheduler (set by subclasses)
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')  # Override in subclass if using max metric
        
        # Early stopping
        self.early_stopping: Optional[EarlyStoppingV2] = None
        
        # Metrics tracking for plotting
        self.train_history: Dict[str, list] = {}
        self.val_history: Dict[str, list] = {}
        self._csv_header_written = False
    
    def _set_seed(self, seed: int) -> None:
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if self.rank == 0:
            self.model_logger.info(f"Seed set to {seed}")

    def _configure_performance(self) -> bool:
        """Apply the optional ``train.*`` throughput switches.

        ``train.cudnn_benchmark`` (default ``False``)
            Lets cuDNN auto-tune convolution algorithms.  Only safe/helpful when
            the input shapes stay constant across steps (they do here: fixed
            ``data.image_size`` and fixed batch size except the last partial
            batch).  It changes *which* algorithm runs, so results can differ in
            the last bits compared to a run without it.

        ``train.tf32`` (default ``False``)
            Enables TensorFloat32 for matmul/cudnn on Ampere+.  Faster, at the
            cost of ~10 bits of mantissa on those ops.

        ``train.channels_last`` (default ``False``)
            Stores the model in NHWC and (when supported) runs convolutions in
            NHWC, usually together with AMP.  Returns the flag so the epoch
            loops can convert floating-point inputs with
            :meth:`_to_device`; non-floating tensors are left untouched.

        Returns
        -------
        bool
            ``True`` when channels_last is enabled (the model is converted in
            :meth:`to_device` once the model exists).
        """
        train_cfg = self.config.train
        cudnn_benchmark = bool(train_cfg.get('cudnn_benchmark', False))
        tf32 = bool(train_cfg.get('tf32', False))
        channels_last = bool(train_cfg.get('channels_last', False))

        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32

        if self.rank == 0:
            self.model_logger.info(
                "Performance settings: "
                f"cudnn.benchmark={cudnn_benchmark}, tf32={tf32} "
                f"(matmul={torch.backends.cuda.matmul.allow_tf32}, "
                f"cudnn={torch.backends.cudnn.allow_tf32}), "
                f"channels_last={channels_last}"
            )
        return channels_last

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a batch tensor to the device, honouring ``train.channels_last``.

        Only floating-point tensors are converted to NHWC; index/label tensors
        are moved as-is.  3-D tensors (e.g. spectra) are moved as-is because
        channels_last only applies to 4-D/5-D conv activations.
        """
        if self.use_channels_last and tensor.dtype.is_floating_point and tensor.dim() == 4:
            return tensor.to(self.device, memory_format=torch.channels_last, non_blocking=True)
        return tensor.to(self.device, non_blocking=True)

    def _apply_channels_last(self, model: nn.Module) -> nn.Module:
        """Convert ``model`` to NHWC when ``train.channels_last`` is enabled.

        Called right after the model is built and moved to the device, BEFORE
        DDP wrapping / optimizer construction so buffers, parameters and the
        optimizer all see the same memory layout.
        """
        if self.use_channels_last:
            model.to(memory_format=torch.channels_last)
            if self.rank == 0:
                self.model_logger.info(
                    "Model parameters converted to channels_last (NHWC) memory format"
                )
        return model

    def _log_bn_freeze_audit(self, header: str = "BatchNorm / Freeze audit") -> None:
        """Log the freeze + BatchNorm tables (rank 0) and any divergence warnings.

        ``repr(model)`` shows none of this state, so without it a config change
        to ``train.fine_tuning.*`` / ``*_batchnorm_*`` leaves no trace in the log.

        This runs BEFORE the epoch loop, while the per-encoder BatchNorm policy has
        not been applied yet (``_apply_fine_tuning_policy`` only stores the intent;
        ``train_epoch`` makes it effective).  No BatchNorm can update its running
        statistics in that state, so a ``update_batchnorm_stats_*`` request is
        reported as a pending ``[BN-policy]`` note here instead of a warning, and is
        only verified against the real state by the per-epoch ``[BN-state]`` line.
        """
        if self.rank != 0 or self.model is None:
            return
        in_force = (bool(getattr(self, "_bn_policy_in_force", False))
                    and bool(getattr(self.model, "training", False)))
        if not in_force:
            header = (f"{header} [PRE-POLICY state: train_epoch() has not run yet; the "
                      f"BatchNorm policy is applied by model.train() + enforce_frozen_modes()]")
        log_bn_and_freeze_state(
            self.model, self.model_logger, config=self.config,
            optimizer=self.optimizer, header=header,
            phase="train" if in_force else "eval",
        )

    def _log_bn_state_line(self, epoch: int, where: str = "epoch start",
                           check_policy: bool = True) -> None:
        """Re-log the enforced BatchNorm state after ``model.train()``.

        ``model.train()`` at the start of every epoch resets ALL sub-modules
        (including frozen encoders) to train mode; the models re-apply their
        per-encoder policy, and this line records the result so the log proves
        the policy is actually in force.

        With ``check_policy`` the BN/freeze divergences are re-checked in the
        ``'train'`` phase - the only state in which ``update_batchnorm_stats_*``
        can be verified.  Warnings are emitted once per run (the state is static),
        prefixed with the epoch that first detected them.
        """
        if self.rank != 0 or self.model is None:
            return
        self.model_logger.info(f"[BN-state] epoch {epoch} {where}: {bn_state_line(self.model)}")
        if not check_policy or getattr(self, "_bn_policy_warned", False):
            return
        # Reached only from train_epoch() right after the enforcement, so from here
        # on the startup audit may treat the policy as in force.
        self._bn_policy_in_force = True
        divergences = [w for w in bn_freeze_warnings(self.model, config=self.config,
                                                     phase="train")
                       if w.startswith(("[BN-DIVERGENCE]", "[BN-DDP]"))]
        for warning in divergences:
            self.model_logger.warning(f"[BN-policy] first detected in epoch {epoch} ({where}): "
                                      f"{warning}")
        self._bn_policy_warned = bool(divergences)

    
    def wrap_model_ddp(self, model: nn.Module, **ddp_kwargs) -> nn.Module:
        """
        Wrap model in DDP if multi-GPU.
        
        Args:
            model: Model to wrap
            **ddp_kwargs: Additional DDP arguments
            
        Returns:
            DDP-wrapped model or original model if single GPU
        """
        if self.world_size > 1:
            default_kwargs = {
                'device_ids': [self.device_id],
                'find_unused_parameters': False,
                # 'broadcast_buffers': False,
                # 'gradient_as_bucket_view': True,
                # 'bucket_cap_mb': 25,
            }
            default_kwargs.update(ddp_kwargs)
            model = DDP(model, **default_kwargs)
            
            if self.rank == 0:
                self.model_logger.info(f"Model wrapped in DDP with {self.world_size} GPUs")
        
        return model
    
    def copy_config_to_logdir(self) -> None:
        """
        Save the fully resolved (architecture-merged) configuration to the experiment
        directory as config.yaml for reproducibility.

        This is a truthful record of the exact configuration the model was built with
        (stage config + resolved architecture, including the ``architecture_source``
        provenance) — not a byte copy of the authoring YAML, which may point at an
        architecture config elsewhere. Falls back to copying the source YAML if the
        config is not marked as resolved.
        """
        if self.rank != 0:
            return

        import shutil
        from ..utils import is_resolved_config, save_resolved_config

        dest_path = os.path.join(self.log_dir, 'config.yaml')
        if is_resolved_config(self.config):
            save_resolved_config(self.config, dest_path)
            self.model_logger.info(f"Resolved configuration saved to: {dest_path}")
        elif hasattr(self.config, 'config_path') and os.path.exists(self.config.config_path):
            shutil.copyfile(self.config.config_path, dest_path)
            self.model_logger.info(f"Configuration file copied to: {dest_path}")
    
    def copy_architecture_config_to_logdir(self, arch_config_path: Optional[str] = None, destination_path: Optional[str] = None) -> None:
        """
        Copy architecture config file to experiment directory.
        
        This ensures the architecture config used for building the model is
        saved with the experiment for reproducibility. On resume, the same
        architecture config will be available.
        
        Args:
            arch_config_path: Path to architecture config file. If None, will
                            look for it in the shared architecture folder.
            destination_path: Path where the architecture config should be copied. If None, will use a default name.
        """
        if self.rank != 0:
            return

        import shutil

        try:
            # If no path provided, use the architecture file actually used for this
            # run (stamped by load_and_merge_architecture_config) — e.g. the SSL
            # bundle's architecture in stage 2 — not a package-folder copy.
            if arch_config_path is None:
                arch_config_path = getattr(
                    getattr(self.config, 'architecture_setup', None),
                    'architecture_source',
                    None,
                )
                if not arch_config_path:
                    self.file_logger.warning(
                        "No architecture source recorded (config.architecture_setup."
                        "architecture_source) and no explicit path given — "
                        "cannot archive architecture.yaml in the experiment bundle"
                    )
                    return

            # Validate existence
            if not os.path.exists(arch_config_path):
                self.file_logger.error(f"Architecture config file not found: {arch_config_path}")
                return

            # Copy architecture config
            # arch_config_name = os.path.basename(arch_config_path)
            dest_path = os.path.join(self.log_dir, "architecture.yaml") # using a consistent name
            if destination_path:
                dest_path = os.path.join(destination_path, 'architecture.yaml')
            shutil.copyfile(arch_config_path, dest_path)
            self.file_logger.info(f"Copied architecture config to experiment: {dest_path}")

        except Exception as e:
            self.file_logger.error(f"Failed to copy architecture config from {arch_config_path}: {e}")
    
    @staticmethod
    def load_architecture_config_from_experiment(experiment_dir: str) -> Optional[str]:
        """
        Load architecture config path from experiment directory.
        
        Looks for architecture_base.yaml (or any .yaml file) in the experiment folder.
        This is used during resume or when loading pre-trained models.
        
        Args:
            experiment_dir: Path to experiment directory
            
        Returns:
            Path to architecture config if found, None otherwise
        """
        if not os.path.exists(experiment_dir):
            return None
        
        # Look for architecture_base.yaml first, then any yaml file
        arch_candidates = ['architecture_base.yaml']
        
        for candidate in arch_candidates:
            candidate_path = os.path.join(experiment_dir, candidate)
            if os.path.exists(candidate_path):
                return candidate_path
        
        # If specific name not found, look for any yaml file that looks like architecture
        for filename in os.listdir(experiment_dir):
            if 'architecture' in filename.lower() and filename.endswith('.yaml'):
                return os.path.join(experiment_dir, filename)
        
        return None
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        filename: Optional[str] = None,
        mode: str = 'ssl',
        **extra_state
    ) -> None:
        """
        Save checkpoint with atomic write.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            filename: Custom filename (default: 'last_ssl_model.pth')
            mode: Mode of training (e.g., 'ssl', 'classification')
            **extra_state: Additional state to save
        """
        if self.rank != 0:
            return  # Only rank 0 saves
        
        # Get model state (handle DDP wrapper)
        if isinstance(self.model, DDP):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        # Move to CPU while preserving the per-module `_metadata`
        # (e.g. torchvision MNASNet's `version`): a plain dict
        # comprehension would drop it, and load_state_dict() then raises
        # "version should be set to 1 or 2 instead of None".
        model_state_cpu = OrderedDict((k, v.cpu()) for k, v in model_state.items())
        if hasattr(model_state, "_metadata"):
            model_state_cpu._metadata = model_state._metadata
        
        # Build checkpoint dict
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_cpu,
            'best_metric': self.best_metric,
            'config': self.config,
        }
        
        # Add optimizer state if exists
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self._serialize_optimizer_to_cpu(
                self.optimizer.state_dict()
            )
        
        # Add scheduler state if exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Add scaler state if exists
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Add any extra state
        checkpoint.update(extra_state)
        
        # Atomic write
        if filename is None:
            filename = f'last_{mode}_model.pth'
        
        path = os.path.join(self.checkpoint_dir, filename)
        tmp_path = path + '.tmp'
        
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        
        # self.model_logger.info(f"Checkpoint saved: {path}")
        
        # Save best model copy if requested
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f'best_{mode}_model.pth')
            torch.save(checkpoint, best_path + '.tmp')
            os.replace(best_path + '.tmp', best_path)
            # self.model_logger.info(f"Best model saved. Val Loss: {self.best_metric:.4f}")
    
    def _serialize_optimizer_to_cpu(self, optimizer_state_dict: Dict) -> Dict:
        """Move optimizer state to CPU for safe serialization."""
        cpu_optim = {
            'state': {},
            'param_groups': optimizer_state_dict.get('param_groups', [])
        }
        
        for key, val in optimizer_state_dict.get('state', {}).items():
            cpu_optim['state'][key] = {}
            for k, v in val.items():
                if isinstance(v, torch.Tensor):
                    cpu_optim['state'][key][k] = v.cpu()
                else:
                    cpu_optim['state'][key][k] = v
        
        return cpu_optim
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> int:
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Epoch to resume from
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load model state
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Strip DDP prefix if present (keeps state-dict `_metadata` intact)
        target_model = self.model.module if isinstance(self.model, DDP) else self.model
        new_state_dict = strip_ddp_prefix(state_dict)
        new_state_dict = ensure_state_dict_metadata(new_state_dict, target_model)
        
        # Load into model (handle DDP wrapper)
        target_model.load_state_dict(new_state_dict)
        
        # Load optimizer if requested and available
        if load_optimizer and 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self._move_optimizer_to_device(self.optimizer, self.device)
        
        # Load scheduler if available
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler if available
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore best metric
        if 'best_metric' in checkpoint:
            self.best_metric = checkpoint['best_metric']
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        
        if self.rank == 0:
            self.model_logger.info(f"Checkpoint loaded: {checkpoint_path}")
            self.model_logger.info(f"Resuming from epoch {start_epoch}")
        
        return start_epoch
    
    def _move_optimizer_to_device(self, optimizer, device):
        """Move optimizer state tensors to specified device."""
        for state in optimizer.state.values():
            for k, v in list(state.items()):
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    @abstractmethod
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dict of training metrics
        """
        raise NotImplementedError
    
    @abstractmethod
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dict of validation metrics
        """
        raise NotImplementedError
    
    def _log_augmentation_info(self, train_loader: DataLoader) -> None:
        """Log the augmentation setup actually used by the training dataset.

        Unwraps any ``Subset`` wrapper and reports:
          - img_aug_prob (probability that augmentation is applied per sample)
          - image reader type
          - which image pipeline is running: with ``legacy_img_aug=True`` the
            legacy pipeline is authoritative and the config ``image_transforms``
            per-transform entries are ignored (the block's ``fill`` setting
            still applies); otherwise the ``image_transforms`` block is used
          - every image transform (true/false + prob + parameters)
          - fluorescence augmentation type and probability

        Never raises: logging problems must not break training.
        """
        if self.rank != 0:
            return
        try:
            from torch.utils.data import Subset
            ds = train_loader.dataset
            while isinstance(ds, Subset):
                ds = ds.dataset
        except Exception as e:  # pragma: no cover - defensive
            self.model_logger.warning(f"Could not introspect training dataset for augmentation log: {e}")
            return

        pre_lines: List[str] = []
        img_aug_prob = getattr(ds, "img_aug_prob", None)
        if img_aug_prob is not None:
            pre_lines.append(f"img_aug_prob (P(augmentation applied)): {img_aug_prob}")

        reader = getattr(ds, "image_reader", None)
        if reader is not None:
            pre_lines.append(f"image reader: {type(reader).__name__}")
        # State which pipeline is actually running: with
        # ``legacy_img_aug: True`` the legacy pipeline is authoritative and the
        # config ``image_transforms`` block is ignored completely, so the log
        # names the legacy pipeline — and its active transforms — on the
        # ``image transforms`` line itself.
        transform = getattr(ds, "transform", None)
        if getattr(ds, "legacy_img_aug", False):
            active = (list(getattr(transform, "active_transforms", []))
                      if transform is not None else [])
            pre_lines.append(
                "image transforms: LEGACY pipeline (legacy_img_aug=True; "
                f"active: {', '.join(active) or 'none'}) — "
                "the per-transform image_transforms entries are ignored "
                "(the block's fill setting still applies)")
        else:
            pre_lines.append("image transforms: image_transforms config block")

        fluo_prob = getattr(ds, "fluo_aug_prob", None)
        post_lines = []
        if fluo_prob is not None:
            fl_type = getattr(ds, "fl_aug_type", None)
            if fl_type is None:
                fluo_augmentor = getattr(ds, "fluo_augmentor", None)
                fl_type = (type(fluo_augmentor).__name__
                           if fluo_augmentor is not None else "none")
            line = f"fluorescence augmentation: type={fl_type}, prob={fluo_prob}"
            if fl_type == "pca_jitter":
                line += f", std={getattr(ds, 'fluo_pca_std', 0.1)}"
            post_lines.append(line)

        if hasattr(transform, "format_log"):
            # Class-driven rendering: the augmentation class owns the format
            kwargs = dict(pre_lines=pre_lines, post_lines=post_lines)
            transform.log_to(self.model_logger, **kwargs)          # -> training.log
            self.file_logger.info(transform.format_log(**kwargs))  # -> timestamped log + console
        else:
            # Fallback: a plain transform object without the config API
            lines = list(pre_lines)
            tf_desc = type(transform).__name__ if transform is not None else "<none>"
            lines.append("    " + tf_desc)
            lines.extend(post_lines)
            msg = "\n".join(lines)
            self.model_logger.info(msg)
            self.file_logger.info(msg)
        return

    @staticmethod
    def _module_param_groups(model) -> list:
        """Group model parameters by top-level module (first name segment).

        Returns an ordered list of ``(name, total, trainable)`` tuples ending with
        a ``('TOTAL', ...)`` row.  Works for any nn.Module layout (SSL: img_encoder,
        fl_encoder, projectors, logit_scale; classification: img_encoder,
        fl_encoder, classifier, ...).

        Multi-GPU wrappers (DDP / DataParallel) are unwrapped first, so the
        per-part rows are identical on 1 GPU and N GPUs (same structural check
        as _log_model_info).
        """
        if hasattr(model, 'module') and isinstance(getattr(model, 'module'), nn.Module):
            model = model.module
        groups: Dict[str, List[int]] = {}
        order: List[str] = []
        for name, param in model.named_parameters():
            head = name.split('.', 1)[0]
            if head not in groups:
                groups[head] = [0, 0]
                order.append(head)
            groups[head][0] += param.numel()
            if param.requires_grad:
                groups[head][1] += param.numel()
        items = [(name, groups[name][0], groups[name][1]) for name in order]
        items.append(("TOTAL", sum(t for _, t, _ in items),
                      sum(tr for _, _, tr in items)))
        return items

    def _log_model_parameters(self) -> None:
        """Log each encoder/module's parameter count separately.

        Written to ``training.log`` (model logger) and the file logger
        (timestamped log + console).  Never raises: logging problems must not
        break training.
        """
        if self.rank != 0:
            return
        try:
            groups = self._module_param_groups(self.model)
        except Exception as e:  # pragma: no cover - defensive
            self.model_logger.warning(f"Could not enumerate per-module parameters: {e}")
            return
        lines = ["--- Model Parameters ---"]
        for name, total, trainable in groups:
            lines.append(f"{name:<15}: {total:>12,} total | {trainable:,} trainable")
        msg = "\n".join(lines)
        self.model_logger.info(msg)
        try:
            self.file_logger.info(msg)
        except Exception:  # pragma: no cover - defensive
            pass

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
    ) -> None:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs to train (uses config if None)
        """
        if num_epochs is None:
            num_epochs = self.config.train.epochs
        
        if self.rank == 0:
            self.model_logger.info(f"Starting training for {num_epochs} epochs")
            # Copy config to log directory for reproducibility
            self.copy_config_to_logdir()
            # Log each encoder's parameters separately (training.log)
            self._log_model_parameters()
            # Freeze + BatchNorm state table (repr(model) cannot show it)
            self._log_bn_freeze_audit()
            # Record exactly which augmentations this run uses
            self._log_augmentation_info(train_loader)
        
        from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
        

        
        # Step-based schedulers (OneCycleLR, CosineAnnealingWarmRestarts,
        # CLIPCosineWarmupScheduler) are NEVER stepped here: they count
        # optimizer steps, and a pre-loop step would shift their whole curve
        # (OneCycleLR already performs its own initial step when constructed).
        if self.scheduler is not None and not is_step_based_scheduler(self.scheduler):
            # This pre-loop step is intentional legacy behavior to initialize
            # the LR for epoch 0; step_scheduler() silences the "stepped before
            # optimizer.step()" warning it triggers.
            step_scheduler(self.scheduler)
            if self.rank == 0:
                self.model_logger.info(
                    f"Scheduler initial step (pre-loop) — LR: {self.scheduler.get_last_lr()}"
                )
        
        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                
                # Set epoch for distributed sampler (critical for proper shuffling)
                if hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
                
                # Set epoch for dataset (critical for deterministic augmentation)
                # This ensures augmentations vary per epoch for better training diversity
                train_dataset = train_loader.dataset
                if hasattr(train_dataset, 'set_epoch'):
                    train_dataset.set_epoch(epoch)
                elif hasattr(train_dataset, 'dataset'):
                    # Handle Subset wrapper
                    if hasattr(train_dataset.dataset, 'set_epoch'):
                        train_dataset.dataset.set_epoch(epoch)
                
                # Set epoch for validation dataset too
                if val_loader is not None:
                    val_dataset = val_loader.dataset
                    if hasattr(val_dataset, 'set_epoch'):
                        val_dataset.set_epoch(epoch)
                    elif hasattr(val_dataset, 'dataset'):
                        if hasattr(val_dataset.dataset, 'set_epoch'):
                            val_dataset.dataset.set_epoch(epoch)
                
                # Train
                train_metrics = self.train_epoch(train_loader, epoch)
                
                # Call post-train hook if implemented by subclass
                if hasattr(self, 'on_train_epoch_end'):
                    self.on_train_epoch_end(train_metrics)
                
                # Validate (only on rank 0, other ranks free GPU memory)
                val_metrics = {}
                if val_loader is not None:
                    if self.rank == 0:
                        # Only rank 0 performs validation
                        val_metrics = self.validate_epoch(val_loader, epoch)
                        # Call post-validation hook if implemented by subclass
                        if hasattr(self, 'on_val_epoch_end'):
                            self.on_val_epoch_end(val_metrics)
                    else:
                        # Other ranks: free GPU cache during validation to reduce memory usage
                        torch.cuda.empty_cache()
                
                # Log metrics (rank 0 only)
                if self.rank == 0:
                    self._log_epoch_metrics(epoch, train_metrics, val_metrics)
                
                # Check if this is the best model and save checkpoint (rank 0 only)
                if self.rank == 0:
                    is_best = False
                    if val_loader is not None and val_metrics:
                        current_val_loss = val_metrics.get('loss', float('inf'))
                        if current_val_loss < self.best_metric:
                            self.best_metric = current_val_loss
                            is_best = True
                            self.model_logger.info(f"New best model! Val Loss: {current_val_loss:.4f}")
                    
                    # Save checkpoint (prefix keys to avoid collision)
                    # Always save last_model.pth, and best_model.pth when is_best=True
                    checkpoint_state = {}
                    for k, v in train_metrics.items():
                        checkpoint_state[f'train_{k}'] = v
                    for k, v in val_metrics.items():
                        checkpoint_state[f'val_{k}'] = v
                    self.save_checkpoint(epoch, is_best=is_best, **checkpoint_state)
                

                from torch.optim.lr_scheduler import ReduceLROnPlateau
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        if val_loader is not None and val_metrics:
                            self.scheduler.step(val_metrics.get('loss', float('inf')))
                    elif not is_step_based_scheduler(self.scheduler):
                        self.scheduler.step()
                
                # Early stopping check - use file-based communication (legacy approach)
                if self.early_stopping is not None and val_loader is not None:
                    if self.rank == 0:
                        stop = self.early_stopping(
                            val_metrics.get('loss', float('inf')),
                            self.model
                        )
                        if stop:
                            self.model_logger.info("Early stopping triggered")
                            # Write signal file for other ranks to read
                            early_stop_file = os.path.join(self.log_dir, '.early_stop_signal')
                            with open(early_stop_file, 'w') as f:
                                f.write('1')
                            break
                    else:
                        # Non-rank0: check for early stop signal file
                        early_stop_file = os.path.join(self.log_dir, '.early_stop_signal')
                        if os.path.exists(early_stop_file):
                            break
        
        except KeyboardInterrupt:
            if self.rank == 0:
                self.model_logger.info("\nTraining interrupted by user (Ctrl+C)")
            # Don't use sys.exit - just let exception propagate
            raise
        
        finally:
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if self.rank == 0:
            self.model_logger.info("Training completed")
            # Generate final plots if method implemented by subclass
            if hasattr(self, 'plot_training_progress'):
                self.plot_training_progress()
    
    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log metrics for current epoch to model.log, log.txt (CSV), and tensorboard."""
        # Log to model.log (human-readable)
        msg = f"Epoch {epoch+1}/{self.config.train.epochs}"
        
        if train_metrics:
            # Filter out non-scalar values (lists, arrays) for display
            train_display = {
                k: v for k, v in train_metrics.items() 
                if not isinstance(v, (list, np.ndarray))
            }
            msg += " | Train: " + ", ".join(
                f"{k}={v:.4f}" for k, v in train_display.items()
            )
        
        if val_metrics:
            # Filter out non-scalar values (lists, arrays) for display
            val_display = {
                k: v for k, v in val_metrics.items() 
                if not isinstance(v, (list, np.ndarray))
            }
            msg += " | Val: " + ", ".join(
                f"{k}={v:.4f}" for k, v in val_display.items()
            )
        
        self.model_logger.info(msg)
        
        # Store metrics for plotting (only scalar values)
        for k, v in train_metrics.items():
            if not isinstance(v, (list, np.ndarray)):
                if k not in self.train_history:
                    self.train_history[k] = []
                self.train_history[k].append(v)
        
        if val_metrics:
            for k, v in val_metrics.items():
                if not isinstance(v, (list, np.ndarray)):
                    if k not in self.val_history:
                        self.val_history[k] = []
                    self.val_history[k].append(v)
        
        # Log to log.txt (CSV format) - compatible with legacy
        log_subdir = os.path.join(self.log_dir, 'logs')
        current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
        
        log_metrics_to_file(
            log_subdir,
            epoch,
            train_metrics.get('loss', float('nan')),
            val_metrics.get('loss', float('nan')) if val_metrics else float('nan'),
            current_lr,
            accuracy=val_metrics.get('accuracy', None) if val_metrics else None,
            header_written=self._csv_header_written
        )
        self._csv_header_written = True
        
        # Log to tensorboard (only scalar values)
        if self.tb_logger is not None:
            for k, v in train_metrics.items():
                if not isinstance(v, (list, np.ndarray)):
                    self.tb_logger.log_scalar(f'train/{k}', v, epoch)
            if val_metrics:
                for k, v in val_metrics.items():
                    if not isinstance(v, (list, np.ndarray)):
                        self.tb_logger.log_scalar(f'val/{k}', v, epoch)
            self.tb_logger.log_scalar('learning_rate', current_lr, epoch)


def build_stage_dataloaders(
    config,
    train_dataset,
    val_dataset,
    rank: int,
    world_size: int,
    *,
    train_drop_last: bool,
    val_drop_last: bool,
    train_always_sharded: bool,
    val_sharded: bool,
    val_rank0_only: bool,
    worker_init_fn=None,
):
    """Build the train/val DataLoaders shared by both stage workers.

    Centralises the DataLoader boilerplate so the SSL and classification
    stages behave consistently:

      * ``train.num_workers`` is respected exactly as configured (no silent
        per-stage caps) — the effective value equals the config value.
      * The ``fork`` multiprocessing context is used whenever worker
        processes are spawned (the datasets hold open HDF5 handles, which
        are not picklable — fork is the reliable context on Linux).
      * Train sharding: ``DistributedSampler`` for SSL (always, even at
        world_size=1, to keep the legacy epoch/RNG semantics) or plain
        shuffle for single-process classification.
      * Validation sharding is stage-specific: classification shards the
        val set across ranks; SSL validates on rank 0 only.

    Args:
        config: merged configuration (reads ``train.batch_size`` and
            ``train.num_workers``).
        train_dataset: training dataset from ``build_dataset_from_config``.
        val_dataset: validation dataset (may be ``None``).
        rank / world_size: distributed rank and size.
        train_drop_last: SSL drops partial batches (contrastive loss needs
            full batches); classification keeps them.
        val_drop_last: SSL drops partial val batches; classification keeps
            them.
        train_always_sharded: use a DistributedSampler for training even at
            world_size=1 (SSL).
        val_sharded: shard the validation set across ranks (classification).
        val_rank0_only: build the val loader on rank 0 only (SSL).
        worker_init_fn: callable(worker_id) that seeds each DataLoader worker
            process for reproducibility. If None, no worker_init_fn is passed.

    Returns:
        (train_loader, val_loader_or_None)
    """
    num_workers = int(config.train.get('num_workers', 0) or 0)
    batch_size = int(config.train.get('batch_size', 1))

    def _loader(dataset, *, shuffle, sampler=None, drop_last, persistent):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=True,
            drop_last=drop_last,
            persistent_workers=persistent and num_workers > 0,
            multiprocessing_context='fork' if num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
        )

    if train_always_sharded or world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
        train_loader = _loader(
            train_dataset, shuffle=False, sampler=train_sampler,
            drop_last=train_drop_last, persistent=True,
        )
    else:
        train_loader = _loader(
            train_dataset, shuffle=True, sampler=None,
            drop_last=train_drop_last, persistent=True,
        )

    val_loader = None
    if val_dataset is not None and not (val_rank0_only and rank != 0):
        val_sampler = None
        if val_sharded and world_size > 1:
            val_sampler = DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
        val_loader = _loader(
            val_dataset, shuffle=False, sampler=val_sampler,
            drop_last=val_drop_last, persistent=False,
        )

    return train_loader, val_loader
