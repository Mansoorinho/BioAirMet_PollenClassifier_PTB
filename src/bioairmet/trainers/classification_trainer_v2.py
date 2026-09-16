'''
@file    :   classification_trainer_v2.py
@create date : 2026-01-10 14:05:08
@modify date 2026-05-27 14:57:44
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    This file defines the ClassificationTrainerV2 class, which is an enhanced trainer for classification tasks in the BioAirMet project. 
    It inherits from BaseTrainerV2 and includes specific logic for training and validating classification models, including handling frozen encoders in V2 models, tracking accuracy metrics, generating confusion matrices, and plotting training progress.
    The file also includes a legacy compatibility wrapper function, train_supervised_worker_v2, which allows using ClassificationTrainerV2 as a drop-in replacement for the older _train_supervised_worker function
    without changing the calling code in main.py.
    ]
'''

import os
import contextlib
import shutil
from typing import Dict, Optional, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .base_trainer_v2 import BaseTrainerV2, build_stage_dataloaders
from ..models import build_model_from_config
from ..models._common import strip_ddp_prefix, ensure_state_dict_metadata
from ..data import build_dataset_from_config, warn_category_map_num_classes
from ..utils import (
    build_classification_loss_from_config,
    resolve_focal_alpha,
    check_label_coverage,
    build_optimizer_from_config,
    build_scheduler_from_config,
    step_scheduler,
    is_step_based_scheduler,
    get_fine_tuning_param_groups,
    EarlyStoppingV2,
    AverageMeter,
    accuracy,
    plot_classification_metrics,
    plot_confusion_matrix,
    get_sorted_confusion_matrix,
    CLIPCosineWarmupScheduler
)


class ClassificationTrainerV2(BaseTrainerV2):
    """
    Enhanced classification trainer with modular design.
    
    Key features:
    - Inherits common functionality from BaseTrainerV2
    - Focused train_epoch() and validate_epoch() methods
    - Proper accuracy and loss tracking
    - Confusion matrix generation
    - Works with both legacy and V2 classification models
    - Handles frozen encoder properly (especially with V2 models)
    
    Usage:
        trainer = ClassificationTrainerV2(config, rank, world_size, device_id)
        trainer.setup()
        trainer.train(train_loader, val_loader)
    """
    
    def __init__(self, config, rank, world_size, device_id):
        super().__init__(config, rank, world_size, device_id)
        
        # Classification-specific attributes
        self.criterion = None
        self.grad_accum_steps = config.train.gradient_accumulation_steps
        self.use_mixed_precision = config.train.mixed_precision
        self.num_classes = config.architecture_setup.classification_model.num_classes
        
        # Track best metrics (best model based on lowest loss)
        self.best_metric = float('inf')  # Best validation loss (lower is better)
        self.best_val_acc = 0.0  # Track best accuracy for logging
        
        # Class names for confusion matrix
        self.class_names: Optional[Dict[int, str]] = None
        
        # History tracking for plotting
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []
        self.lrs: List[float] = []
        self.best_confusion_matrix: Optional[np.ndarray] = None
    
    def setup(self, steps_per_epoch) -> None:
        """
        Setup model, data, optimizer, and training components.
        
        This is separated from __init__ to allow flexible initialization.
        
        NOTE: Model weight loading (SSL pretraining, classification pretraining) is handled
        by build_model_from_config() which calls build_classification_model_from_config_v2().
        This setup() method only handles resume optimizer state loading.
        """
        # Build model (weight loading happens inside build_model_from_config)
        self.config.architecture_setup.type = 'classification'
        self.num_classes = int(self.config.architecture_setup.classification_model.num_classes)
        self.model = self._apply_channels_last(build_model_from_config(self.config).to(self.device))
        
        # If using V2 model with frozen encoders, set them to eval mode
        if hasattr(self.model, '_ft_policy'):
            if self.rank == 0:
                policy = self.model._ft_policy
                self.model_logger.info(
                    "V2 model fine-tuning policy: "
                    f"img_encoder={policy['img']}, fl_encoder={policy['fl']}"
                    + (" (encoders frozen)" if self.model._encoders_frozen else ""))
        elif hasattr(self.model, '_encoders_frozen') and self.model._encoders_frozen:
            if self.rank == 0:
                self.model_logger.info("Using V2 model with properly frozen encoders")
        
        # Wrap in DDP if multi-GPU
        # For classification with frozen encoders, we can set find_unused_parameters=False
        ddp_kwargs = {
            'find_unused_parameters': False,
            'broadcast_buffers': True,
        }
        self.model = self.wrap_model_ddp(self.model, **ddp_kwargs)
        
        # Build loss
        self.criterion = build_classification_loss_from_config(self.config).to(self.device)
        
        # Build optimizer
        param_groups = get_fine_tuning_param_groups(
            self.model, None, self.config, mode='classification'
        )
        self.optimizer = build_optimizer_from_config(self.config, param_groups)
        
        # If resuming training, load optimizer state and training metadata
        # This must happen AFTER optimizer is created but BEFORE scheduler
        if self._should_resume_training():
            self._load_resume_optimizer_state()
        
        # Build scheduler if specified
        if self.config.train.get('scheduler'):
            self.scheduler = build_scheduler_from_config(
                self.config, self.optimizer, steps_per_epoch
            )
            # The scheduler state is restored here — the optimizer-state loader
            # above runs while self.scheduler is still None, so it cannot do it.
            if self._should_resume_training():
                self._load_resume_scheduler_state()
        
        # Setup mixed precision
        if self.use_mixed_precision:
            self.scaler = GradScaler()
        
        # Setup early stopping (legacy parity: minimize validation loss)
        self.early_stopping = EarlyStoppingV2(
            patience=self.config.train.early_stopping.patience,
            min_delta=self.config.train.early_stopping.min_delta,
            mode='min',
            verbose=True,
            trace_func=self.model_logger.info,
            rank=self.rank
        )
        
        # Log model info
        if self.rank == 0:
            self._log_model_info()

            if self.scheduler is not None:
                self.model_logger.info(f"Scheduler: {self.scheduler.__class__.__name__}")
            if self.optimizer is not None:
                self.model_logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
                self.model_logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']}")
                self.model_logger.info(f"Weight Decay: {self.optimizer.param_groups[0]['weight_decay']}")
                
                # Log initialization mode
                # init_mode = getattr(self.config, 'model_initialization', {}).get('mode', 'unknown')
                pre_train_mode = getattr(self.config.model_initialization, 'pretraining', {}).get('enable', True)
                if pre_train_mode:
                    pre_train_mode = getattr(self.config.model_initialization, 'pretraining', {}).get('mode', "ssl_pretraining")
                init_mode = pre_train_mode if pre_train_mode else 'resuming'
                self.model_logger.info(f"Model Initialization Mode: {init_mode}")
    
    def _log_model_info(self) -> None:
        """Log model architecture and parameter counts."""
        effective_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        total_params = sum(p.numel() for p in effective_model.parameters())
        trainable_params = sum(
            p.numel() for p in effective_model.parameters() if p.requires_grad
        )
        
        self.model_logger.info("\n--- Classification Model Summary ---")
        self.model_logger.info(f"Model Architecture:\n{effective_model}")
        self.model_logger.info(f"Model Type: {effective_model.__class__.__name__}")
        self.model_logger.info(f"Number of Classes: {self.num_classes}")
        self.model_logger.info(f"Total Parameters: {total_params:,}")
        self.model_logger.info(f"Trainable Parameters: {trainable_params:,}")
        self.model_logger.info(f"Frozen Parameters: {total_params - trainable_params:,}")
    
    
    def _should_resume_training(self) -> bool:
        """Check if resume mode is enabled in config."""
        model_init = getattr(self.config, 'model_initialization', {})
        resume_config = model_init.get('resume', {})
        return resume_config.get('enable', False) and resume_config.get('checkpoint_path') is not None
    
    def _load_resume_optimizer_state(self) -> None:
        """
        Load optimizer state and training metadata for resuming training.
        
        This should only be called AFTER the optimizer is created.
        Restores: optimizer state, epoch, and best metrics.
        (The scheduler state is restored by _load_resume_scheduler_state(),
        which runs after the scheduler has been built.)
        
        NOTE: Model weights are loaded by build_model_from_config() via the model builder.
        This method only handles optimizer/scheduler state and training metadata.
        """
        resume_config = self.config.model_initialization.get('resume', {})
        checkpoint_path = resume_config.get('checkpoint_path')
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            if self.rank == 0:
                self.model_logger.warning(f"Resume checkpoint path not found: {checkpoint_path}")
            return
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.rank == 0:
                    self.model_logger.info("Optimizer state restored from checkpoint")
            
            # Load training metadata
            if 'epoch' in checkpoint:
                self.current_epoch = checkpoint['epoch']+1  # Resume from next epoch
                if self.rank == 0:
                    self.model_logger.info(f"Resuming from epoch {self.current_epoch}")
            
            if 'best_loss' in checkpoint:
                self.best_metric = checkpoint['best_loss']
                if self.rank == 0:
                    self.model_logger.info(f"Best loss: {self.best_metric:.4f}")
            
            if 'best_acc' in checkpoint:
                self.best_val_acc = checkpoint['best_acc']
                if self.rank == 0:
                    self.model_logger.info(f"Best accuracy: {self.best_val_acc:.2f}%")
            
            if self.rank == 0:
                self.model_logger.info(f"Resume checkpoint loaded successfully from: {checkpoint_path}")
                
        except Exception as e:
            if self.rank == 0:
                self.model_logger.error(f"Failed to load resume checkpoint: {e}")
    
    def _load_resume_scheduler_state(self) -> None:
        """
        Restore the scheduler state from the resume checkpoint.

        Must be called AFTER the scheduler has been built: the optimizer-state
        loader in setup() runs first, while self.scheduler is still None, so
        the scheduler state is restored here instead.
        """
        resume_config = self.config.model_initialization.get('resume', {})
        checkpoint_path = resume_config.get('checkpoint_path')

        if self.scheduler is None:
            return
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            if self.rank == 0:
                self.model_logger.warning(f"Resume checkpoint path not found: {checkpoint_path}")
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if self.rank == 0:
                    self.model_logger.info("Scheduler state restored from checkpoint")
            elif self.rank == 0:
                self.model_logger.warning(
                    "Resume checkpoint has no scheduler state — the scheduler "
                    "starts from the beginning of its schedule.")
        except Exception as e:
            if self.rank == 0:
                self.model_logger.error(f"Failed to load resume scheduler state: {e}")
    

    def _apply_gradient_manipulations(self, clip_ui: Dict[str, str]) -> torch.Tensor:
        """
        Unified gradient manipulation pipeline: compute norm, apply clipping, noise, 
        normalization, and image encoder scaling.
        
        Args:
            clip_ui: Dictionary to accumulate gradient metrics for logging
            
        Returns:
            grad_norm: The computed gradient norm
        """
        # Always compute gradient norm for logging
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=float('inf')  # No clipping, just compute norm
        )
        clip_ui["g_norm"] = f"{float(grad_norm):.2e}"
        
        # Apply gradient clipping if enabled
        if self.config.train.gradient_clipping.enable:
            self._apply_gradient_clipping(grad_norm, clip_ui)
        
        # Apply gradient manipulation based on norm_mode
        if getattr(self.config.train.gradient_regularization, "enabled", False):
            norm_mode = self.config.train.gradient_regularization.get("norm_mode", "gradient_noise")
            
            if norm_mode == "gradient_noise":
                self._apply_gradient_noise(clip_ui)
            elif norm_mode == "l2_norm":
                self._apply_l2_gradient_normalization(clip_ui)
            elif norm_mode == "linf_norm":
                self._apply_linf_gradient_normalization(clip_ui)
            elif norm_mode == "step_scaling":
                self._apply_step_scaling(clip_ui, grad_norm)
            else:
                raise ValueError(f"Unsupported norm_mode: {norm_mode}")
        
        return grad_norm
    
    def _apply_gradient_clipping(self, grad_norm: torch.Tensor, clip_ui: Dict[str, str]) -> None:
        """Apply gradient clipping based on configured mode."""
        mode = self.config.train.gradient_clipping.get("clipping_mode", "adaptive")
        
        if mode == "adaptive":
            current_scale = max(self.scaler.get_scale(), 1e-12) if self.scaler else 1.0
            adaptive_clip = 1.0 / current_scale
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=adaptive_clip)
            clip_ui["clip"] = f"{adaptive_clip:.2e}"
            
        else:
            # Fixed clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.train.gradient_clipping.clip_value
            )
            clip_ui["clip"] = f"{self.config.train.gradient_clipping.clip_value:.2e}"
    
    def _apply_gradient_noise(self, clip_ui: Dict[str, str]) -> None:
        """Add Gaussian noise to gradients."""
        noise_std = self.config.train.gradient_normalization.get("noise_std", 0.00001)
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.add_(torch.randn_like(p.grad) * noise_std)
        
        clip_ui["noise"] = f"{float(noise_std):.2e}"
    
    def _apply_l2_gradient_normalization(self, clip_ui: Dict[str, str]) -> None:
        """Normalize gradients using L2 norm to unit norm."""
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in self.model.parameters() 
                        if p.grad is not None]),
            2
        )
        
        # Normalize gradients to have L2 norm of 1
        for p in self.model.parameters():
            if p.grad is not None and total_norm > 0:
                p.grad.div_(total_norm + 1e-6)
        
        clip_ui["grad_norm"] = f"{float(total_norm):.2e}"
    
    def _apply_linf_gradient_normalization(self, clip_ui: Dict[str, str]) -> None:
        """Normalize gradients using L-infinity norm to unit norm."""
        total_norm = max(
            torch.norm(p.grad.detach(), float('inf')) for p in self.model.parameters() 
            if p.grad is not None
        )
        
        # Normalize gradients to have L-infinity norm of 1
        for p in self.model.parameters():
            if p.grad is not None and total_norm > 0:
                p.grad.div_(total_norm + 1e-6)
        
        clip_ui["grad_norm"] = f"{float(total_norm):.2e}"
    
    def _apply_step_scaling(self, clip_ui: Dict[str, str], grad_norm: torch.Tensor) -> None:
        """Apply step scaling: normalize direction then apply explicit step size."""
        # Step 1: normalize gradients to unit direction
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Step 2: apply explicit step size
        grad_scale = self.config.train.gradient_regularization.get("grad_scale", 3e-5)
        grad_scale = min(grad_scale, 1.0 / (grad_norm + 1e-6))
        
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.mul_(grad_scale)
        
        clip_ui['scale'] = f"{grad_scale:.2e}"
    
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dict with training metrics (loss, top1_acc, top5_acc)
        """
        
        self.model.train()
        
        # CRITICAL FIX: If using V2 model with frozen encoders, keep them in eval mode
        effective_model = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(effective_model, 'enforce_frozen_modes'):
            # Per-encoder policy: frozen parts in eval, unfrozen (full or last-N)
            # parts in train mode with live BN stats.
            effective_model.enforce_frozen_modes()
        elif hasattr(effective_model, '_encoders_frozen') and effective_model._encoders_frozen:
            effective_model.img_encoder.eval()
            effective_model.fl_encoder.eval()
        else:
            if self.rank == 0:
                self.model_logger.info("Encoders statistics are not frozen using .eval(); training model")

        # Record the BatchNorm state AFTER model.train() + the per-encoder
        # enforcement above, so the log proves the policy is in force.
        # only in the first epoch, to avoid cluttering the logs with repeated messages.
        if epoch == 0:
            self._log_bn_state_line(epoch, where="after model.train()/enforce_frozen_modes")
        
        # Metrics
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        
        # Progress bar (only on rank 0)
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1} Train",
            disable=(self.rank != 0)
        )
        
        self.optimizer.zero_grad()
        

        try:
            total_batches = len(train_loader)
        except TypeError:                      # IterableDataset-style, no __len__
            total_batches = 0
        final_window_size = total_batches % self.grad_accum_steps
        if final_window_size == 0:
            final_window_size = self.grad_accum_steps
        final_window_start = total_batches - final_window_size
        
        for batch_idx, batch in enumerate(pbar):
            # Extract data
            img1 = self._to_device(batch['image'][0])
            img2 = self._to_device(batch['image'][1])
            fl_spectra = self._to_device(batch['fluorescence'])
            labels = batch['label'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda', enabled=self.use_mixed_precision):
                outputs = self.model(img1, img2, fl_spectra)
                loss = self.criterion(outputs, labels)
                
                # Scale loss for gradient accumulation (the partial final window
                # of the epoch is divided by its own size).
                effective_accum_steps = (
                    final_window_size if batch_idx >= final_window_start
                    else self.grad_accum_steps
                )
                loss = loss / effective_accum_steps
            
            is_last_accum_step = (
                (batch_idx + 1) % self.grad_accum_steps == 0
                or (total_batches > 0 and batch_idx == total_batches - 1)
            )
            sync_ctx = (
                contextlib.nullcontext()
                if (not isinstance(self.model, DDP) or is_last_accum_step)
                else self.model.no_sync()
            )
            with sync_ctx:
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            # Update weights every grad_accum_steps
            if is_last_accum_step:
                clip_ui = {}
                
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                # Apply gradient clipping
                self._apply_gradient_manipulations(clip_ui)
                
                # Perform optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Step-based schedulers advance ONLY when the optimizer updates,
            if is_last_accum_step and is_step_based_scheduler(self.scheduler):
                step_scheduler(self.scheduler)
            
            # Compute accuracy
            acc1 = accuracy(outputs, labels, topk=(1,))[0]
            
            # Update metrics
            batch_size = img1.size(0)
            losses.update(loss.item() * effective_accum_steps, batch_size)
            top1.update(acc1[0], batch_size)
            
            # Update progress bar
            if self.rank == 0:
                postfix = {
                    'loss': losses.avg,
                    'acc1': top1.avg.item(),
                    'lr': self.optimizer.param_groups[0]['lr'],
                }
                try:
                    postfix.update(clip_ui)
                except:
                    pass
                pbar.set_postfix(postfix)
        
        
        return {
            'loss': losses.avg,
            'top1_acc': top1.avg.item(),
        }
    
    def validate_epoch(
        self, 
        val_loader: DataLoader, 
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dict with validation metrics including confusion matrix data
        """
        self.model.eval()
        
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        
        all_labels_local = []
        all_preds_local = []
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Val", disable=(self.rank != 0))
        with torch.no_grad():
            for batch in pbar:
                img1 = batch['image'][0].to(self.device, non_blocking=True)
                img2 = batch['image'][1].to(self.device, non_blocking=True)
                fl_spectra = batch['fluorescence'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                # Forward pass
                with autocast(device_type='cuda', enabled=self.use_mixed_precision):
                    outputs = self.model(img1, img2, fl_spectra)
                    loss = self.criterion(outputs, labels)
                
                # Metrics
                acc1 = accuracy(outputs, labels, topk=(1,))[0]
                
                losses.update(loss.item(), img1.size(0))
                top1.update(acc1[0], img1.size(0))
                
                # Update progress bar
                if self.rank == 0:
                    pbar.set_postfix({
                        'loss': losses.avg,
                        'acc': top1.avg.item()
                    })
                
                # Collect predictions for confusion matrix
                all_labels_local.extend(labels.cpu().numpy())
                all_preds_local.extend(outputs.argmax(dim=1).cpu().numpy())

        # Aggregate validation metrics across all ranks for parity and correctness
        if dist.is_initialized():
            total_loss_sum = torch.tensor(losses.sum, device=self.device)
            total_top1_sum = torch.tensor(top1.sum, device=self.device)
            total_count = torch.tensor(losses.count, device=self.device)

            dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_top1_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)

            denom = total_count.item()
            global_loss = total_loss_sum.item() / denom if denom > 0 else 0.0
            global_top1 = total_top1_sum.item() / denom if denom > 0 else 0.0
        else:
            global_loss = float(losses.avg)
            global_top1 = float(top1.avg.item())

        # Gather predictions to rank 0 for full-dataset confusion matrix in DDP
        all_labels = []
        all_preds = []
        if dist.is_initialized() and self.world_size > 1:
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, (all_labels_local, all_preds_local))
            if self.rank == 0:
                for labels_part, preds_part in gathered:
                    all_labels.extend(labels_part)
                    all_preds.extend(preds_part)
        else:
            if self.rank == 0:
                all_labels = all_labels_local
                all_preds = all_preds_local
        
        return {
            'loss': global_loss,
            'top1_acc': global_top1,
            'all_labels': all_labels,
            'all_preds': all_preds,
        }

    def _plot_best_confusion_matrix_from_checkpoint(self, val_loader: DataLoader) -> None:
        """
        Plot confusion matrix once after training completes by loading the best checkpoint
        and evaluating on the full validation dataset (rank 0 only).
        """
        if self.rank != 0:
            return

        best_ckpt_path = os.path.join(self.checkpoint_dir, 'best_classification_model.pth')
        if not os.path.exists(best_ckpt_path):
            self.model_logger.warning(
                f"Best checkpoint not found at {best_ckpt_path}; skipping best confusion matrix plotting."
            )
            return

        try:
            # Build standalone evaluation loader over full validation dataset
            # (avoid DistributedSampler shard when plotting final confusion matrix)
            full_val_loader = DataLoader(
                val_loader.dataset,
                batch_size=val_loader.batch_size,
                num_workers=val_loader.num_workers,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                persistent_workers=val_loader.persistent_workers,
            )

            # Load best checkpoint into a fresh model
            eval_model = build_model_from_config(self.config).to(self.device)
            checkpoint = torch.load(best_ckpt_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)

            # Handle optional DDP prefix compatibility
            # (keeps state-dict `_metadata` intact)
            cleaned_state_dict = ensure_state_dict_metadata(
                strip_ddp_prefix(state_dict), eval_model
            )
            eval_model.load_state_dict(cleaned_state_dict, strict=False)
            eval_model.eval()

            all_labels = []
            all_preds = []
            top1 = AverageMeter('Acc@1', ':6.2f')

            with torch.no_grad():
                pbar = tqdm(full_val_loader, desc="Best Checkpoint Validation", disable=False)
                for batch in pbar:
                    img1 = batch['image'][0].to(self.device, non_blocking=True)
                    img2 = batch['image'][1].to(self.device, non_blocking=True)
                    fl_spectra = batch['fluorescence'].to(self.device, non_blocking=True)
                    labels = batch['label'].to(self.device, non_blocking=True)

                    with autocast(device_type='cuda', enabled=self.use_mixed_precision):
                        outputs = eval_model(img1, img2, fl_spectra)

                    acc1 = accuracy(outputs, labels, topk=(1,))[0]
                    top1.update(acc1[0], img1.size(0))

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                    pbar.set_postfix(acc1=top1.avg.item())

            if len(all_labels) == 0 or len(all_preds) == 0:
                self.model_logger.warning("No validation samples collected for best confusion matrix plotting.")
                return

            labels_np = np.array(all_labels)
            preds_np = np.array(all_preds)
            unique_labels, counts = np.unique(labels_np, return_counts=True)
            class_counts = dict(zip(unique_labels, counts))

            cm_filtered, true_classes_present, pred_classes_all = get_sorted_confusion_matrix(
                labels_np, preds_np, class_counts
            )

            plot_subdir = os.path.join(self.log_dir, 'plots')
            os.makedirs(plot_subdir, exist_ok=True)
            numerical_to_category: Dict[int, str] = self.class_names or {}
            if not numerical_to_category:
                dataset = val_loader.dataset
                if hasattr(dataset, '_get_category_mapping'):
                    try:
                        category_mapping = dataset._get_category_mapping()
                        if category_mapping:
                            numerical_to_category = {v: k for k, v in category_mapping.items()}
                    except Exception as mapping_error:
                        self.model_logger.warning(
                            f"Could not resolve category mapping for confusion matrix: {mapping_error}"
                        )

            if not numerical_to_category:
                self.model_logger.warning(
                    "Category mapping unavailable; confusion matrix will use generic class labels."
                )
            final_acc = float(top1.avg.item())

            plot_confusion_matrix(
                cm=cm_filtered,
                total_samples=len(all_labels),
                true_classes=true_classes_present,
                pred_classes=pred_classes_all,
                class_names=numerical_to_category,
                accuracy=final_acc,
                title="Confusion Matrix - Best Checkpoint",
                save_path=plot_subdir,
                normalize=False,
            )

            plot_confusion_matrix(
                cm=cm_filtered,
                total_samples=len(all_labels),
                true_classes=true_classes_present,
                pred_classes=pred_classes_all,
                class_names=numerical_to_category,
                accuracy=final_acc,
                title="Normalized Confusion Matrix - Best Checkpoint",
                save_path=plot_subdir,
                normalize=True,
            )

            self.best_confusion_matrix = cm_filtered
            self.model_logger.info(
                f"Best confusion matrices generated from checkpoint: {best_ckpt_path}"
            )
        except Exception as e:
            self.model_logger.warning(f"Failed to generate best confusion matrix from checkpoint: {e}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
    ) -> None:
        """
        Main training loop with classification-specific logic.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs to train (uses config if None)
        """
        if num_epochs is None:
            num_epochs = self.config.train.epochs
        
        if self.rank == 0:
            # log data, data count, and class distribution info for training and validation datasets
            self.model_logger.info(f"Training dataset size: {len(train_loader.dataset)} samples")
            self.model_logger.info(f"Validation dataset size: {len(val_loader.dataset)} samples")
            self.model_logger.info(f"Starting classification training for {num_epochs} epochs")
            # Log each encoder's parameters separately (training.log)
            self._log_model_parameters()
            # Freeze + BatchNorm state table (repr(model) cannot show train/eval
            # mode, requires_grad or running-stat updates)
            self._log_bn_freeze_audit()
            # Record exactly which augmentations this run uses (training.log)
            self._log_augmentation_info(train_loader)

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate on all ranks so DDP metrics are globally reduced
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Log metrics
            if self.rank == 0:
                self._log_epoch_metrics(epoch, train_metrics, val_metrics)

                # Keep explicit history for plotting
                self.train_losses.append(float(train_metrics.get('loss', 0.0)))
                self.train_accs.append(float(train_metrics.get('top1_acc', 0.0)))
                if val_metrics:
                    self.val_losses.append(float(val_metrics.get('loss', 0.0)))
                    self.val_accs.append(float(val_metrics.get('top1_acc', 0.0)))
                self.lrs.append(float(self.optimizer.param_groups[0]['lr']))
            
            # Track best metrics (legacy parity for checkpoint selection: val_loss)
            is_best = False
            if val_metrics:
                if 'top1_acc' in val_metrics and val_metrics['top1_acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['top1_acc']
                if 'loss' in val_metrics and val_metrics['loss'] < self.best_metric:
                    self.best_metric = val_metrics['loss']
                    is_best = True
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best, mode='classification')
            
            # Step epoch-based schedulers after each epoch (only on rank 0 to avoid race conditions)
            if self.rank == 0 and self.scheduler is not None:
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                
                # ReduceLROnPlateau requires a metric to step
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # Step with validation loss (lower is better)
                    self.scheduler.step(val_metrics.get('loss', float('inf')))
                # Step-based schedulers (OneCycleLR, CosineAnnealingWarmRestarts,
                # CLIPCosineWarmupScheduler) are stepped once per optimizer step
                # in train_epoch and must NOT be stepped here.
                elif not is_step_based_scheduler(self.scheduler):
                    self.scheduler.step()
            
            # Early stopping check (broadcast decision for synchronized exit)
            if self.early_stopping is not None and val_loader is not None:
                stop = False
                if self.rank == 0:
                    # Legacy parity: early stopping on validation loss (lower is better)
                    stop = self.early_stopping(
                        val_metrics.get('loss', float('inf')),
                        self.model
                    )
                    if stop:
                        self.model_logger.info("Early stopping triggered")
                
                # Broadcast stop decision to all ranks
                if dist.is_initialized():
                    stop_tensor = torch.tensor([1 if stop else 0], device=self.device)
                    dist.broadcast(stop_tensor, src=0)
                    stop = bool(stop_tensor.item())
                
                if stop:
                    break
        
        if self.rank == 0:
            self.model_logger.info(
                f"Training completed. Best Val Loss: {self.best_metric:.4f}, "
                f"Best Val Acc: {self.best_val_acc:.2f}%"
            )
            # Generate final plots
            self.plot_training_progress()

            # Plot best confusion matrix once at the end whenever validation data exists.
            # This should still run after early stopping so the saved best checkpoint
            # gets visualized instead of being skipped.
            if val_loader is not None:
                self._plot_best_confusion_matrix_from_checkpoint(val_loader)
    
    def on_train_epoch_end(self, train_metrics: Dict[str, float]) -> None:
        """
        Hook called after each training epoch to track metrics.
        
        Args:
            train_metrics: Dictionary of training metrics
        """
        if not hasattr(self, 'train_losses'):
            self.train_losses = []
            self.train_accs = []
            self.lrs = []
        
        self.train_losses.append(train_metrics['loss'])
        self.train_accs.append(train_metrics.get('top1_acc', 0.0))
        current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
        self.lrs.append(current_lr)
    
    def on_val_epoch_end(self, val_metrics: Dict[str, float]) -> None:
        """
        Hook called after each validation epoch to track metrics.
        
        Args:
            val_metrics: Dictionary of validation metrics
        """
        if not hasattr(self, 'val_losses'):
            self.val_losses = []
            self.val_accs = []
        
        self.val_losses.append(val_metrics['loss'])
        self.val_accs.append(val_metrics.get('top1_acc', 0.0))
        
        # Store confusion matrix if available  
        if 'confusion_matrix' in val_metrics and val_metrics.get('top1_acc', 0.0) > self.best_val_acc:
            self.best_confusion_matrix = val_metrics['confusion_matrix']
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        filename: Optional[str] = None,
        mode: str = 'ssl',
        **extra_state
    ) -> None:
        """
        Override save_checkpoint to include classification-specific metrics.
        
        Saves model state, optimizer state, scheduler state, and classification metrics
        (best_loss, best_acc) for proper resume functionality.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            filename: Custom filename
            mode: Training mode (should be 'classification')
            **extra_state: Additional state to save
        """
        # Add classification-specific metrics
        extra_state['best_loss'] = self.best_metric
        extra_state['best_acc'] = self.best_val_acc
        
        # Call parent save_checkpoint
        super().save_checkpoint(
            epoch=epoch,
            is_best=is_best,
            filename=filename,
            mode=mode,
            **extra_state
        )
    
    def plot_training_progress(self) -> None:
        """Generate training progress plots after training completes."""
        if self.rank != 0:
            return
        
        if not hasattr(self, 'train_losses') or len(self.train_losses) == 0:
            self.model_logger.warning("No training history to plot")
            return
        
        plot_subdir = os.path.join(self.log_dir, 'plots')
        os.makedirs(plot_subdir, exist_ok=True)
        
        # Plot training metrics (loss, accuracy, LR)
        plot_classification_metrics(
            self.train_losses,
            self.val_losses,
            self.lrs,
            plot_subdir,
            filename_prefix="final_classification_metrics"
        )
        
        self.model_logger.info(f"Training plots saved to {plot_subdir}")


# Legacy compatibility wrapper
def train_supervised_worker_v2(rank, world_size, config, device_id):
    """
    Wrapper function for legacy compatibility.
    
    This allows using ClassificationTrainerV2 as a drop-in replacement for
    _train_supervised_worker without changing the calling code in main.py.
    
    Args:
        rank: Process rank
        world_size: Total processes
        config: Config dict
        device_id: GPU device ID
    """
    import shutil
    from torch.utils.data import DataLoader, DistributedSampler
    
    # Worker init function for reproducibility
    def _worker_init_fn(worker_id):
        import random
        import numpy as np
        import torch
        worker_seed = config.seed + rank + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)
    
    # Create trainer
    trainer = ClassificationTrainerV2(config, rank, world_size, device_id)
    # trainer.setup()
    # moved after dataloader
    
    # Build datasets
    train_dataset, val_dataset = build_dataset_from_config(config)

    # Category map vs num_classes: the configured num_classes is
    # authoritative; if the map holds labels outside [0, num_classes) the
    # samples are excluded from the datasets — warn exactly once (rank 0).
    if rank == 0:
        warn_category_map_num_classes(config, trainer.model_logger)
    
    # Build data loaders (shared with the SSL stage — see
    # build_stage_dataloaders for the exact semantics).
    train_loader, val_loader = build_stage_dataloaders(
        config, train_dataset, val_dataset, rank, world_size,
        train_drop_last=False,   # classification keeps partial batches
        val_drop_last=False,
        train_always_sharded=False,  # plain shuffle at world_size=1
        val_sharded=True,         # shard the val set across ranks
        val_rank0_only=False,
        worker_init_fn=_worker_init_fn,
    )
    # Class-coverage check: report (rank 0) which classes/samples were
    # excluded because their label is outside the configured num_classes or
    # not in the category map, and which configured classes have no training
    # samples at all (their heads would stay untrained).
    check_label_coverage(config, train_dataset, val_dataset,
                         logger=trainer.model_logger if rank == 0 else None)
    # Focal loss: derive per-class alpha weights from the training set's
    # class imbalance (no-op unless loss.name=FocalLoss_gt_Corrected and
    # alpha is absent/'auto'). All ranks compute (each builds its own loss
    # from its own config object); only rank 0 logs. Runs before setup() so
    # the experiment config.yaml copied to the log dir carries the weights
    # (standalone validation of the experiment reuses them).
    resolve_focal_alpha(config, train_dataset,
                        logger=trainer.model_logger if rank == 0 else None)
    trainer.setup(steps_per_epoch=len(train_loader))
    if rank == 0:
        trainer.model_logger.info(
            f"DataLoaders: batch_size={config.train.batch_size}, "
            f"num_workers={config.train.num_workers} (as configured in train)"
        )

    # Build class names mapping for confusion matrix plots when possible
    if val_dataset is not None and hasattr(val_dataset, '_get_category_mapping'):
        try:
            category_mapping = val_dataset._get_category_mapping()
            trainer.class_names = {v: k for k, v in category_mapping.items()}
        except Exception:
            trainer.class_names = None
    
    # Save the experiment bundle (rank 0):
    #   • config.yaml         — fully resolved config (architecture merged in)
    #   • <category map>      — the category map used for this run
    #   • architecture.yaml   — the architecture file actually used (resolved at
    #                           startup), e.g. the SSL bundle's architecture
    if rank == 0:
        trainer.copy_config_to_logdir()
        cat_map_cfg = config.data.get('cat_map_path', None)
        if cat_map_cfg:
            # Resolve the map (absolute or relative to the config dir) and archive
            # a copy next to the experiment config so it is self-contained. Warn
            # (do not crash) if the file cannot be found.
            config_dir = os.path.dirname(config.config_path) if config.config_path else None
            candidates = [cat_map_cfg]
            if config_dir:
                candidates += [os.path.join(config_dir, cat_map_cfg),
                               os.path.join(config_dir, os.path.basename(cat_map_cfg))]
            cat_map_src = next((p for p in candidates if os.path.isfile(p)), None)
            if cat_map_src is not None:
                log_root = config.logging.get('log_dir') or config.logging.checkpoint_dir
                shutil.copyfile(cat_map_src,
                                os.path.join(log_root, os.path.basename(cat_map_src)))
            else:
                print(f"Warning: category map not found for archiving (tried: {candidates})")

        # Archive the architecture config actually used for this run
        trainer.copy_architecture_config_to_logdir()
        
    # Train (no special KeyboardInterrupt handling - let it propagate naturally like legacy)
    trainer.train(train_loader, val_loader)


__all__ = ['ClassificationTrainerV2', 'train_supervised_worker_v2']

