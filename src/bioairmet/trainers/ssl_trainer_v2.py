'''
@file    :   ssl_trainer_v2.py
@create date : 2026-01-10 14:04:08
@modify date 2026-05-27 12:31:54
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    This file defines the SSLTrainerV2 class, an enhanced self-supervised learning trainer for the BioAirMet project.
    SSLTrainerV2 inherits from BaseTrainerV2 and implements focused train_epoch() and validate_epoch() methods that compute SSL-specific metrics such as alignment and uniformity.
    The trainer is designed for memory-efficient validation and includes hooks for tracking training progress and plotting metrics after training completes.
    Additionally, a train_ssl_worker_v2 function is provided as a legacy compatibility wrapper to allow using SSLTrainerV2 as a drop-in replacement for the original _train_ssl_worker without changing the calling code in
    main.py.
    ]
'''


import os
import contextlib
from typing import Dict #, Optional
import torch
# import torch.nn as nn
import torch.distributed as dist
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np
import random

from .base_trainer_v2 import BaseTrainerV2, build_stage_dataloaders
from ..models import build_model_from_config
from ..data import build_dataset_from_config
from ..utils import (
    # build_ssl_loss_from_config,
    build_optimizer_from_config,
    build_scheduler_from_config,
    step_scheduler,
    is_step_based_scheduler,
    get_fine_tuning_param_groups,
    get_resume_section,
    EarlyStoppingV2,
    AverageMeter,
    Clip_ContrastiveLoss,
    plot_ssl_metrics,
    CLIPCosineWarmupScheduler
)


def compute_ssl_loss(criterion, model_output: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute the contrastive loss from an SSL forward-pass output dict.

    Args:
        criterion: the CLIP-style contrastive loss module.
        model_output: the dict returned by ``SSLModel_SingleIMG.forward`` with keys
            ``image_embeddings``, ``fl_embeddings`` and ``logit_scale``.

    Returns:
        torch.Tensor: the scalar contrastive loss.
    """
    return criterion(
        model_output["image_embeddings"],
        model_output["fl_embeddings"],
        model_output["logit_scale"],
    )


def compute_ssl_metrics(img_embeddings: torch.Tensor,
                        fl_embeddings: torch.Tensor) -> tuple:
    """Compute the ``(alignment, uniformity)`` metrics from L2-normalised embeddings.

    Alignment is the mean squared L2 distance between the positive (image,
    fluorescence) pairs. Uniformity is the CLIP-style log-mean of
    ``exp(-t * pairwise_dist^2)`` averaged over the two modalities; it is 0.0 when
    the batch size is < 2 (``torch.pdist`` needs at least two points).

    Args:
        img_embeddings: image embeddings of shape (B, D).
        fl_embeddings: fluorescence embeddings of shape (B, D).

    Returns:
        tuple: ``(alignment, uniformity)`` as Python floats.
    """
    # Detach so the metric computation never builds a graph (works in both the
    # training (grad-enabled) and validation (no_grad) paths).
    img_norm = img_embeddings.detach()
    fl_norm = fl_embeddings.detach()

    # Alignment: L2 distance between positive pairs.
    align = ((img_norm - fl_norm) ** 2).sum(dim=1).mean().item()

    batch_size = img_norm.shape[0]
    if batch_size >= 2:
        t = 2.0  # temperature parameter
        pdist_i = torch.pdist(img_norm.float(), p=2).pow(2)
        pdist_j = torch.pdist(fl_norm.float(), p=2).pow(2)
        # Add epsilon before log to prevent log(0) = -inf
        uniform_i = (pdist_i.mul(-t).exp().mean() + 1e-8).log().item()
        uniform_j = (pdist_j.mul(-t).exp().mean() + 1e-8).log().item()
        uniform = (uniform_i + uniform_j) / 2
    else:
        # Fallback for batch_size < 2: uniformity undefined, use neutral value.
        uniform = 0.0
    return align, uniform


class SSLTrainerV2(BaseTrainerV2):
    """
    Enhanced SSL trainer with modular design.
    
    Key features:
    - Inherits common functionality from BaseTrainerV2
    - Focused train_epoch() and validate_epoch() methods
    - Proper SSL metric computation (alignment, uniformity)
    - Memory-efficient validation
    
    Usage:
        trainer = SSLTrainerV2(config, rank, world_size, device_id)
        trainer.setup()
        trainer.train(train_loader, val_loader)
    """
    
    def __init__(self, config, rank, world_size, device_id, skip_seed: bool = False):
        super().__init__(config, rank, world_size, device_id, skip_seed=skip_seed)
        
        # SSL-specific attributes
        self.criterion = None
        self.val_criterion = None
        self.grad_accum_steps = max(1, int(config.train.get('gradient_accumulation_steps', 1)))
        self.use_mixed_precision = bool(
            config.train.get('mixed_precision', True) and self.device.type == 'cuda'
        )
        
        # Metrics tracking
        self.best_val_loss = float('inf')
        
        # History for plotting
        self.train_losses = []
        self.val_losses = []
        self.val_alignments = []
        self.val_uniformities = []
        self.lrs = []
        self.logit_scales = []
    
    def setup(self, steps_per_epoch) -> None:
        """
        Setup model, data, optimizer, and training components.
        
        This is separated from __init__ to allow flexible initialization.
        """
        # Build model
        self.config.architecture_setup.type = 'ssl'
        self.model = self._apply_channels_last(build_model_from_config(self.config).to(self.device))
        
        # Apply encoder freezing (train.fine_tuning) so a frozen encoder is
        # held in eval mode with frozen BN stats, consistent with the optimizer
        # grouping in get_fine_tuning_param_groups(mode='ssl').
        ft_cfg = self.config.train.get('fine_tuning', {}) or {}
        if ft_cfg.get('freeze_image_encoder') or ft_cfg.get('freeze_fluorescence_encoder'):
            if ft_cfg.get('freeze_image_encoder'):
                self.model.freeze_encoder('img_encoder')
            if ft_cfg.get('freeze_fluorescence_encoder'):
                self.model.freeze_encoder('fl_encoder')
            if self.rank == 0:
                self.model_logger.info(
                    "SSL fine_tuning: frozen encoders = "
                    f"{sorted(self.model._frozen_encoders)}")

        # Wrap in DDP if multi-GPU
        self.model = self.wrap_model_ddp(self.model)
        
        # Build loss.
        # since we only use one gpu for validation we define two losses.
        loss_cfg = self.config.train.loss
        self.criterion = Clip_ContrastiveLoss(
            emulate_old_loss=loss_cfg.get('emulate_old_loss', True),
            local_loss=loss_cfg.get('local_loss', False),
            gather_with_grad=loss_cfg.get('gather_with_grad', False)
        ).to(self.device)

        self.val_criterion = Clip_ContrastiveLoss(
            emulate_old_loss= True, ## Validation must use local loss to avoid cross-GPU communication
        ).to(self.device)  # Validation cannot gather across ranks because only rank 0 runs it

        if self.rank == 0:
            self.model_logger.info(
                "SSL loss configured with "
                f"emulate_old_loss={loss_cfg.get('emulate_old_loss', True)}, "
                f"local_loss={loss_cfg.get('local_loss', False)}, "
                f"gather_with_grad={loss_cfg.get('gather_with_grad', True)}"
            )

        # Build optimizer
        param_groups = get_fine_tuning_param_groups(
            model=self.model, loss_module=self.criterion, config=self.config, mode='ssl'
        )
        self.optimizer = build_optimizer_from_config(self.config, param_groups)
        
        # Build scheduler if specified
        if self.config.train.get('scheduler'):
            self.scheduler = build_scheduler_from_config(
                self.config, self.optimizer, steps_per_epoch = steps_per_epoch
            )
            if self.rank == 0:
                scheduler_cfg = self.config.train.scheduler
                self.model_logger.info(
                    f"LR Scheduler: {scheduler_cfg.name}, "
                    f"Parameters: {dict(scheduler_cfg)}"
                )
        else:
            self.scheduler = None
            if self.rank == 0:
                self.model_logger.info("LR Scheduler: None")
        
        # Setup mixed precision
        if self.use_mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Setup early stopping
        self.early_stopping = EarlyStoppingV2(
            patience=self.config.train.early_stopping.patience,
            min_delta=self.config.train.early_stopping.min_delta,
            mode='min',  # Minimize loss for SSL
            verbose=True,
            trace_func=self.model_logger.info,
            rank=self.rank
        )
        
        # Log model info
        if self.rank == 0:
            self._log_model_info()
    
    def _log_model_info(self) -> None:
        """Log model architecture and parameter counts."""
        effective_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        total_params = sum(p.numel() for p in effective_model.parameters())
        trainable_params = sum(
            p.numel() for p in effective_model.parameters() if p.requires_grad
        )
        
        self.model_logger.info("\n--- Model Summary ---")
        self.model_logger.info(f"Total Parameters: {total_params:,}")
        self.model_logger.info(f"Trainable Parameters: {trainable_params:,}")
        self.model_logger.info(f"Model: {effective_model.__class__.__name__}")
        self.model_logger.info(f"Architecture: {effective_model}")
    
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
        # clip_ui["g_norm"] = f"{float(grad_norm):.2e}"
        
        # Apply gradient clipping if enabled
        if self.config.train.gradient_clipping.enable:
            self._apply_gradient_clipping(grad_norm, clip_ui)
        
        # Apply gradient manipulation based on norm_mode
        if getattr(self.config.train.gradient_regularization, "enable", False):
            norm_mode = self.config.train.gradient_regularization.get("norm_mode", "gradient_noise")
            
            if norm_mode == "gradient_noise":
                self._apply_gradient_noise(clip_ui)
            elif norm_mode == "l2_norm":
                self._apply_l2_gradient_normalization(clip_ui)
            elif norm_mode == "linf_norm":
                self._apply_linf_gradient_normalization(clip_ui)
            elif norm_mode == "step_scaling":
                self._apply_step_scaling(clip_ui, grad_norm)
            elif norm_mode == "img_encoder_grad_down_scaling":
                self._apply_imgenc_grad_scaling(clip_ui)
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
            # clip_ui["clip"] = f"{adaptive_clip:.2e}"
            
        else:
            # Fixed clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.train.gradient_clipping.clip_value
            )
            # clip_ui["clip"] = f"{self.config.train.gradient_clipping.clip_value:.2e}"
    
    def _apply_gradient_noise(self, clip_ui: Dict[str, str]) -> None:
        """Add Gaussian noise to gradients."""
        noise_std = self.config.train.gradient_regularization.get("noise_std", 0.00001)
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
    
    def _apply_imgenc_grad_scaling(self, clip_ui: Dict[str, str]) -> None:
        """Scale down gradients for image encoder parameters."""
        imgEnc_grad_scale = self.config.train.gradient_regularization.get("imgEnc_grad_scale", 0.1)
        for name, param in self.model.named_parameters():
            if param.grad is not None and 'image_encoder' in name:
                param.grad.data *= imgEnc_grad_scale
        
        clip_ui["imgEnc_grad_scale"] = f"{float(imgEnc_grad_scale):.2e}"

    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dict with training metrics (loss, alignment, uniformity, etc.)
        """
        
        self.model.train()

        # Frozen encoders (if any) are re-put in eval() mode by the model itself
        # on every forward; log the resulting state once per epoch.  The
        # train.fine_tuning BN policy checks are classification-stage, so this
        # stage only logs the state (check_policy=False).
        # only in the first epoch, to avoid cluttering the logs with repeated messages.
        if epoch == 0:
            self._log_bn_state_line(epoch, where="after model.train()", check_policy=False)

        total_batches = len(train_loader)
        final_window_size = total_batches % self.grad_accum_steps
        if final_window_size == 0:
            final_window_size = self.grad_accum_steps
        final_window_start = total_batches - final_window_size
        
        # Metrics
        losses = AverageMeter('Loss', ':.4e')
        alignment = AverageMeter('Align', ':.4f')
        uniformity = AverageMeter('Uniform', ':.4f')
        logit_scale = AverageMeter('LogitScale', ':.4f')
        learning_rate = AverageMeter('LR', ':.2e')
        
        # Progress bar (only on rank 0)
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1} Train",
            disable=(self.rank != 0)
        )
        
        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(pbar):
            clip_ui = {}
            # Extract data
            images = self._to_device(batch['image'])
            fl_features = self._to_device(batch['fluorescence'])
            
            # Forward pass with mixed precision. The model returns a dict with
            # 'image_embeddings', 'fl_embeddings' and 'logit_scale'.
            with autocast(device_type=self.device.type, enabled=self.use_mixed_precision):
                model_output = self.model(images, fl_features)
                loss = compute_ssl_loss(self.criterion, model_output)
            img_embeddings = model_output['image_embeddings']
            fl_embeddings = model_output['fl_embeddings']
            model_logit_scale = model_output['logit_scale']
                
            # Scale loss for gradient accumulation.
            # The final partial accumulation window (if any) uses its actual size
            # so that no gradients are silently dropped or under-weighted.
            effective_accum_steps = (
                final_window_size if batch_idx >= final_window_start else self.grad_accum_steps
            )
            loss = loss / effective_accum_steps
            
            # Use model.no_sync() for DDP on all but the last accumulation step.
            # Legacy mode keeps the original all-reduce behavior on every backward pass.
            is_last_accum_step = (
                (batch_idx + 1) % self.grad_accum_steps == 0
                or batch_idx == total_batches - 1
            )
            sync_ctx = (
                contextlib.nullcontext()
                if (not isinstance(self.model, DDP) or is_last_accum_step)
                else self.model.no_sync()
            )
            # sync_ctx = contextlib.nullcontext() # legacy behavior: always synchronize (no no_sync)
            with sync_ctx:
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            # Update weights every grad_accum_steps
            if is_last_accum_step:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                # Apply all gradient manipulations
                self._apply_gradient_manipulations(clip_ui)
                
                # Perform optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)

                # Step-based schedulers advance once per OPTIMIZER step, i.e.
                # once per gradient-accumulation window — never per micro-batch.
                # (Stepping per micro-batch made the schedule run
                # `gradient_accumulation_steps` times too fast.)
                if is_step_based_scheduler(self.scheduler):
                    step_scheduler(self.scheduler)

            # Compute SSL metrics (detached for efficiency).
            with torch.no_grad():
                align, uniform = compute_ssl_metrics(img_embeddings, fl_embeddings)
            
            # Update metrics
            batch_size = images.size(0)
            losses.update(loss.item() * effective_accum_steps, batch_size)
            alignment.update(align, batch_size)
            uniformity.update(uniform, batch_size)
            logit_scale.update(model_logit_scale.item() if model_logit_scale is not None else None, batch_size)

            # Update progress bar
            if self.rank == 0:
                postfix = {
                    "loss": f"{losses.avg:.3f}",
                    "align": f"{alignment.avg:.2f}",
                    "uniform": f"{uniformity.avg:.2f}",
                    "lr": f"{(self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0):.2e}",
                }
                # only show the logit when it is learnable
                if self.config.train.loss.learnable_temp:
                    postfix["logit"] = f"{logit_scale.avg:.2f}"
                postfix.update(clip_ui)
                pbar.set_postfix(postfix)
        
        if dist.is_initialized():
            _loss_sum = torch.tensor(losses.sum, device=self.device)
            _loss_count = torch.tensor(losses.count, device=self.device)
            dist.all_reduce(_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(_loss_count, op=dist.ReduceOp.SUM)
            if self.rank == 0:
                losses.avg = _loss_sum.item() / _loss_count.item() if _loss_count.item() > 0 else 0.0

        return {
            'loss': losses.avg,
            'alignment': alignment.avg,
            'uniformity': uniformity.avg,
            'logit_scale': logit_scale.avg,
            'learning_rate': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
        }
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch. This method should only be called on rank 0.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dict with validation metrics
        """
        # Validation should only be called on rank 0
        assert self.rank == 0, "validate_epoch should only be called on rank 0"
        
        self.model.eval()
        
        losses = AverageMeter('Loss', ':.4e')
        alignment = AverageMeter('Align', ':.4f')
        uniformity = AverageMeter('Uniform', ':.4f')
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Val", disable=(self.rank != 0))
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device, non_blocking=True)
                fl_features = batch['fluorescence'].to(self.device, non_blocking=True)
                
                # validation to produce numerically identical results.
                with autocast(device_type=self.device.type, enabled=self.use_mixed_precision):
                    model_output = self.model(images, fl_features)
                    loss = compute_ssl_loss(self.val_criterion, model_output)

                # Metrics (match the training computation exactly). The embeddings
                # are already L2-normalised inside the model, so no re-normalisation.
                img_embeddings = model_output['image_embeddings']
                fl_embeddings = model_output['fl_embeddings']
                align, uniform = compute_ssl_metrics(img_embeddings, fl_embeddings)
                
                losses.update(loss.item(), images.size(0))
                alignment.update(align, images.size(0))
                uniformity.update(uniform, images.size(0))
                # CRITICAL: Delete tensors after each batch (match legacy behavior)
                del images, fl_features, img_embeddings, fl_embeddings, loss, model_output
                torch.cuda.empty_cache()
                
                # Update progress bar
                if self.rank == 0:
                    pbar.set_postfix({
                        'loss': losses.avg,
                        'align': alignment.avg,
                        'uniform': uniformity.avg
                    })
        
        return {
            'loss': losses.avg,
            'alignment': alignment.avg,
            'uniformity': uniformity.avg,
        }
    
    def on_train_epoch_end(self, train_metrics: Dict[str, float]) -> None:
        """
        Hook called after each training epoch to track metrics.
        
        Args:
            train_metrics: Dictionary of training metrics
        """
        self.train_losses.append(train_metrics['loss'])
        current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
        self.lrs.append(current_lr)
        self.logit_scales.append(train_metrics.get('logit_scale', 0.0))
    
    def on_val_epoch_end(self, val_metrics: Dict[str, float]) -> None:
        """
        Hook called after each validation epoch to track metrics.
        
        Args:
            val_metrics: Dictionary of validation metrics
        """
        self.val_losses.append(val_metrics['loss'])
        self.val_alignments.append(val_metrics.get('alignment', 0.0))
        self.val_uniformities.append(val_metrics.get('uniformity', 0.0))
    
    def plot_training_progress(self) -> None:
        """Generate training progress plots after training completes."""
        if self.rank != 0:
            return
        
        plot_subdir = os.path.join(self.log_dir, 'plots')
        os.makedirs(plot_subdir, exist_ok=True)
        
        # Plot SSL metrics with validation data if available
        if len(self.val_losses) > 0:
            plot_ssl_metrics(
                self.train_losses,
                self.val_losses,
                self.lrs,
                self.val_alignments,
                self.val_uniformities,
                plot_subdir,
                filename_prefix="final_ssl_training_metrics"
            )
        else:
            # Plot without validation
            plot_ssl_metrics(
                self.train_losses,
                [],
                self.lrs,
                [],
                [],
                plot_subdir,
                filename_prefix="final_ssl_training_metrics"
            )
        
        self.model_logger.info(f"Training plots saved to {plot_subdir}")

# Legacy compatibility wrapper
def train_ssl_worker_v2(rank, world_size, config, device_id):
    """
    Wrapper function for legacy compatibility.
    
    This allows using SSLTrainerV2 as a drop-in replacement for
    _train_ssl_worker without changing the calling code in main.py.
    
    Args:
        rank: Process rank
        world_size: Total processes
        config: Config dict
        device_id: GPU device ID
    """
    import os
    import shutil
    from torch.utils.data import DataLoader, DistributedSampler
    from functools import partial
    
    # Worker init function for HDF5
    def _worker_init_fn(worker_id, base_seed, rank):
        import random
        import numpy as np
        import torch
        worker_seed = base_seed + rank + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)
        # os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    
    # CRITICAL: Build datasets BEFORE model to match legacy RNG state
    # Dataset building consumes random numbers (augmentation setup, shuffling)
    # This must happen before model initialization for reproducibility
    seed = config.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed} for random, numpy, torch, and cudnn.")
    
    train_dataset, val_dataset = build_dataset_from_config(config)
    
    # Create trainer and setup model.
    # skip_seed=True because seed was already set above before dataset building.
    # Passing skip_seed=False would reset the RNG here and give different weight
    # initialization compared to legacy (which sets seed once before datasets).
    trainer = SSLTrainerV2(config, rank, world_size, device_id, skip_seed=True)
    # trainer.setup()
    # moved after dataloader to use the steps per epoch in scheduler
    
    # Build data loaders (shared with the classification stage — see
    # build_stage_dataloaders for the exact semantics).
    train_loader, val_loader = build_stage_dataloaders(
        config, train_dataset, val_dataset, rank, world_size,
        train_drop_last=True,      # contrastive loss needs full batches
        val_drop_last=True,
        train_always_sharded=True,  # SSL always uses a DistributedSampler
        val_sharded=False,          # validation runs on rank 0 only
        val_rank0_only=True,
        worker_init_fn=lambda wid: _worker_init_fn(wid, config.seed, rank),
    )
    trainer.setup(steps_per_epoch=len(train_loader))
    if rank == 0:
        trainer.model_logger.info(
            f"DataLoaders: batch_size={config.train.batch_size}, "
            f"num_workers={config.train.num_workers} (as configured in train)"
        )
    
    # Log the dataset sizes (rank 0).  The full data-augmentation block — the
    # legacy-vs-enhanced switch, every image transform with its probability and
    # parameters, and the fluorescence augmentation — is written at the start of
    # BaseTrainerV2.train() via the shared _log_augmentation_info() helper, so
    # SSL and classification emit the identical "--- Data Augmentation (TRAIN)
    # ---" block.  (An earlier version read the training dataset's
    # ``augmentation`` attribute here, but the config-driven Stage1Dataset
    # exposes ``transform``/``fluorescence_augmentation`` instead, so that check
    # never matched and silently logged nothing.)
    if rank == 0:
        trainer.model_logger.info(f"Train dataset size: {len(train_dataset)}")
        if val_dataset:
            trainer.model_logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Save the experiment bundle (rank 0):
    #   • config.yaml         — fully resolved config (architecture merged in)
    #   • architecture.yaml   — the architecture file actually used for this run
    if rank == 0:
        trainer.copy_config_to_logdir()
        trainer.copy_architecture_config_to_logdir()
    
    # Resume from checkpoint if enabled.
    # Unified location (both stages): model_initialization.resume;
    # legacy location: top-level `resume` section (old configs / bundles).
    resume_cfg = get_resume_section(config)
    if resume_cfg.get('enable', False) and resume_cfg.get('checkpoint_path'):
        checkpoint_path = resume_cfg['checkpoint_path']
        if os.path.exists(checkpoint_path):
            start_epoch = trainer.load_checkpoint(
                checkpoint_path,
                load_optimizer=resume_cfg.get('load_optimizer', True)
            )
            trainer.current_epoch = start_epoch
            if rank == 0:
                trainer.model_logger.info(f"Resuming from epoch {start_epoch}")
        elif rank == 0:
            trainer.model_logger.warning(
                f"Resume enabled but checkpoint not found: {checkpoint_path}. Starting from scratch."
            )
    
    # Train (no special KeyboardInterrupt handling - let it propagate naturally like legacy)
    trainer.train(train_loader, val_loader)


__all__ = ['SSLTrainerV2', 'train_ssl_worker_v2']
