'''
@file    :   validate_classification_v2.py
@create date : 2026-01-15 11:25:33
@modify date 2026-06-16 14:26:27
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    This script provides an enhanced validation pipeline for classification models in the BioAirMet project, specifically designed for V2 models and trainers.
    Key features include:
    - Dedicated handling for V2 models with frozen encoders, ensuring proper evaluation mode and BatchNorm behavior.
    - Comprehensive performance measurement including Top-1 accuracy, cross-entropy loss, and calibration metrics (ACE).
    - Advanced error analysis through the generation of both absolute and normalized confusion matrices with automated class sorting.
    - Support for standalone evaluation of saved checkpoints (best/last) on external HDF5 datasets.
    - Detailed per-class accuracy reporting and logging for in-depth model assessment.
    ]
'''

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
import numpy as np
import datetime
import logging
import sys
import random
from typing import Dict, Optional
from collections import defaultdict, Counter
import gc

from ..utils.config_parser import parse_config, load_experiment_config
from ..models import build_model_from_config
from ..models._common import strip_ddp_prefix, ensure_state_dict_metadata
from ..data.datasets_cls import Stage2Dataset
from ..utils.classification_losses import build_classification_loss_from_config
from ..utils.metrics import AverageMeter, accuracy
from ..utils.plots import plot_confusion_matrix, get_sorted_confusion_matrix
from ..utils.calib_tools import ace
from tqdm import tqdm
from easydict import EasyDict as edict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize
from ..models.ssl_models import build_ssl_model_from_config
from ..models.classification_models_v2 import HoloClassifierV2


def setup_logging(log_path: str) -> logging.Logger:
    """Setup logging to both console and file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path)
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(checkpoint_path: str, model: nn.Module, device: torch.device, logger: logging.Logger) -> nn.Module:
    """
    Load checkpoint into model with proper error handling.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        device: Device to load on
        logger: Logger instance
        
    Returns:
        Model with loaded weights
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        # Log checkpoint metadata if available
        if 'epoch' in checkpoint:
            logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if 'val_top1_acc' in checkpoint:
            logger.info(f"  Checkpoint val accuracy: {checkpoint['val_top1_acc']:.2f}%")
        if 'val_loss' in checkpoint:
            logger.info(f"  Checkpoint val loss: {checkpoint['val_loss']:.4f}")
    else:
        state_dict = checkpoint
    
    # Handle DDP prefix (keeps state-dict `_metadata` intact)
    new_state_dict = ensure_state_dict_metadata(strip_ddp_prefix(state_dict), model)
    
    # Load weights
    load_result = model.load_state_dict(new_state_dict, strict=False)
    
    if load_result.missing_keys:
        logger.warning(f"Missing keys: {len(load_result.missing_keys)} (expected for partial loading)")
    if load_result.unexpected_keys:
        logger.warning(f"Unexpected keys: {len(load_result.unexpected_keys)}")
    
    logger.info("✓ Checkpoint loaded successfully")
    return model


def prepare_model_for_validation(model: nn.Module, logger: logging.Logger) -> nn.Module:
    """
    Prepare model for validation by setting proper eval modes and freezing parameters.
    
    Args:
        model: Model to prepare
        logger: Logger instance
        
    Returns:
        Prepared model
    """
    model.eval()
    
    # If using V2 model with frozen encoders, ensure encoders stay in eval mode
    if hasattr(model, '_encoders_frozen') and model._encoders_frozen:
        logger.info("V2 model detected - encoders are frozen")
        if hasattr(model, 'img_encoder'):
            model.img_encoder.eval()
        if hasattr(model, 'fl_encoder'):
            model.fl_encoder.eval()
    
    # Freeze all parameters (no gradient computation needed for validation)
    for param in model.parameters():
        param.requires_grad = False
    
    # Force BatchNorm to use stored statistics
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.track_running_stats = False
    
    # Log parameter summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("Model summary:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Frozen parameters: {total_params - trainable_params:,}")
    
    if trainable_params > 0:
        logger.warning("⚠ Some parameters still trainable during validation!")
    
    return model


def validate_classification_model_v2(
    experiment_dir: str = None,
    checkpoint_type: str = "best",
    gpu_id: int = 0,
    data_path: str = None,
    save_path: str = None,
    batch_size: Optional[int] = None,
    # --- backward-compatible aliases (deprecated) ---
    checkpoint_dir: Optional[str] = None,
    plot_path: Optional[str] = None,
):
    """
    Validate a classification model using V2-compatible architecture.

    Args:
        experiment_dir: Directory containing checkpoint and config.yaml
        checkpoint_type: 'best' or 'last'
        gpu_id: GPU device ID (-1 for CPU)
        data_path: Path to validation dataset
        save_path: Directory to save results
        batch_size: Optional override for batch size
        checkpoint_dir: (deprecated) alias of experiment_dir
        plot_path: (deprecated) alias of save_path
    """
    # Resolve canonical names (accept the old kwargs for backward compatibility)
    if experiment_dir is None:
        experiment_dir = checkpoint_dir
    if save_path is None:
        save_path = plot_path if plot_path is not None else "./validation_results"
    if experiment_dir is None or data_path is None:
        raise ValueError("experiment_dir and data_path are required")

    # Setup
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0 else "cpu")
    exp_name = os.path.basename(experiment_dir)
    output_dir = os.path.join(save_path,f"{data_path.split('/')[-1].split('.')[0]}_{checkpoint_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(os.path.join(output_dir, "validation_log.txt"))
    set_seed(42)
    
    # Header
    logger.info("=" * 80)
    logger.info("BioAirMet Classification Validation (V2)")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Experiment dir: {experiment_dir}")
    logger.info(f"Type: {checkpoint_type}")
    logger.info(f"Device: {device}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)
    
    # Load config — the experiment bundle is self-contained:
    #   new bundles : config.yaml already contains the merged architecture
    #   old bundles : config.yaml + architecture.yaml are merged on the fly
    config = load_experiment_config(experiment_dir)
    logger.info(f"Loaded resolved config from: {os.path.join(experiment_dir, 'config.yaml')}")
    
    # Override settings for validation
    config.architecture_setup.type = 'classification'
    config.data.train_data_path = data_path
    
    if not hasattr(config.train, 'mixed_precision'):
        config.train.mixed_precision = False
    
    # Build dataset (no augmentation for validation)
    cat_map_cfg = config.data.get('cat_map_path', None)
    cat_map_path = None
    if cat_map_cfg:
        # Prefer the copy archived into the experiment dir by the trainer; fall
        # back to the configured path (absolute or CWD-relative). Never crash if
        # the map is missing — generic Class_0, Class_1, ... labels are used.
        for cand in (
            os.path.join(experiment_dir, os.path.basename(cat_map_cfg)),
            cat_map_cfg,
        ):
            if os.path.isfile(cand):
                cat_map_path = cand
                break
    if cat_map_path:
        print(f"Using category map from: {cat_map_path}")
    else:
        print("Warning: category map not found; plots will use generic Class_0, Class_1, ... labels")
    val_dataset = Stage2Dataset(
        hdf5_path=data_path,
        image_key=config.data.get('image_column_name', 'images'),
        fluorescence_key=config.data.get('fluorescence_spectra_column_name', 'relative_spectra'),
        label_key=config.data.get('category_number_column_name', 'category_num'),
        img_reader_type =  config.data.get('img_reader_type', 'legacy'),
        img_aug_prob=0.0,  # No augmentation
        fluo_aug_prob=0.0,
        cat_map_path=cat_map_path,
    )
    
    logger.info(f"Dataset loaded: {len(val_dataset)} samples")
    
    # Get category mapping
    category_mapping = val_dataset._get_category_mapping()
    numerical_to_category: Dict[int, str] = {v: k for k, v in (category_mapping or {}).items()}
    
    # Build dataloader
    effective_batch_size = batch_size if batch_size is not None else config.train.batch_size
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        # worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id)
    )
    
    logger.info(f"Batch size: {effective_batch_size}")
    logger.info(f"Number of batches: {len(val_loader)}")

    # Build model
    # model = build_model_from_config(config).to(device)
    ssl_base_model = build_ssl_model_from_config(config)
    num_classes = config.architecture_setup.classification_model.num_classes
    model = HoloClassifierV2(ssl_base_model, config, num_classes).to(device)
    
    logger.info(f"Model architecture: {model.__class__.__name__}")
    
    # Load checkpoint
    # checkpoint_file = f"{checkpoint_type}_model.pth"
    available_pth_files = [f for f in os.listdir(experiment_dir) if f.endswith('.pth')]
    checkpoint_file = [f for f in available_pth_files if f.startswith(f"{checkpoint_type}_") and f.endswith('.pth')]
    if not checkpoint_file:
        raise FileNotFoundError(f"No checkpoint file found for type '{checkpoint_type}' in directory: {experiment_dir}")
    checkpoint_file = checkpoint_file[0]  # Select first matching file
    checkpoint_path = os.path.join(experiment_dir, checkpoint_file)
    model = load_checkpoint(checkpoint_path, model, device, logger)
    
    # Prepare for validation
    model = prepare_model_for_validation(model, logger)
    
    # Build loss function
    criterion = build_classification_loss_from_config(config).to(device)
    
    # Validation loop
    logger.info("\nStarting validation...")
    logger.info("-" * 80)
    
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    all_labels = []
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", disable=False)
        for batch in pbar:
            img1 = batch['image'][0].to(device, non_blocking=True)
            img2 = batch['image'][1].to(device, non_blocking=True)
            fluo_features = batch['fluorescence'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            # Forward pass
            with autocast(device_type='cuda', enabled=config.train.mixed_precision):
                outputs = model(img1, img2, fluo_features)
                loss = criterion(outputs, labels)
            
            # Compute accuracy
            acc1 = accuracy(outputs, labels, topk=(1,))[0]
            
            # Update metrics
            batch_size_current = img1.size(0)
            losses.update(loss.item(), batch_size_current)
            top1.update(acc1[0], batch_size_current)
            
            # Store predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_probs.append(torch.nn.functional.softmax(outputs, dim=1).cpu())
            all_targets.append(labels.cpu())
            
            # Update progress
            pbar.set_postfix(loss=losses.avg, acc1=top1.avg.item())
    
    # Final metrics
    logger.info("-" * 80)
    logger.info(f"Validation Complete:")
    logger.info(f"  Average Loss: {losses.avg:.4f}")
    logger.info(f"  Top-1 Accuracy: {top1.avg:.2f}%")
    
    # Calibration metrics
    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)
    ace_score = ace(labels=all_targets, probs=all_probs)
    logger.info(f"  ACE (Calibration): {ace_score:.8f}")
    
    # Confusion matrix
    logger.info("\nGenerating confusion matrices...")
    num_classes = config.architecture_setup.classification_model.num_classes
    
    class_counts = Counter(all_labels)
    cm, true_classes, pred_classes = get_sorted_confusion_matrix(
        all_labels, all_preds, class_counts
    )
    
    # Normalized confusion matrix
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype('float'), 
        row_sums, 
        where=row_sums != 0, 
        out=np.zeros_like(cm, dtype='float')
    )
    
    # Plot both versions
    plot_confusion_matrix(
        cm=cm,
        total_samples=len(val_dataset),
        true_classes=true_classes,
        pred_classes=pred_classes,
        class_names=numerical_to_category,
        accuracy=top1.avg,
        save_path=output_dir,
        title='Absolute Values',
        normalize=False
    )
    
    plot_confusion_matrix(
        cm=cm_normalized,
        total_samples=len(val_dataset),
        true_classes=true_classes,
        pred_classes=pred_classes,
        class_names=numerical_to_category,
        accuracy=top1.avg,
        save_path=output_dir,
        title='Normalized Values',
        normalize=True
    )
    
    logger.info(f"Confusion matrices saved to: {output_dir}")
    
    # additional metrics
    logger.info("\nComputing additional metrics...")
    compute_additional_metrics(all_targets, all_probs, logger)
    
    # Per-class accuracy
    logger.info("\nPer-class accuracy:")
    for i, (true_cls, pred_cls) in enumerate(zip(true_classes, pred_classes)):
        if cm[i].sum() > 0:
            class_acc = 100.0 * cm[i, i] / cm[i].sum()
            class_name = numerical_to_category.get(true_cls, f"Class_{true_cls}")
            logger.info(f"  {class_name:30s}: {class_acc:6.2f}% ({cm[i, i]:4d}/{cm[i].sum():4d})")
    
    logger.info("=" * 80)
    logger.info("Validation session complete!")
    logger.info("=" * 80)
    
    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()


def compute_additional_metrics(all_targets, all_probs, logger):
    all_targets_np = all_targets.cpu().numpy()
    all_probs_np = all_probs.cpu().numpy()

    n_classes = all_probs_np.shape[1]
    classes = np.arange(n_classes)

    y_pred = np.argmax(all_probs_np, axis=1)

    # ------------------------------------------------------------------
    # AUROC / AUPRC
    # ------------------------------------------------------------------
    try:
        y_true_bin = label_binarize(
            all_targets_np,
            classes=classes
        )

        # keep only classes that have at least one positive sample
        valid_classes = y_true_bin.sum(axis=0) > 0

        macro_auc = roc_auc_score(
            y_true_bin[:, valid_classes],
            all_probs_np[:, valid_classes],
            average="macro"
        )

        micro_auc = roc_auc_score(
            y_true_bin[:, valid_classes],
            all_probs_np[:, valid_classes],
            average="micro"
        )

        macro_auprc = average_precision_score(
            y_true_bin[:, valid_classes],
            all_probs_np[:, valid_classes],
            average="macro"
        )

        micro_auprc = average_precision_score(
            y_true_bin[:, valid_classes],
            all_probs_np[:, valid_classes],
            average="micro"
        )

        logger.info(f"  Macro AUROC: {macro_auc:.4f}")
        logger.info(f"  Micro AUROC: {micro_auc:.4f}")
        logger.info(f"  Macro AUPRC: {macro_auprc:.4f}")
        logger.info(f"  Micro AUPRC: {micro_auprc:.4f}")

    except Exception as e:
        logger.warning(f"Could not compute AUROC/AUPRC: {e}")

    # ------------------------------------------------------------------
    # Precision / Recall / F1
    # ------------------------------------------------------------------
    try:
        for avg in ["micro", "macro", "weighted"]:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets_np,
                y_pred,
                average=avg,
                zero_division=0,
            )

            logger.info(
                f"  {avg.capitalize()} "
                f"P={precision:.4f} "
                f"R={recall:.4f} "
                f"F1={f1:.4f}"
            )

    except Exception as e:
        logger.warning(f"Could not compute precision/recall/F1: {e}")

def main():
    """CLI entry point for bioairmet-validate command."""
    parser = argparse.ArgumentParser(
        description="BioAirMet Classification Validation Script (V2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--experiment_dir', '--checkpoint_dir',
        dest='experiment_dir',
        type=str,
        required=True,
        help='Path to experiment directory containing checkpoint and config.yaml '
             '(--checkpoint_dir is a deprecated alias)'
    )

    parser.add_argument(
        '--checkpoint_type',
        type=str,
        default='best',
        choices=['best', 'last'],
        help='Type of checkpoint to load (default: best)'
    )

    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='GPU device ID (-1 for CPU)'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to validation dataset (HDF5 file)'
    )

    parser.add_argument(
        '--save_path', '--plot_path',
        dest='save_path',
        type=str,
        default='./validation_results',
        help='Directory to save validation results (--plot_path is a deprecated alias)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Override batch size (uses config value if not specified)'
    )

    args = parser.parse_args()

    validate_classification_model_v2(
        experiment_dir=args.experiment_dir,
        checkpoint_type=args.checkpoint_type,
        gpu_id=args.gpu_id,
        data_path=args.data_path,
        save_path=args.save_path,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
