'''
@file    :   inference.py
@create date : 2025-09-13 13:05:36
@modify date 2026-02-23 14:40:08
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    Inference engine for trained BioAirMet classification models.
    This module (formerly ``test_model.py``) provides run_inference, which
    loads the model checkpoint, performs inference on unlabeled raw event
    data or labeled HDF5 datasets with optional confidence thresholding, and
    saves detailed predictions (top-3 and full distributions) to a CSV file.
    It also generates summary statistics and confusion matrices for labeled
    datasets.
    Additionally, a legacy entry point (main_worker_test) is included for
    backward compatibility with config-driven distributed testing.
    ]
'''

import argparse
import os
import time
import json
import glob
import logging
import sys
import datetime
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast
from tqdm import tqdm

from ..utils.config_parser import (
    parse_config,
    load_and_merge_architecture_config,
    load_experiment_config,
    resolve_inference_settings,
)
from ..data.datasets_cls import ValidationDataset_Unlabeled, Stage2Dataset, image_reader_kwargs
from ..models import build_model_from_config
from ..models._common import strip_ddp_prefix, ensure_state_dict_metadata
from ..models.ssl_models import build_ssl_model_from_config
from ..models.classification_models_v2 import HoloClassifierV2
from easydict import EasyDict as edict

from ..utils.metrics import AverageMeter, accuracy
from ..utils.plots import plot_confusion_matrix, get_sorted_confusion_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_num_workers(num_workers: int, logger=None) -> int:
    """Cap DataLoader worker count at the number of available CPUs.

    Requesting more workers than there are CPUs cannot speed inference up
    and can stall the run, so the count is overwritten with
    ``os.cpu_count()`` (and a warning is emitted) — this guarantees the
    inference run proceeds on any machine.
    """
    if num_workers is None or num_workers <= 0:
        return num_workers or 0
    cpu_count = os.cpu_count() or 1
    if num_workers > cpu_count:
        msg = (f"num_workers={num_workers} exceeds the CPU count ({cpu_count}); "
               f"overwriting with num_workers={cpu_count}")
        if logger is not None:
            logger.warning(msg)
        else:
            print(f"WARNING: {msg}")
        return cpu_count
    return num_workers


def _setup_logging(log_path: str) -> logging.Logger:
    """
    Create a logger for one inference run (console + log file).

    Uses an explicit per-run logger (not ``logging.basicConfig``) so that
    repeated ``run_inference`` calls in the same process (API / notebook
    usage) each get their own log file instead of silently reusing the
    first run's handler.
    """
    log_dir = os.path.dirname(os.path.abspath(log_path))
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"bioairmet.inference.{os.path.basename(log_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    fileh = logging.FileHandler(log_path, mode="w")
    fileh.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(fileh)
    return logger


def _load_checkpoint(checkpoint_path: str, model: nn.Module, device: torch.device,
                     logger: logging.Logger) -> nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    # Checkpoints store non-tensor objects (epoch, optimizer state, metrics),
    # so weights_only=False is intentionally required here.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = ckpt.get("model_state_dict", ckpt)
    if "epoch" in ckpt:
        logger.info(f"  Saved at epoch  : {ckpt['epoch']}")
    if "val_top1_acc" in ckpt:
        logger.info(f"  Val accuracy    : {ckpt['val_top1_acc']:.2f}%")

    # Strip DDP "module." prefix (keeps state-dict `_metadata` intact)
    state_dict = ensure_state_dict_metadata(strip_ddp_prefix(state_dict), model)

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        logger.warning(f"  Missing keys    : {len(result.missing_keys)}")
    if result.unexpected_keys:
        logger.warning(f"  Unexpected keys : {len(result.unexpected_keys)}")

    logger.info("✓ Checkpoint loaded")
    return model


def _prepare_model(model: nn.Module, logger: logging.Logger) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.track_running_stats = False
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model ready  |  params: {total:,}  |  mode: eval / no-grad")
    return model


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_inference(
    data_path: str,
    checkpoint_path: str,
    save_path: str,
    config_path: str,
    gpu_id: int = 0,
    batch_size: int = 64,
    num_workers: int = 6,
    img_reader_type: str = "legacy",
    confidence_threshold: float = 0.0,
    min_intensity_delta: float = 0.1,
    min_area: int = 500,
    min_solidity: float = 0.7,
    skip_validation: bool = False,
    output_mode: str = "compact"
):
    """
    Run inference on unlabeled raw event data or a labeled HDF5 file.

    Parameters
    ----------
    data_path             : path to raw event directory (unlabeled) OR an HDF5 file (labeled).
    checkpoint_path       : path to the saved ``*.pth`` classification model checkpoint.
    save_path             : base output directory. For labeled HDF5 input the
                            outputs are written into a subdirectory named after
                            the HDF5 file without extension (e.g.
                            ``save_path/dataxyz/``); for unlabeled raw-directory
                            input they are written directly into ``save_path``.
    config_path           : path to the YAML config that was used to *train* the classifier.
    gpu_id                : CUDA device ID (-1 → CPU).
    batch_size            : inference batch size.
    num_workers           : DataLoader worker count. Capped automatically at
                            the machine's CPU count (with a warning) so the
                            run never stalls on a small machine.
    img_reader_type       : image normalisation strategy ('legacy', 'minmax', …).
    confidence_threshold  : minimum top-1 probability to assign a class label.
                            Samples below this threshold are labelled "low_confidence".
                            0.0 (default) disables thresholding (always assign top-1).
    min_intensity_delta   : minimum max–min intensity delta for a valid spectrum
                            (unlabeled raw-directory mode only).
    min_area              : minimum region area in pixels (unlabeled mode only).
    min_solidity          : minimum region solidity (unlabeled mode only).
    skip_validation       : when True (unlabeled mode only), the per-event
                            quality checks above are skipped and every event
                            with usable image files is kept, even events that
                            fail the area / solidity / intensity-delta gates.
                            Default False — events that fail the quality
                            checks are dropped from the dataset.
    output_mode           : predictions CSV detail level.
                            'compact'  (default) → image_path | predicted_class | probability
                            'detailed'           → compact + predicted_class_id,
                                                    top-2/top-3 alternatives and the full
                                                    per-class probability distribution
                            (labeled mode always appends true_class | correct).

    Returns
    -------
    dict
        Machine-readable result — suitable for programmatic / API use:
        ``mode``, ``output_mode``, ``num_samples``, ``csv_path``,
        ``summary_json_path``, ``log_path``, ``predictions`` (pandas
        DataFrame), ``class_map`` (int → name) and ``summary`` (dict with
        accuracy, confidence statistics and per-class counts).
    """
    if output_mode not in ("compact", "detailed"):
        raise ValueError(
            f"output_mode must be 'compact' or 'detailed', got {output_mode!r}")
    # Labeled HDF5 input: write all outputs into a subdirectory named after
    # the HDF5 file (without extension), e.g.
    #   ./validation_results/dataxyz/matrix_normalized.png
    # so several files evaluated against the same model don't mix up.
    # Unlabeled raw-directory mode writes directly into save_path.
    is_unlabeled = os.path.isdir(data_path)
    if is_unlabeled:
        out_dir = save_path
    else:
        hdf5_stem = os.path.splitext(os.path.basename(data_path))[0] or "hdf5"
        out_dir = os.path.join(save_path, hdf5_stem)
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(out_dir, f"inference_{timestamp}.log")
    logger = _setup_logging(log_file)

    # Never spawn more dataloader workers than there are CPUs.
    num_workers = _clamp_num_workers(num_workers, logger=logger)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0 else "cpu")

    logger.info("=" * 70)
    logger.info("BioAirMet  --  Inference")
    logger.info("=" * 70)
    logger.info(f"Timestamp      : {timestamp}")
    logger.info(f"Data path      : {data_path}")
    logger.info(f"Checkpoint     : {checkpoint_path}")
    logger.info(f"Output dir     : {out_dir}")
    logger.info(f"Config         : {config_path}")
    logger.info(f"Device         : {device}")
    logger.info(f"Output mode    : {output_mode}")
    logger.info(f"Threshold      : {confidence_threshold if confidence_threshold > 0 else 'disabled (top-1 always assigned)'}")
    logger.info("=" * 70)

    # ---- Config -------------------------------------------------------
    # Prefer the self-contained experiment bundle (config.yaml +
    # architecture.yaml side by side; old bundles are merged on the fly).
    # Otherwise fall back to the shared merge, which resolves the
    # architecture file from the config's own references / package folder.
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if os.path.basename(config_path) == 'config.yaml' and os.path.exists(os.path.join(config_dir, 'architecture.yaml')):
        config = load_experiment_config(config_dir)
        logger.info(f"Loaded resolved experiment config from: {config_dir}")
    else:
        config = parse_config(config_path)
        config.config_path = os.path.abspath(config_path)
        config, arch_path = load_and_merge_architecture_config(config)
        logger.info(f"Loaded config from: {config_path}")
        logger.info(f"Architecture resolved from: {arch_path}")
        
    
    # Override settings for validation
    config.architecture_setup.type = 'classification'
    config.data.train_data_path = data_path
    

    # ---- Resolve category map -----------------------------------------
    # Priority 1: path stored in config, resolved relative to experiment dir
    # Priority 2: built-in cats_dict.txt shipped with the package
    experiment_dir = os.path.dirname(os.path.abspath(config_path))
    _raw_cat_map   = config.data.get("cat_map_path", None)

    cat_map_path = None
    if _raw_cat_map:
        # Try as-is (absolute) then relative to experiment dir
        candidate = _raw_cat_map if os.path.isabs(_raw_cat_map) \
                    else os.path.join(experiment_dir, os.path.basename(_raw_cat_map))
        if os.path.isfile(candidate):
            cat_map_path = candidate
            logger.info(f"Category map   : {cat_map_path}  (experiment dir)")
        else:
            logger.warning(f"cat_map_path '{_raw_cat_map}' not found relative to experiment dir")

    if cat_map_path is None:
        # Fall back to the built-in cats_dict.txt bundled with the package
        _pkg_root    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _builtin     = os.path.join(_pkg_root, "config", "classification", "cats_dict.txt")
        if os.path.isfile(_builtin):
            cat_map_path = _builtin
            logger.info(f"Category map   : {cat_map_path}  (built-in fallback)")
        else:
            logger.warning("No category map found — class indices will be used as names")

    effective_img_reader = img_reader_type or config.data.get("img_reader_type", "legacy")

    # ---- Detect mode (unlabeled dir vs. labeled HDF5) -----------------
    is_unlabeled = os.path.isdir(data_path)
    logger.info(f"Mode           : {'unlabeled (raw directory)' if is_unlabeled else 'labeled (HDF5)'}")

    if is_unlabeled:
        # image_size / image_normalization are passed at config level (NOT
        # pre-extracted) because the dataset's path mode re-resolves them
        # through image_reader_kwargs — which is where the
        # image_normalization.enable flag is honoured.
        dataset = ValidationDataset_Unlabeled(
            unlabeled_path=data_path,
            img_reader_type=effective_img_reader,
            image_size=config.data.get('image_size', 200),
            image_normalization=config.data.get('image_normalization', None),
            min_intensity_delta=min_intensity_delta,
            min_area=min_area,
            min_solidity=min_solidity,
            skip_validation=skip_validation
        )
        has_labels = False
    else:
        try:
            head_num_classes = int(config.architecture_setup.classification_model.num_classes)
        except Exception:
            head_num_classes = None
        dataset = Stage2Dataset(
            hdf5_path=data_path,
            image_key=config.data.get("image_column_name", "images"),
            fluorescence_key=config.data.get("fluorescence_spectra_column_name", "relative_spectra"),
            label_key=config.data.get("category_number_column_name", "category_num"),
            img_reader_type=effective_img_reader,
            img_aug_prob=0.0,
            fluo_aug_prob=0.0,
            cat_map_path=cat_map_path,
            num_classes=head_num_classes,
            **image_reader_kwargs(config.data),
        )
        has_labels = True
        if head_num_classes is not None:
            try:
                n_raw = int(dataset.class_counts(raw=True).sum())
            except Exception:
                n_raw = None
            if n_raw is not None and n_raw != len(dataset):
                logger.warning(
                    f"{n_raw - len(dataset)} sample(s) excluded from the inference "
                    f"dataset: their labels are outside num_classes={head_num_classes} "
                    f"(or not in the category map) — they are skipped in this run."
                )

    logger.info(f"Dataset size   : {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=lambda wid: np.random.seed(42 + wid),
        # Python ≥ 3.14 defaults to the forkserver start method, which pickles
        # the dataset — but it holds unpicklable threading.Lock / h5py handles.
        # Use fork explicitly (same convention as the SSL trainer).
        multiprocessing_context="fork" if num_workers > 0 else None,
    )

    # ---- Model --------------------------------------------------------
    # model = build_model_from_config(config).to(device)
    num_classes = config.architecture_setup.classification_model.num_classes
    
    ssl_base_model = build_ssl_model_from_config(config)
    model = HoloClassifierV2(ssl_base_model, config, num_classes).to(device)
    model = _load_checkpoint(checkpoint_path, model, device, logger)
    model = _prepare_model(model, logger)

    # Category mapping (int → class name)
    if has_labels and cat_map_path and os.path.exists(cat_map_path):
        num2name = {v: k for k, v in dataset._get_category_mapping().items()}
    elif cat_map_path and os.path.exists(cat_map_path):
        from ..data.datasets_cls import Stage2Dataset as _S2
        _tmp = _S2.__new__(_S2)
        _tmp.cat_map_path = cat_map_path
        num2name = {v: k for k, v in (_tmp._get_category_mapping() or {}).items()}
    else:
        num2name = {}

    # ---- Inference loop -----------------------------------------------
    logger.info("\nRunning inference ...")
    all_preds      = []
    all_probs      = []
    all_labels     = []
    all_img_paths  = []
    top1_meter     = AverageMeter("Acc@1", ":6.2f")

    with torch.no_grad():
        pbar = tqdm(loader, desc="Inference")
        for batch in pbar:
            img1 = batch["image"][0].to(device, non_blocking=True)
            img2 = batch["image"][1].to(device, non_blocking=True)
            fluo = batch["fluorescence"].to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(img1, img2, fluo)

            probs = torch.softmax(logits, dim=1).cpu()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_probs.append(probs)

            if "image_path" in batch:
                all_img_paths.extend(batch["image_path"])

            if has_labels:
                labels = batch["label"].to(device, non_blocking=True)
                all_labels.extend(labels.cpu().numpy().tolist())
                acc1 = accuracy(logits, labels, topk=(1,))[0]
                top1_meter.update(acc1[0], img1.size(0))
                pbar.set_postfix(acc1=f"{top1_meter.avg:.2f}")

    all_probs = torch.cat(all_probs).numpy()  # (N, C)
    all_probs = all_probs.astype(np.float64)  # float32→float64 so np.round(…, 4) is exact
    N, C = all_probs.shape

    # ---- Top-3 per sample -------------------------------------------
    top3_indices = np.argsort(all_probs, axis=1)[:, ::-1][:, :3]   # (N, 3) descending
    top3_probs   = all_probs[np.arange(N)[:, None], top3_indices]   # (N, 3)

    top1_conf  = top3_probs[:, 0]                                    # (N,) top-1 probability
    top1_class = top3_indices[:, 0]                                  # (N,) top-1 class index

    # ---- Apply confidence threshold ----------------------------------
    LOW_CONF_LABEL = "low_confidence"
    use_threshold  = confidence_threshold > 0.0
    if use_threshold:
        final_class_num  = np.where(top1_conf >= confidence_threshold, top1_class, -1).astype(int).tolist()
        final_class_name = [
            num2name.get(int(c), str(c)) if c >= 0 else LOW_CONF_LABEL
            for c in final_class_num
        ]
    else:
        final_class_num  = [int(c) for c in top1_class.tolist()]
        final_class_name = [num2name.get(int(c), str(c)) for c in final_class_num]

    n_low = sum(1 for c in final_class_num if c < 0)

    # ---- Build output DataFrame ---------------------------------------
    # output_mode = "compact"  →  image_path | predicted_class | probability
    #                            (+ true_class | correct in labeled mode)
    # output_mode = "detailed" →  compact + predicted_class_id, top-2 / top-3
    #                            alternatives and the FULL per-class softmax
    #                            distribution (prob_<classname> columns)
    records = {}

    # Data path first (unlabeled raw-event mode)
    if all_img_paths:
        records["image_path"] = all_img_paths

    # Primary prediction + its probability
    records["predicted_class"] = final_class_name   # "low_confidence" when below threshold
    records["probability"]     = np.round(top1_conf, 4).tolist()

    if output_mode == "detailed":
        records["predicted_class_id"] = [c if c >= 0 else -1 for c in final_class_num]
        # 2nd and 3rd most likely alternatives (class name + probability)
        records["top2_class"] = [num2name.get(int(i), str(i)) for i in top3_indices[:, 1]]
        records["top2_prob"]  = np.round(top3_probs[:, 1], 4).tolist()
        records["top3_class"] = [num2name.get(int(i), str(i)) for i in top3_indices[:, 2]]
        records["top3_prob"]  = np.round(top3_probs[:, 2], 4).tolist()
        # Full softmax distribution (all classes) — wide but useful for deeper analysis
        for c in range(C):
            col_name = num2name.get(c, f"class_{c}")
            records[f"prob_{col_name}"] = np.round(all_probs[:, c], 4).tolist()

    # Stable 0-based row index (matches dataset order)
    records["sample_index"] = list(range(N))

    # Ground-truth columns (labeled mode)
    if has_labels:
        records["true_class"] = [num2name.get(int(l), str(l)) for l in all_labels]
        # Threshold-aware: a sample labelled "low_confidence" counts as incorrect.
        records["correct"] = [int(p >= 0 and int(p) == int(t))
                              for p, t in zip(final_class_num, all_labels)]

    df_out   = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, f"predictions_{timestamp}.csv")
    df_out.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Predictions saved : {csv_path}  ({len(df_out)} rows x {len(df_out.columns)} cols, output_mode={output_mode})")

    # ---- Prediction summary ------------------------------------------
    from collections import Counter
    assigned_names = [n for n in final_class_name if n != LOW_CONF_LABEL]
    class_counts_pred = Counter(assigned_names)
    n_assigned = len(assigned_names)

    summary = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "mode": "unlabeled" if is_unlabeled else "labeled",
        "output_mode": output_mode,
        "num_samples": int(N),
        "num_classes": int(C),
        "confidence_threshold": float(confidence_threshold) if use_threshold else None,
        "num_assigned": int(n_assigned),
        "num_low_confidence": int(n_low),
        "mean_top1_confidence": round(float(top1_conf.mean()), 6),
        "median_top1_confidence": round(float(np.median(top1_conf)), 6),
        "per_class_prediction_counts": dict(sorted(class_counts_pred.items(), key=lambda x: -x[1])),
        "class_map": {str(k): v for k, v in sorted(num2name.items())},
        "data_path": os.path.abspath(data_path),
        "checkpoint_path": os.path.abspath(checkpoint_path),
    }
    if has_labels:
        summary["top1_accuracy"] = round(float(top1_meter.avg), 6)
        summary["num_correct"] = int(sum(records["correct"]))

    # Machine-readable summary (API / downstream pipelines)
    json_path = os.path.join(out_dir, f"inference_summary_{timestamp}.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total samples       : {N}")
    if use_threshold:
        logger.info(f"  Threshold           : {confidence_threshold:.2f}")
        logger.info(f"  Assigned (≥ thresh) : {n_assigned}  ({100*n_assigned/N:.1f}%)")
        logger.info(f"  Low confidence      : {n_low}  ({100*n_low/N:.1f}%)")
    else:
        logger.info("  No threshold applied (top-1 always assigned)")
    logger.info(f"  Mean top-1 conf     : {summary['mean_top1_confidence']:.4f}")
    logger.info(f"  Median top-1 conf   : {summary['median_top1_confidence']:.4f}")

    # Per-class count of final predictions (excluding low_confidence)
    logger.info(f"\n  Per-class prediction counts ({len(class_counts_pred)} classes):")
    logger.info(f"  {'Class':<30}  {'Count':>7}  {'%':>6}")
    logger.info(f"  {'-'*46}")
    for cls_name, cnt in class_counts_pred.most_common():
        logger.info(f"  {cls_name:<30}  {cnt:>7}  {100*cnt/max(n_assigned,1):>5.1f}%")

    # ---- Labeled-mode metrics ----------------------------------------
    cm_pngs = []
    if has_labels:
        logger.info("\n" + "=" * 70)
        logger.info(f"  Top-1 Accuracy : {top1_meter.avg:.2f}%")

        class_counts_true = Counter(all_labels)
        cm, true_cls, pred_cls = get_sorted_confusion_matrix(
            top1_class.tolist(), all_labels, class_counts_true
        )
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = np.divide(cm.astype("float"), row_sums,
                             where=row_sums != 0,
                             out=np.zeros_like(cm, dtype="float"))
        plot_confusion_matrix(
            cm=cm, total_samples=N,
            true_classes=true_cls, pred_classes=pred_cls,
            class_names=num2name, accuracy=top1_meter.avg,
            save_path=out_dir, title="Absolute Values", normalize=False,
        )
        plot_confusion_matrix(
            cm=cm_norm, total_samples=N,
            true_classes=true_cls, pred_classes=pred_cls,
            class_names=num2name, accuracy=top1_meter.avg,
            save_path=out_dir, title="Normalized Values", normalize=True,
        )
        cm_pngs = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir))
                   if f.endswith(("_absolute.png", "_normalized.png"))]
        logger.info(f"  Confusion matrices saved to: {out_dir}")

    logger.info("\n" + "=" * 70)
    logger.info("OUTPUT FILES")
    logger.info("=" * 70)
    logger.info(f"  Predictions CSV  : {csv_path}")
    logger.info(f"  Summary JSON     : {json_path}")
    logger.info(f"  Run log          : {log_file}")
    for p in cm_pngs:
        logger.info(f"  Confusion matrix : {p}")
    logger.info(f"\nCSV column guide (output_mode={output_mode}):")
    if all_img_paths:
        logger.info("  image_path         – path to the first holographic image")
    logger.info("  predicted_class  – top-1 class name (or 'low_confidence' if below threshold)")
    logger.info("  probability      – confidence of the predicted class")
    if output_mode == "detailed":
        logger.info("  predicted_class_id – integer class id (-1 if low_confidence)")
        logger.info("  top2_class/prob    – 2nd most likely class and its probability")
        logger.info("  top3_class/prob    – 3rd most likely class and its probability")
        logger.info("  prob_<classname>   – full softmax probability for every class")
    logger.info("  sample_index     – 0-based row index (dataset order)")
    if has_labels:
        logger.info("  true_class         – ground-truth label (labeled mode only)")
        logger.info("  correct            – 1 if the final assigned class matches truth (labeled mode only)")
    logger.info("=" * 70)
    logger.info("Inference complete!")
    logger.info("=" * 70)

    # Machine-readable result (CLI, notebooks and API consumers)
    result = {
        "mode": summary["mode"],
        "output_mode": output_mode,
        "num_samples": int(N),
        "csv_path": csv_path,
        "summary_json_path": json_path,
        "log_path": log_file,
        "confusion_matrix_pngs": cm_pngs,
        "predictions": df_out,
        "class_map": num2name,
        "summary": summary,
    }

    torch.cuda.empty_cache()
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Legacy entry-point (config-file driven, kept for backward compatibility)
# ---------------------------------------------------------------------------

def resolve_test_checkpoint(config):
    """
    Resolve the checkpoint to evaluate in test mode.

    Priority:
      1. ``test.checkpoint_path`` (if set and the file exists)
      2. the most recent ``best_*.pth``, then ``last_*.pth``, under
         ``logging.checkpoint_dir``

    Raises:
        FileNotFoundError: with an actionable message when nothing resolves.
    """
    test_cfg = getattr(config, "test", None)
    path = None
    if test_cfg is not None:
        path = test_cfg.get("checkpoint_path", None) if hasattr(test_cfg, "get") else None
    if path and os.path.isfile(path):
        return path

    logging_cfg = getattr(config, "logging", None)
    cand_dir = None
    if logging_cfg is not None and hasattr(logging_cfg, "get"):
        cand_dir = logging_cfg.get("checkpoint_dir", None)
    if cand_dir and os.path.isdir(cand_dir):
        for pattern in ("best_*.pth", "last_*.pth"):
            matches = sorted(glob.glob(os.path.join(cand_dir, pattern)))
            if matches:
                return matches[-1]

    if path:
        raise FileNotFoundError(
            f"test.checkpoint_path is set but the file does not exist: {path}")
    raise FileNotFoundError(
        "No checkpoint resolved for test mode. Set 'test.checkpoint_path' in the "
        "config, or place a best_classification_model.pth / last_classification_model.pth "
        f"checkpoint in 'logging.checkpoint_dir' (currently {cand_dir!r})."
    )


def resolve_test_data_path(config):
    """
    Resolve the labeled HDF5 to evaluate in test mode.

    Priority: ``test.data_path`` → ``data.validation_path``
              → ``data.train_data_path``.

    Raises:
        FileNotFoundError: when no path resolves.
    """
    test_cfg = getattr(config, "test", None)
    path = None
    if test_cfg is not None and hasattr(test_cfg, "get"):
        path = test_cfg.get("data_path", None)
    if path:
        return path
    path = (config.data.get("validation_path", None)
            or config.data.get("train_data_path", None))
    if not path:
        raise FileNotFoundError(
            "No test data resolved. Set 'test.data_path' or 'data.validation_path' "
            "in the config.")
    return path


def _test_model_worker(rank, world_size, config, device_id=None):
    """
    Worker for config-driven test-mode evaluation on labeled HDF5 data.

    Single-process runs (world_size == 1) delegate to ``run_inference`` so
    test mode produces the full professional output set (predictions CSV,
    summary JSON, confusion matrices, run log) under
    ``<checkpoint_dir>/test_<timestamp>/``.

    Distributed runs (world_size > 1) keep the legacy metric-only evaluation
    loop (per-rank accuracy, no CSV).
    """
    if device_id is None:
        device_id = rank if torch.cuda.is_available() else -1
    device = torch.device(
        f"cuda:{device_id}" if torch.cuda.is_available() and device_id >= 0 else "cpu")

    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)

    checkpoint_path = resolve_test_checkpoint(config)
    data_path = resolve_test_data_path(config)
    batch_size, _b_src, num_workers, _w_src = resolve_inference_settings(config)
    num_workers = _clamp_num_workers(num_workers)

    if world_size == 1:
        # Unified professional output via the shared inference core.
        cfg_path = getattr(config, "config_path", None)
        if not cfg_path:
            raise FileNotFoundError(
                "config.config_path is not set — test mode needs the config file "
                "path to reconstruct the model (run via bioairmet-train --mode test).")
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(checkpoint_path)),
            f"test_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        result = run_inference(
            data_path=data_path,
            checkpoint_path=checkpoint_path,
            save_path=save_path,
            config_path=cfg_path,
            gpu_id=device_id,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        acc = result["summary"].get("top1_accuracy")
        if acc is not None:
            print(f"Test Acc@1: {acc * 100:.2f}%")
        print(f"Test results saved to: {save_path}")
        return

    # ---- Distributed (world_size > 1): legacy metric-only loop ----------
    cat_map_path = config.data.get("cat_map_path", None)
    try:
        head_num_classes = int(config.architecture_setup.classification_model.num_classes)
    except Exception:
        head_num_classes = None
    test_dataset = Stage2Dataset(
        hdf5_path=data_path,
        image_key=config.data.get("image_column_name", "images"),
        fluorescence_key=config.data.get("fluorescence_spectra_column_name", "relative_spectra"),
        label_key=config.data.get("category_number_column_name", "category_num"),
        img_reader_type=config.data.get("img_reader_type", "legacy"),
        img_aug_prob=0.0,
        fluo_aug_prob=0.0,
        cat_map_path=cat_map_path,
        num_classes=head_num_classes,
        **image_reader_kwargs(config.data),
    )

    sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) \
        if world_size > 1 else None

    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        # forkserver (Python ≥ 3.14 default) pickles the dataset, which holds
        # unpicklable threading.Lock / h5py handles — use fork explicitly.
        multiprocessing_context="fork" if num_workers > 0 else None,
    )

    config.architecture_setup.type = "classification"
    model = build_model_from_config(config).to(device)

    # Checkpoints store non-tensor objects (epoch, optimizer state, metrics),
    # so weights_only=False is intentionally required here.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = ensure_state_dict_metadata(strip_ddp_prefix(state_dict), model)
    model.load_state_dict(state_dict, strict=False)

    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank])

    model.eval()
    all_preds, all_labels = [], []
    top1 = AverageMeter("Acc@1", ":6.2f")

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Rank {rank}", disable=(rank != 0)):
            img1 = batch["image"][0].to(device, non_blocking=True)
            img2 = batch["image"][1].to(device, non_blocking=True)
            fluo = batch["fluorescence"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            outputs = model(img1, img2, fluo)
            acc1 = accuracy(outputs, labels, topk=(1,))[0]
            top1.update(acc1[0], img1.size(0))

            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if world_size > 1:
        dist.destroy_process_group()

    if rank == 0:
        print(f"Test Acc@1: {top1.avg:.2f}%")


def main_worker_test(config_path):
    """
    Legacy entry point: parse config and launch _test_model_worker.
    """
    config = parse_config(config_path)
    config.config_path = os.path.abspath(config_path)

    world_size = torch.cuda.device_count() if config.distributed.world_size == -1 \
        else config.distributed.world_size
    world_size = max(world_size, 1)

    os.environ["MASTER_ADDR"] = config.distributed.master_addr
    os.environ["MASTER_PORT"] = config.distributed.master_port

    if world_size > 1:
        torch.multiprocessing.spawn(
            _test_model_worker, args=(world_size, config), nprocs=world_size, join=True
        )
    else:
        _test_model_worker(0, 1, config)


# ---------------------------------------------------------------------------
# CLI  (used by run_inference.sh)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BioAirMet Inference – unlabeled or labeled data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path",       type=str, required=True,
                        help="Raw event directory (unlabeled) or HDF5 file (labeled)")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to classification model *.pth checkpoint")
    parser.add_argument("--save_path",       type=str, required=True,
                        help="Directory to save predictions and plots")
    parser.add_argument("--config_path",     type=str, required=True,
                        help="YAML config used to train the model")
    parser.add_argument("--gpu_id",          type=int, default=0,
                        help="CUDA device ID (-1 for CPU)")
    parser.add_argument("--batch_size",      type=int, default=64)
    parser.add_argument("--num_workers",     type=int, default=6)
    parser.add_argument("--img_reader_type", type=str, default="legacy",
                        choices=["legacy", "minmax", "global", "none"],
                        help="Image reader/normalisation; MUST match the one the model was "
                             "trained with. 'legacy' = 16-bit -> per-image min-max -> 8-bit "
                             "PIL -> [0,1]; 'minmax' = per-image min-max on the float tensor; "
                             "'global' = divide by the file's full scale (65535/255). "
                             "'none' is an alias of 'global'.")
    parser.add_argument("--confidence_threshold", type=float, default=0.0,
                        help="Min top-1 probability to assign a label. 0.0 = disabled (always top-1).")
    parser.add_argument("--min_intensity_delta", type=float, default=0.1,
                        help="Minimum intensity delta for valid spectra (unlabeled mode)")
    parser.add_argument("--min_area", type=int, default=500,
                        help="Minimum area for valid spectra (unlabeled mode)")
    parser.add_argument("--min_solidity", type=float, default=0.7,
                        help="Minimum solidity for valid spectra (unlabeled mode)")
    parser.add_argument("--skip_validation", action="store_true",
                        help="Skip the per-event quality checks (area / solidity / "
                             "intensity delta) in unlabeled raw-directory mode — "
                             "keeps every event with usable image files")
    parser.add_argument("--output_mode", type=str, default="compact",
                        choices=["compact", "detailed"],
                        help="Predictions CSV detail level: 'compact' (image_path, "
                             "predicted_class, probability) or 'detailed' (adds class id, "
                             "top-2/top-3 and the full per-class probability distribution)")

    args = parser.parse_args()

    run_inference(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        save_path=args.save_path,
        config_path=args.config_path,
        gpu_id=args.gpu_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_reader_type=args.img_reader_type,
        confidence_threshold=args.confidence_threshold,
        min_intensity_delta=args.min_intensity_delta,
        min_area=args.min_area,
        min_solidity=args.min_solidity,
        skip_validation=args.skip_validation,
        output_mode=args.output_mode
    )