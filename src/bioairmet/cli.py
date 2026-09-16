'''
@file    :   cli.py
@create date : 2026-01-25 09:15:20
@modify date 2026-06-23 14:42:19
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    Command-Line Interface (CLI) entry points for the BioAirMet package.
    - bioairmet-train     : SSL pre-training and supervised fine-tuning.
    - bioairmet-inference : Model inference — given data, produce predictions CSV.
                            All model/data settings are read from the experiment config.yaml.
    - bioairmet-validate  : V2 validation pipeline with calibration metrics.
    ]
'''

import argparse

from bioairmet.main import run_training


def main_train():
    """Entry point for SSL and classification training.

    V2 trainers are always used (legacy V1 trainers have been removed).
    """
    parser = argparse.ArgumentParser(
        description="BioAirMet Training - SSL pretraining or Classification fine-tuning"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ssl", "classification"],
        required=True,
        help="Training mode: 'ssl' for pretraining, 'classification' for fine-tuning"
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        help="Print the fully resolved configuration (architecture merged in) "
             "with a CHANGE-ME checklist, then exit — no training, no side effects"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config value, e.g. --set train.epochs=5 "
             "--set data.val_split_ratio=0.05 (repeatable; values are parsed as YAML)"
    )

    args = parser.parse_args()

    # Direct call into the training orchestrator (no sys.argv reconstruction).
    run_training(args.config_path, args.mode, show_config=args.show_config,
                 overrides=args.overrides)


def main_inference():
    """
    Entry point for model inference.

    Given an experiment directory and input data, runs the trained classification
    model and saves a predictions CSV (plus accuracy/confusion matrices when the
    input is a labeled HDF5 file).

    All model settings — image normalisation strategy, batch size, number of
    DataLoader workers — are read automatically from the experiment's config.yaml.
    The user only needs to supply: experiment directory, data path, save path,
    confidence threshold, and GPU device.
    """
    parser = argparse.ArgumentParser(
        prog="bioairmet-inference",
        description=(
            "BioAirMet Inference — run a trained classification model and save predictions.\n\n"
            "Given an experiment directory and input data, produces:\n"
            "  • predictions_<timestamp>.csv  — per-sample predictions (compact by default;\n"
            "                                   --output_mode detailed adds the full per-class\n"
            "                                   probability distribution)\n"
            "  • inference_summary_<timestamp>.json — machine-readable run summary\n"
            "  • inference_<timestamp>.log    — detailed run log\n"
            "  • absolute_values_absolute.png + normalized_values_normalized.png (labeled data only)\n\n"
            "All model settings (image normalisation, batch size, num workers) are read\n"
            "automatically from the experiment config.yaml — no manual tuning required."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Unlabeled raw event directory → predictions CSV only\n"
            "  bioairmet-inference \\\n"
            "      --experiment_dir /path/to/experiment \\\n"
            "      --data_path /path/to/raw_events \\\n"
            "      --save_path ./predictions\n\n"
            "  # Labeled HDF5 → predictions + accuracy + confusion matrices\n"
            "  bioairmet-inference \\\n"
            "      --experiment_dir /path/to/experiment \\\n"
            "      --data_path /path/to/dataset.h5 \\\n"
            "      --save_path ./predictions \\\n"
            "      --confidence_threshold 0.85\n"
        ),
    )

    # --- Required arguments -----------------------------------------------
    req = parser.add_argument_group("required")
    req.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        metavar="DIR",
        help=(
            "Path to the experiment directory produced by bioairmet-train. "
            "Must contain config.yaml and a checkpoint (best_*.pth or last_*.pth)."
        ),
    )
    req.add_argument(
        "--data_path",
        type=str,
        required=True,
        metavar="PATH",
        help=(
            "Input data — either: "
            "(1) an HDF5 file (labeled) → predictions + accuracy + confusion matrices, or "
            "(2) a raw event directory (unlabeled) → predictions CSV only."
        ),
    )
    req.add_argument(
        "--save_path",
        type=str,
        required=True,
        metavar="DIR",
        help="Output directory. Created automatically if it does not exist.",
    )

    # --- Optional arguments -----------------------------------------------
    opt = parser.add_argument_group("optional")
    opt.add_argument(
        "--checkpoint_type",
        type=str,
        choices=["best", "last"],
        default="best",
        help="Checkpoint to load: 'best' (highest val accuracy) or 'last' (final epoch). Default: best",
    )
    opt.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="CUDA device ID. Use -1 to run on CPU. Default: 0",
    )
    opt.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=(
            "Override batch size from the experiment config.yaml. "
            "Omit to use the value stored in the experiment config."
        ),
    )
    opt.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum top-1 softmax probability required to assign a class label. "
            "Samples below this score are labelled 'low_confidence' in the output CSV. "
            "0.0 (default) disables thresholding — every sample always gets a top-1 prediction."
        ),
    )
    opt.add_argument(
        "--output_mode",
        type=str,
        choices=["compact", "detailed"],
        default="compact",
        help=(
            "Predictions CSV detail level. 'compact' (default): image_path, "
            "predicted_class, probability. 'detailed': compact + predicted_class_id, "
            "top-2/top-3 alternatives and the full per-class probability distribution. "
            "(labeled mode always adds true_class, correct)"
        ),
    )
    opt.add_argument(
        "--min_intensity_delta",
        type=float,
        default=0.1,
        help="Minimum intensity delta for valid spectra (unlabeled mode)",
    )
    opt.add_argument(
        "--min_area",
        type=int,
        default=500,
        help="Minimum area for valid spectra (unlabeled mode)",
    )
    opt.add_argument(
        "--min_solidity",
        type=float,
        default=0.7,
        help="Minimum solidity for valid spectra (unlabeled mode)",
    )

    opt.add_argument(
        "--skip_validation",
        action="store_true",
        help=(
            "Skip the per-event quality checks (min area / solidity / intensity delta) "
            "in unlabeled raw-directory mode — keeps every event with usable image "
            "files instead of dropping the ones that fail the checks."
        ),
    )

    args = parser.parse_args()

    import os
    import glob

    experiment_dir = args.experiment_dir

    # --- Resolve config.yaml ----------------------------------------------
    config_path = os.path.join(experiment_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"config.yaml not found in experiment directory: {experiment_dir}"
        )

    # --- Resolve checkpoint -----------------------------------------------
    matches = sorted(glob.glob(os.path.join(experiment_dir, f"{args.checkpoint_type}_*.pth")))
    if not matches:
        available = glob.glob(os.path.join(experiment_dir, "*.pth"))
        raise FileNotFoundError(
            f"No '{args.checkpoint_type}_*.pth' checkpoint found in: {experiment_dir}\n"
            f"Available checkpoints: {[os.path.basename(p) for p in available]}"
        )
    checkpoint_path = matches[-1]   # most recent if multiple

    # --- Read model/data settings from config (overridable on the CLI) ---
    from bioairmet.utils.config_parser import (
        parse_config as _parse_config,
        resolve_inference_settings,
    )
    cfg = _parse_config(config_path)
    img_reader_type = cfg.data.get("img_reader_type", "legacy")
    _bs, _bs_src, _nw, _nw_src = resolve_inference_settings(cfg)
    batch_size = args.batch_size if args.batch_size is not None else _bs
    batch_source = "CLI --batch_size" if args.batch_size is not None else _bs_src
    num_workers = _nw
    num_workers_source = _nw_src

    # --- Pre-run summary --------------------------------------------------
    is_unlabeled = os.path.isdir(args.data_path)
    mode_str = (
        "UNLABELED  (raw directory → predictions CSV)"
        if is_unlabeled
        else "LABELED    (HDF5 → predictions + accuracy + confusion matrices)"
    )
    thresh_str = (
        f"{args.confidence_threshold:.2f}"
        if args.confidence_threshold > 0
        else "disabled (top-1 always assigned)"
    )

    print()
    print("=" * 68)
    print("  BioAirMet  —  Inference")
    print("=" * 68)
    print(f"  Experiment dir    : {experiment_dir}")
    print(f"  Checkpoint        : {os.path.basename(checkpoint_path)}")
    print(f"  Config            : {config_path}")
    print(f"  Data              : {args.data_path}")
    print(f"  Mode              : {mode_str}")
    print(f"  Save to           : {args.save_path}")
    print(f"  Device            : {'cpu' if args.gpu_id < 0 else f'cuda:{args.gpu_id}'}")
    print(f"  Confidence thr.   : {thresh_str}")
    print("  — from config.yaml —")
    print(f"  img_reader_type   : {img_reader_type}")
    print(f"  batch_size        : {batch_size}  ({batch_source})")
    print(f"  num_workers       : {num_workers}  ({num_workers_source}; capped at the CPU count if higher)")
    if args.skip_validation:
        print("  Quality checks  : SKIPPED (--skip_validation)")
    print("=" * 68)
    print()

    from bioairmet.training.inference import run_inference

    run_inference(
        data_path=args.data_path,
        checkpoint_path=checkpoint_path,
        save_path=args.save_path,
        config_path=config_path,
        gpu_id=args.gpu_id,
        batch_size=batch_size,
        num_workers=num_workers,
        img_reader_type=img_reader_type,
        confidence_threshold=args.confidence_threshold,
        min_intensity_delta=args.min_intensity_delta,
        min_area=args.min_area,
        min_solidity=args.min_solidity,
        skip_validation=args.skip_validation,
        output_mode=args.output_mode,
    )


def main_validate():
    """Entry point for V2 validation script."""
    parser = argparse.ArgumentParser(
        description="BioAirMet Validation - Validate classification model with calibration metrics"
    )
    parser.add_argument(
        "--experiment_dir", "--checkpoint_dir",
        dest="experiment_dir",
        type=str,
        required=True,
        help="Path to experiment directory containing the checkpoint and config.yaml "
             "(--checkpoint_dir is a deprecated alias)"
    )
    parser.add_argument(
        "--checkpoint_type",
        type=str,
        choices=["best", "last"],
        default="best",
        help="Which checkpoint to use: 'best' or 'last' (default: best)"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU device ID, -1 for CPU (default: 0)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to validation dataset (HDF5 file)"
    )
    parser.add_argument(
        "--save_path", "--plot_path",
        dest="save_path",
        type=str,
        default="./validation_results",
        help="Directory to save validation results (default: ./validation_results; "
             "--plot_path is a deprecated alias)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config (optional)"
    )

    args = parser.parse_args()

    from bioairmet.utils.validate_classification_v2 import validate_classification_model_v2

    validate_classification_model_v2(
        experiment_dir=args.experiment_dir,
        checkpoint_type=args.checkpoint_type,
        gpu_id=args.gpu_id,
        data_path=args.data_path,
        save_path=args.save_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    print("BioAirMet CLI Tools")
    print("\nAvailable commands:")
    print("  bioairmet-train     - Train SSL or classification models")
    print("  bioairmet-inference - Run inference / produce predictions CSV")
    print("  bioairmet-validate  - Validate with calibration metrics")
