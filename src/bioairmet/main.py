'''
@file    :   main.py
@create date : 2025-07-22 16:17:49
@modify date 2026-05-27 13:47:31
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    Core training orchestration for BioAirMet.
    This module provides main() and _worker_fn() for launching SSL pre-training,
    supervised classification fine-tuning, or model evaluation. It is used both
    by the CLI entry points (bioairmet-train, bioairmet-eval) and by the
    scripts/main.py convenience wrapper.
    ]
'''

import argparse
import os
import sys
import datetime
import warnings

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

# Suppress specific UserWarnings from torchvision models
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.models._utils')

from bioairmet.utils import parse_config
from bioairmet.utils.config_parser import (
    load_and_merge_architecture_config,
    format_resolved_yaml,
    find_placeholder_paths,
    apply_overrides,
    validate_config,
    ConfigError,
)
from bioairmet.training import _test_model_worker
from bioairmet.trainers import train_ssl_worker_v2, train_supervised_worker_v2
from easydict import EasyDict as edict


def main():
    """Entry point for ``python -m bioairmet.main`` (and legacy callers)."""
    parser = argparse.ArgumentParser(description="BioAirMet Training and Testing Script")
    parser.add_argument(
        '--config_path', type=str, required=True,
        help='Path to the YAML configuration file'
    )
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['ssl', 'classification', 'test'],
        help='Operation mode: "ssl" for SSL pre-training, "classification" for fine-tuning, "test" for evaluation.'
    )
    parser.add_argument(
        '--show_config', action='store_true',
        help='Print the fully resolved configuration (architecture merged in) and exit — no training, no side effects'
    )
    parser.add_argument(
        '--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
        help="Override any config value, e.g. --set train.epochs=5 (repeatable; "
             "values are parsed as YAML)"
    )
    args, _ = parser.parse_known_args()
    run_training(args.config_path, args.mode, show_config=args.show_config,
                 overrides=args.overrides)


def run_training(config_path, mode, show_config=False, overrides=None):
    """
    Resolve the full configuration for a run and launch the training/test worker.

    Steps:
      1. Parse the stage config YAML.
      2. Set ``architecture_setup.type = mode`` BEFORE merging (the merge
         branches on it).
      3. Resolve + merge the architecture config — fails fast with an
         actionable error if it cannot be resolved.
      4. Apply command-line ``--set KEY=VALUE`` overrides (they win over
         everything in the file).
      5. Optionally print the resolved config (``show_config``) and stop.
      6. Pre-flight checks (training modes): unfilled CHANGE-ME placeholders,
         config shape (validate_config), and GPU requests vs visible GPUs —
         all fail fast, before any directory or process is created.
      7. Create the experiment directory and spawn the worker process(es).
    """
    # Ensure the config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    print(f"Loading configuration from: {config_path}")
    config = parse_config(config_path)
    config.config_path = os.path.abspath(config_path)

    # Set the mode BEFORE merging: the architecture merge branches on
    # architecture_setup.type ('ssl' vs 'classification').
    arch_setup = edict(getattr(config, 'architecture_setup', None) or {})
    arch_setup.type = mode
    config.architecture_setup = arch_setup

    # Resolve + merge the architecture config (fail fast with an actionable error).
    config, arch_path = load_and_merge_architecture_config(config)
    print(f"Architecture config resolved from: {arch_path}")
    print("Using V2 trainers (enhanced, class-based implementation)")

    # Apply command-line overrides (after merging, so they win over the file).
    if overrides:
        apply_overrides(config, overrides)
        for item in overrides:
            print(f"Override applied: {item}")

    if show_config:
        _print_resolved_config(config, mode)
        return

    # --- Pre-flight checks (fail fast, no side effects yet) -----------------
    placeholders = find_placeholder_paths(config)
    if placeholders:
        listing = "\n".join(f"     - {d}  =  {v!r}" for d, v in placeholders)
        raise ConfigError(
            "The config still contains unfilled CHANGE-ME placeholders:\n"
            f"{listing}\n"
            "Fix them in the config file (or override with --set KEY=VALUE), "
            "or re-run with --show_config to inspect the full resolved config."
        )

    if mode in ("ssl", "classification"):
        validate_config(config, mode)
        _check_visible_gpus(config)

    # Generate unique experiment directory name
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    project_name = (getattr(config, "project_name", None) or "").strip()
    if project_name:
        experiment_name = f"{current_time}_{project_name}"
    else:
        experiment_name = f"{current_time}_BioAirMet_experiment"

    config.logging.log_dir = os.path.join(config.logging.checkpoint_dir, experiment_name)
    config.logging.checkpoint_dir = os.path.join(config.logging.checkpoint_dir, experiment_name)

    # Determine world_size and GPU IDs
    if config.distributed.get('gpu_ids') and len(config.distributed.gpu_ids) > 0:
        gpu_ids = config.distributed.gpu_ids
        world_size = len(gpu_ids)
        print(f"Using specified GPUs: {gpu_ids}")
    elif config.distributed.world_size == -1:
        world_size = torch.cuda.device_count()
        if world_size == 0:
            world_size = 1
            print("No GPUs found, falling back to single CPU training.")
        gpu_ids = list(range(world_size))
    else:
        world_size = config.distributed.world_size
        gpu_ids = list(range(world_size))

    os.makedirs(config.logging.checkpoint_dir, exist_ok=True)
    os.makedirs(config.logging.log_dir, exist_ok=True)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"Detected distributed environment: Rank {rank}/{world_size}")
        _worker_fn(rank, world_size, config, mode, gpu_ids)
    else:
        os.environ['MASTER_ADDR'] = config.distributed.master_addr
        os.environ['MASTER_PORT'] = config.distributed.master_port

        if config.distributed.enable and world_size > 1:
            print(f"Using {world_size} GPUs for distributed training (spawning processes).")
            mp.spawn(
                _worker_fn,
                nprocs=world_size,
                args=(world_size, config, mode, gpu_ids),
                join=True
            )
        else:
            print("Using single GPU training.")
            _worker_fn(0, 1, config, mode, gpu_ids)


def _print_resolved_config(config, mode):
    """
    Print the fully resolved configuration (architecture merged in) plus a
    pre-flight checklist of values that still look like unfilled placeholders.
    """
    print()
    print("=" * 70)
    print(f"  RESOLVED CONFIGURATION — mode: {mode}")
    print(f"  (architecture merged from: "
          f"{getattr(getattr(config, 'architecture_setup', None), 'architecture_source', 'unknown')})")
    print("=" * 70)
    print(format_resolved_yaml(config))
    print("=" * 70)
    placeholders = find_placeholder_paths(config)
    if placeholders:
        print("  !  Unfilled values (CHANGE ME) — update these before starting training:")
        for dotted, value in placeholders:
            print(f"     - {dotted}  =  {value!r}")
    else:
        print("  OK  No placeholder values detected — the config looks complete.")
    print("=" * 70)


def _check_visible_gpus(config):
    """Cross-check the requested GPUs against the GPUs actually visible.

    The launcher scripts control visibility via CUDA_VISIBLE_DEVICES (their
    GPU_IDS variable); requesting more GPUs than are visible fails fast here
    instead of dying inside the distributed worker with an opaque NCCL error.
    No-op when no CUDA device is available (CPU-only runs).
    """
    if not torch.cuda.is_available():
        return
    n_visible = torch.cuda.device_count()
    dist_cfg = config.get("distributed") or {}
    world_size = dist_cfg.get("world_size", 1)
    gpu_ids = dist_cfg.get("gpu_ids") or []

    if isinstance(world_size, int) and world_size > 0 and world_size > n_visible:
        raise ConfigError(
            f"distributed.world_size={world_size} but only {n_visible} GPU(s) are visible. "
            "Check the GPU_IDS variable in the launcher script (it sets CUDA_VISIBLE_DEVICES)."
        )
    for g in gpu_ids:
        if isinstance(g, int) and g >= n_visible:
            raise ConfigError(
                f"distributed.gpu_ids={gpu_ids} references GPU index {g}, but only "
                f"{n_visible} GPU(s) are visible. Check the GPU_IDS variable in the "
                "launcher script (it sets CUDA_VISIBLE_DEVICES); gpu_ids are indices "
                "INTO the visible GPUs."
            )


def _worker_fn(rank, world_size, config, mode, gpu_ids):
    """
    Worker function for distributed training.

    Args:
        rank: Process rank
        world_size: Total number of processes
        config: Configuration object
        mode: Training mode ('ssl', 'classification', or 'test')
        gpu_ids: List of GPU IDs to use
    """
    if "LOCAL_RANK" in os.environ:
        device_id = int(os.environ["LOCAL_RANK"])
    else:
        device_id = gpu_ids[rank]

    print(f"Rank {rank}: _worker_fn entered. Using device_id: {device_id}")

    try:
        if world_size > 1 and not dist.is_initialized():
            print(f"Rank {rank}: Initializing process group with NCCL backend...")
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(device_id)
            print(f"Rank {rank}: Process group initialized, device set to GPU {device_id}.")
        elif world_size == 1:
            if torch.cuda.is_available():
                torch.cuda.set_device(device_id)
                print(f"Rank {rank}: Single GPU training, device set to GPU {device_id}.")
            else:
                print(f"Rank {rank}: Single CPU training.")

        if mode == 'ssl':
            print(f"Rank {rank}: Starting Self-Supervised Learning (SSL) pre-training...")
            train_ssl_worker_v2(rank, world_size, config, device_id)
        elif mode == 'classification':
            print(f"Rank {rank}: Starting Supervised Classification fine-tuning...")
            train_supervised_worker_v2(rank, world_size, config, device_id)
        elif mode == 'test':
            print(f"Rank {rank}: Starting model evaluation on test set...")
            _test_model_worker(rank, world_size, config, device_id)

    except KeyboardInterrupt:
        if rank == 0:
            print(f"\n[Rank {rank}] KeyboardInterrupt: Training interrupted by user.")
    except Exception as e:
        print(f"Rank {rank}: Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        if world_size > 1 and dist.is_initialized():
            print(f"Rank {rank}: Cleaning up process group...")
            dist.destroy_process_group()
            print(f"Rank {rank}: Process group destroyed.")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
