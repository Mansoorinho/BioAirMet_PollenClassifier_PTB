"""
BioAirMet: Two-Stage Pollen Classification Framework

A PyTorch-based deep learning library for pollen classification using
holographic images and fluorescence spectra.

Two-stage training pipeline:
1. Self-supervised pretraining (SSL) with CLIP-style contrastive learning
2. Supervised fine-tuning for classification

Example:
    >>> from bioairmet.models import build_model_from_config
    >>> from bioairmet.utils import parse_config
    >>> 
    >>> config = parse_config('config.yaml')
    >>> model = build_model_from_config(config)
"""

__version__ = "1.0.0"
__author__ = "Nabawi, M., Martin, J., Bilson, S., Kadantsev, E., Zeder, Y., Schwendimann, A., Crouzy, B., Erb, S., Haufe, S., & Klein, T."
__email__ = "mansoor.nabawi@gmail.com"
__email__ = "Tobias.Klein@ptb.de"


# Core imports
from . import config
from . import data
from . import models
from . import training
from . import trainers
from . import utils

# Convenience imports
from .models import (
    build_model_from_config,
    load_model,
    build_ssl_model_from_config,
    build_classification_model_from_config_v2,
    HoloClassifierV2,
)

from .data import (
    build_dataset_from_config,
    Stage2Dataset,
    ValidationDataset_Unlabeled,
)

from .utils import (
    parse_config,
    setup_logger,
    setup_model_logger,
    build_optimizer_from_config,
    build_scheduler_from_config,
    build_ssl_loss_from_config,
    build_classification_loss_from_config,
)

from .trainers import (
    SSLTrainerV2,
    ClassificationTrainerV2,
    train_ssl_worker_v2,
    train_supervised_worker_v2,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Modules
    "config",
    "data",
    "models",
    "training",
    "trainers",
    "utils",
    
    # Models
    "build_model_from_config",
    "load_model",
    "build_ssl_model_from_config",
    "build_classification_model_from_config_v2",
    "HoloClassifierV2",
    
    # Data
    "build_dataset_from_config",
    "Stage2Dataset",
    "ValidationDataset_Unlabeled",
    
    # Utils
    "parse_config",
    "setup_logger",
    "setup_model_logger",
    "build_optimizer_from_config",
    "build_scheduler_from_config",
    "build_ssl_loss_from_config",
    "build_classification_loss_from_config",
    
    # Trainers
    "SSLTrainerV2",
    "ClassificationTrainerV2",
    "train_ssl_worker_v2",
    "train_supervised_worker_v2",
]
