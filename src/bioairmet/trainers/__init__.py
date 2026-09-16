"""
Enhanced trainer classes for cleaner, more maintainable training code.

V2 trainers break down monolithic training loops into reusable, testable components.
"""

from .ssl_trainer_v2 import SSLTrainerV2, train_ssl_worker_v2
from .classification_trainer_v2 import ClassificationTrainerV2, train_supervised_worker_v2

__all__ = [
    'SSLTrainerV2',
    'ClassificationTrainerV2',
    'train_ssl_worker_v2',
    'train_supervised_worker_v2',
]
