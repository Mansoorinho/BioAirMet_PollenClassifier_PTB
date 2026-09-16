'''
@file    :   early_stopping_v2.py
@create date : 2026-01-12 14:10:33
@modify date 2026-02-19 10:18:37
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    This utility script defines the EarlyStoppingV2 class, an enhanced version of early stopping designed for the V2 training workflows.
    Key features include:
    - Support for both 'min' mode (e.g., minimizing validation loss) and 'max' mode (e.g., maximizing validation accuracy).
    - Rank-awareness to ensure logging only occurs on the primary process in distributed (DDP) environments.
    - A simplified interface that decouples checkpoint saving from the stopping logic, delegating file I/O to the V2 trainer classes.
    ]
'''

import numpy as np
import torch
import os
from typing import Optional, Callable


class EarlyStoppingV2:
    """
    Enhanced early stopping that supports both minimization (loss) and maximization (accuracy).
    
    IMPROVEMENTS over legacy EarlyStopping:
    - Supports mode='min' (for loss) or mode='max' (for accuracy)
    - Simpler interface (doesn't require all training state)
    - Works with V2 trainer checkpoint saving
    - Cleaner code, easier to understand
    
    Usage:
        # For SSL (minimize loss)
        early_stop = EarlyStoppingV2(patience=10, mode='min')
        
        # For classification (maximize accuracy)
        early_stop = EarlyStoppingV2(patience=10, mode='max')
        
        # In training loop
        if early_stop(current_metric, model):
            break  # Stop training
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = False,
        path: Optional[str] = None,
        trace_func: Callable = print,
        rank: int = 0
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' to minimize metric (loss), 'max' to maximize metric (accuracy)
            verbose: Whether to print messages
            path: Path to save best checkpoint (not used in V2 - handled by trainer)
            trace_func: Function to use for logging
            rank: Process rank (only rank 0 prints)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.path = path
        self.trace_func = trace_func
        self.rank = rank
        
        # State
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        
        # Set comparison function based on mode
        if mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
            self.best_score = float('inf')
        elif mode == 'max':
            self.is_better = lambda new, best: new > best + min_delta
            self.best_score = float('-inf')
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got: {mode}")
    
    def __call__(self, metric: float, model: Optional[torch.nn.Module] = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            metric: Current metric value (loss or accuracy)
            model: Model (not used in V2, kept for compatibility)
            
        Returns:
            True if training should stop, False otherwise
        """
        # Check if this is an improvement
        if self.best_score is None:
            # First call - initialize
            self.best_score = metric
            if self.rank == 0 and self.verbose:
                self.trace_func(f"Initial best {self.mode} metric: {metric:.6f}")
            return False
        
        if self.is_better(metric, self.best_score):
            # Improvement found
            if self.rank == 0 and self.verbose:
                direction = "decreased" if self.mode == 'min' else "increased"
                self.trace_func(
                    f"Metric {direction}: {self.best_score:.6f} → {metric:.6f}"
                )
            self.best_score = metric
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.rank == 0:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.rank == 0:
                    self.trace_func("Early stopping triggered!")
                return True
            
            return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.early_stop = False
