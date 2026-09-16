'''
@file    :   reproducibility.py
@create date : 2025-05-15 10:15:20
@modify date 2026-02-19 10:26:55
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    This utility script provides functionality to ensure reproducibility across different training runs.
    It sets consistent seeds for Python's built-in random module, NumPy, and PyTorch (including CUDA components).
    Additionally, it configures CuDNN to operate in a deterministic mode and disables automatic benchmarking to guarantee identical results on the same hardware.
    ]
'''
import random
import numpy as np
import torch
import os

def set_seed(seed: int):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    
    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed set to {seed} for random, numpy, torch, and cudnn.")
