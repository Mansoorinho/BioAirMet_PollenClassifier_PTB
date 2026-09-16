'''
@file    :   logger.py
@create date : 2025-05-11 14:15:20
@modify date 2026-02-19 10:24:24
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    This utility script provides comprehensive logging functionality for the BioAirMet project. 
    It supports multiple logging backends and formats:
    - Standard Python logging for console and timestamped file outputs.
    - Dedicated model logging for recording architectures and parameters in 'training.log'.
    - CSV-formatted metric logging ('metrics_log.txt') for programmatic training progress tracking.
    - TensorBoard integration via the TensorboardLogger wrapper for visual analysis of losses and metrics.
    All logging functions are rank-aware, ensuring that file I/O operations only occur on the primary process during distributed (DDP) training to prevent file corruption and redundant output.
    ]
'''

import logging
import os
from torch.utils.tensorboard.writer import SummaryWriter
import datetime

def setup_logger(log_dir, rank=0, name='trainer'):
    """
    Sets up a logger for training.
    Args:
        log_dir (str): Directory to save log files.
        rank (int): Rank of the current process in distributed training. Only rank 0 logs to file.
        name (str): Name of the logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_rank{rank}.log')

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False # Prevent messages from being passed to the root logger

    # Clear existing handlers to avoid duplicate logs in re-runs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler for all ranks
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    # File handler for rank 0 only
    if rank == 0:
        # Handler for the timestamped log file
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    return logger

def log_metrics_to_file(log_dir, epoch, train_loss, eval_loss, lr, accuracy=None, header_written=False):
    """
    Logs training and evaluation metrics to a 'log.txt' file in a CSV format.
    If the file is new, it writes the header.

    Args:
        log_dir (str): Directory where the log.txt file is located.
        epoch (int): Current epoch number.
        train_loss (float): Training loss for the current epoch.
        eval_loss (float): Evaluation loss for the current epoch.
        lr (float): Learning rate for the current epoch.
        accuracy (float, optional): Accuracy for classification tasks. Defaults to None.
        header_written (bool): Flag to indicate if the header has already been written.
    """
    log_file_path = os.path.join(log_dir, 'metrics_log.txt')
    
    mode = 'a' # Append mode
    if not os.path.exists(log_file_path) or not header_written:
        mode = 'w' # Write mode to create/overwrite and add header

    with open(log_file_path, mode) as f:
        if mode == 'w':
            if accuracy is not None:
                f.write("Epoch,Train Loss,Eval Loss,Learning Rate,Accuracy\n")
            else:
                f.write("Epoch,Train Loss,Eval Loss,Learning Rate\n")
        
        if accuracy is not None:
            f.write(f"{epoch},{train_loss:.6f},{eval_loss:.6f},{lr:.6f},{accuracy:.6f}\n")
        else:
            f.write(f"{epoch},{train_loss:.6f},{eval_loss:.6f},{lr:.6f}\n")

def setup_model_logger(log_dir, rank=0, name='model_logger'):
    """
    Sets up a dedicated logger for model-specific information (architecture, parameters, etc.).
    Args:
        log_dir (str): Directory to save log files.
        rank (int): Rank of the current process in distributed training. Only rank 0 logs to file.
        name (str): Name of the logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    model_log_file = os.path.join(log_dir, 'training.log')

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers to avoid duplicate logs in re-runs
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler for rank 0 only
    if rank == 0:
        file_handler = logging.FileHandler(model_log_file, mode='w') # Overwrite each run
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    return logger


class TensorboardLogger:
    """
    A wrapper for TensorBoard SummaryWriter.
    """
    def __init__(self, log_dir):
        """
        Initializes the TensorboardLogger.

        Args:
            log_dir (str): Directory where TensorBoard log files will be saved.
        """
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag, value, step):
        """
        Logs a scalar value to TensorBoard.

        Args:
            tag (str): Data identifier.
            value (float): Value to save.
            step (int): Global step value to record.
        """
        self.writer.add_scalar(tag, value, step)

    def log_image(self, tag, img_tensor, step):
        """
        Logs an image to TensorBoard.

        Args:
            tag (str): Data identifier.
            img_tensor (torch.Tensor): Image data.
            step (int): Global step value to record.
        """
        self.writer.add_image(tag, img_tensor, step)

    def log_histogram(self, tag, values, step):
        """
        Logs a histogram to TensorBoard.

        Args:
            tag (str): Data identifier.
            values (torch.Tensor or numpy.array): Values to build histogram.
            step (int): Global step value to record.
        """
        self.writer.add_histogram(tag, values, step)

    def close(self):
        """
        Closes the TensorBoard SummaryWriter.
        """
        self.writer.close()
