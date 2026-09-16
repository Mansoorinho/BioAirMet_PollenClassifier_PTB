'''
@file    :   metrics.py
@create date : 2025-05-18 13:05:36
@modify date 2026-02-19 10:25:43
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    This utility script defines classes and functions for performance measurement and tracking in the BioAirMet project.
    - AverageMeter: Computes and stores the current value, sum, count, and running average of any metric (e.g., loss, accuracy).
    - ProgressMeter: Handles the formatting and display of training/validation progress across batches.
    - accuracy: A utility function to compute Top-K accuracy metrics for classification tasks.
    ]
'''
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        """
        Initializes the AverageMeter.

        Args:
            name (str): Name of the metric.
            fmt (str): Format string for printing the value.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """
        Resets all internal counters to zero.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The current value.
            n (int): The number of samples associated with the current value (e.g., batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        # Safety: avoid division by zero
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self):
        """
        Returns a formatted string representation of the meter's current and average values.
        """
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    """
    Displays the progress of training/validation, including batch number and various metrics.
    """
    def __init__(self, num_batches, meters, prefix=""):
        """
        Initializes the ProgressMeter.

        Args:
            num_batches (int): Total number of batches in the current epoch/phase.
            meters (list): A list of AverageMeter objects to display.
            prefix (str): A prefix string to display before the batch information (e.g., "Epoch: [X]").
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        """
        Prints the current progress, including batch number and all registered meters.

        Args:
            batch (int): The current batch index.
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        """
        Generates the format string for displaying batch numbers.

        Args:
            num_batches (int): Total number of batches.

        Returns:
            str: The format string for batch display.
        """
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output (torch.Tensor): Model predictions (logits or probabilities).
        target (torch.Tensor): Ground truth labels.
        topk (tuple): A tuple of integers specifying the top-k values to compute accuracy for.

    Returns:
        list: A list of accuracy values for each k in topk.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top-k predictions
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
