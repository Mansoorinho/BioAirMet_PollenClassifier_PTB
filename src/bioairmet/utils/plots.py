'''
@file    :   plots.py
@create date : 2025-05-20 13:05:36
@modify date 2026-04-22 11:31:19
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    This utility script provides visualization tools for the BioAirMet project, supporting both training progress monitoring and model evaluation.
    - Specialized plotting functions for SSL metrics (alignment, uniformity) and classification metrics (loss, accuracy, LR).
    - Advanced Confusion Matrix visualization using Seaborn heatmaps, featuring automatic class sorting by frequency and custom highlighting for diagonal (correct) predictions.
    - Robust handling of mismatched label sets between training and validation data.
    - Configured to use a non-interactive Matplotlib backend ('Agg') for stability in multi-threaded and headless environments.
    ]
'''
import matplotlib
matplotlib.use('Agg') # Set backend to non-interactive to prevent memory leaks and thread issues
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix

def plot_classification_metrics(train_losses, val_losses, lrs, save_dir, filename_prefix="classification_metrics"):
    """
    Plots classification training and validation metrics (losses, LR) over epochs.

    Args:
        train_losses (list): List of training loss values per epoch.
        val_losses (list): List of validation loss values per epoch.
        lrs (list): List of learning rate values per epoch.
        save_dir (str): Directory to save the plots.
        filename_prefix (str): Prefix for the filenames of the saved plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    if not train_losses:
        # Nothing recorded yet (aborted / not-yet-started run) — nothing to plot.
        return
    epochs = range(1, len(train_losses) + 1)

    # Plot Losses (validation curve is omitted when validation was disabled)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title("Classification Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename_prefix}_losses.png"))
    plt.close()
    print(f"Classification Losses plot saved to {os.path.join(save_dir, f'{filename_prefix}_losses.png')}")

    # Plot Learning Rate
    if lrs:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(lrs) + 1), lrs, label='Learning Rate', color='red')
        plt.title("Classification Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}_lr.png"))
        plt.close()
        print(f"Classification Learning Rate plot saved to {os.path.join(save_dir, f'{filename_prefix}_lr.png')}")
    
def plot_ssl_metrics(train_losses, val_losses, lrs, alignments, uniformities, save_dir, filename_prefix="ssl_metrics"):
    """
    Plots SSL training and validation metrics (losses, LR, alignment, uniformity) over epochs.

    Args:
        train_losses (list): List of training loss values per epoch.
        val_losses (list): List of validation loss values per epoch.
        lrs (list): List of learning rate values per epoch.
        alignments (list): List of alignment metric values per epoch.
        uniformities (list): List of uniformity metric values per epoch.
        save_dir (str): Directory to save the plots.
        filename_prefix (str): Prefix for the filenames of the saved plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    if not train_losses:
        # Nothing recorded yet (aborted / not-yet-started run) — nothing to plot.
        return
    epochs = range(1, len(train_losses) + 1)

    # Plot Losses (validation curve is omitted when validation was disabled)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title("SSL Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename_prefix}_losses.png"))
    plt.close()
    print(f"SSL Losses plot saved to {os.path.join(save_dir, f'{filename_prefix}_losses.png')}")

    # Plot Learning Rate
    if lrs:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(lrs) + 1), lrs, label='Learning Rate', color='red')
        plt.title("SSL Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}_lr.png"))
        plt.close()
        print(f"SSL Learning Rate plot saved to {os.path.join(save_dir, f'{filename_prefix}_lr.png')}")

    # Plot Alignment and Uniformity (validation metrics — skipped when
    # validation was disabled)
    if alignments or uniformities:
        plt.figure(figsize=(10, 6))
        if alignments:
            plt.plot(range(1, len(alignments) + 1), alignments, label='Alignment')
        if uniformities:
            plt.plot(range(1, len(uniformities) + 1), uniformities, label='Uniformity')
        plt.title("SSL Validation Alignment and Uniformity")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}_alignment_uniformity.png"))
        plt.close()
        print(f"SSL Alignment and Uniformity plot saved to {os.path.join(save_dir, f'{filename_prefix}_alignment_uniformity.png')}")

def get_sorted_confusion_matrix(true_labels, pred_labels, class_counts):
    """
    Returns a confusion matrix and sorted class indices,
    filtering out true label classes with no validation samples.
    Ensures proper alignment between classes and labels.
    """
    # Get all unique classes in true and predicted labels
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(pred_labels)
    
    # Remove -1 from consideration for sorting
    known_true = [l for l in unique_true if l != -1]
    known_pred = [l for l in unique_pred if l != -1]
    
    # Sort all known classes by frequency (descending)
    all_known_classes = list(set(known_true + known_pred))
    sorted_all_classes = sorted(all_known_classes, key=lambda x: -class_counts.get(x, 0))
    
    # Add -1 at the end if present in either true or predicted
    if -1 in unique_true or -1 in unique_pred:
        sorted_all_classes.append(-1)
    
    # Filter true label classes: only keep those present in validation data
    # IMPORTANT: Maintain the same order as sorted_all_classes
    true_classes_present = [cls for cls in sorted_all_classes if cls in unique_true]
    
    # Keep all predicted classes for columns (same order as sorted_all_classes)
    pred_classes_all = sorted_all_classes
    
    # Build full confusion matrix first
    cm_full = confusion_matrix(true_labels, pred_labels, labels=sorted_all_classes)
    
    # Create mapping from class to index in the full matrix
    class_to_full_idx = {cls: i for i, cls in enumerate(sorted_all_classes)}
    
    # Get row indices for classes present in true labels
    row_indices = [class_to_full_idx[cls] for cls in true_classes_present]
    
    # Filter the confusion matrix to keep only relevant rows
    cm_filtered = cm_full[row_indices, :]
    
    return cm_filtered, true_classes_present, pred_classes_all

def plot_confusion_matrix(*, cm, total_samples, true_classes, pred_classes, class_names, accuracy, title, save_path, normalize=False):
    """
    Plots and saves the confusion matrix with properly aligned filtered true label classes.
    When normalize=False, cell colors are based on row percentages, but text shows absolute counts.
    """
    # Map class indices to names for true labels (rows) - maintaining order
    true_label_names = []
    for cls in true_classes:
        if cls == -1:
            true_label_names.append('Unknown')
        elif cls in class_names:
            true_label_names.append(class_names[cls])
        else:
            true_label_names.append(f'Class_{cls}')
    
    # Map class indices to names for predicted labels (columns) - maintaining order
    pred_label_names = []
    for cls in pred_classes:
        if cls == -1:
            pred_label_names.append('Unknown')
        elif cls in class_names:
            pred_label_names.append(class_names[cls])
        else:
            pred_label_names.append(f'Class_{cls}')
    
    num_true_categories = len(true_label_names)
    num_pred_categories = len(pred_label_names)

    # Calculate row-normalized values (0 to 100%) for consistent coloring
    with np.errstate(all='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        # Divide by row sums, defaulting to 0 where row sum is 0 to avoid NaN
        cm_row_normalized = np.divide(cm, row_sums, where=row_sums!=0, out=np.zeros_like(cm, dtype=float))
        cm_row_normalized[np.isnan(cm_row_normalized)] = 0
        cm_color_data = cm_row_normalized * 100  # Scale to 0-100 for the colormap
    
    if normalize:
        # Both color and text show percentages
        cm_to_plot = cm_color_data
        annot_data = cm_color_data
        fmt = '.2f'
    else:
        # Color is based on row percentage, but text shows absolute counts
        cm_to_plot = cm_color_data
        annot_data = cm
        fmt = 'd'

    # Adjust figure size based on the actual matrix dimensions.
    # constrained layout handles the rotated tick labels reliably (tight_layout
    # can clip/fail on large annotated heatmaps).
    plt.figure(figsize=(min(2.0 * num_pred_categories, 27), min(1.3 * num_true_categories, 16)),
               layout='constrained')
    
    # vmin=0, vmax=100 ensures the color scale is always strictly 0% to 100% of the row total
    ax = sns.heatmap(cm_to_plot, annot=annot_data, fmt=fmt,
                     cmap='Reds', cbar=False,
                     xticklabels=pred_label_names,
                     yticklabels=true_label_names,
                     annot_kws={"size": 14},
                     vmin=0, vmax=100)

    # Add dashed blue border to diagonal cells
    for i, true_cls in enumerate(true_classes):
        try:
            j = pred_classes.index(true_cls)  # Find column index for this class
            rect = patches.Rectangle((j, i), 1, 1, fill=False,
                                     edgecolor='deepskyblue', linewidth=2.5, linestyle='--', alpha=0.9)
            ax.add_patch(rect)
        except ValueError:
            # Class not in pred_classes, skip diagonal highlighting
            continue

    plt.title(f'{title}\nAccuracy: {accuracy:.2f}%', fontsize=18, pad=20, weight='bold')
    plt.xlabel('Predicted Labels', fontsize=17, labelpad=7, weight='bold')
    plt.ylabel('True Labels', fontsize=17, labelpad=7, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=16, weight='bold')
    plt.yticks(rotation=0, fontsize=16, weight='bold')

    # (layout is 'constrained' — no tight_layout needed)
    
    # Use title to create a more descriptive filename
    safe_title = title.replace(' ', '_').replace(':', '').lower()
    filename = f'{safe_title}_{"normalized" if normalize else "absolute"}.png'
    save_full_path = os.path.join(save_path, filename)
    
    print(f'Saving confusion matrix plot to {save_full_path}')
    plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
    plt.close()