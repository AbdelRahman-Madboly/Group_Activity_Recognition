"""
Evaluation utilities: confusion matrix, classification report, result summaries.

These functions are called at the end of each training run to produce a
consistent set of outputs across all baselines, making it easy to compare results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import List

from config import ACTIVITIES_INV, NUM_ACTIVITY_CLASSES


def plot_confusion_matrix(preds: List[int], labels: List[int],
                          class_names: List[str], title: str,
                          save_path: str = None):
    """
    Plot a confusion matrix heatmap and optionally save it to disk.
    """
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()
    return cm


def plot_training_curves(history: dict, title: str, save_path: str = None):
    """
    Plot training and validation loss/accuracy curves side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'],   label='Val',   marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train', marker='o')
    axes[1].plot(history['val_acc'],   label='Val',   marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{title} - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    plt.show()


def print_results_summary(baseline_name: str, val_acc: float, test_acc: float,
                           test_f1: float, test_preds: List[int],
                           test_labels: List[int], results_dir: str):
    """
    Print a full results summary and save the confusion matrix and
    classification report to the results directory.
    """
    class_names = [ACTIVITIES_INV[i] for i in range(NUM_ACTIVITY_CLASSES)]

    print("\n" + "=" * 70)
    print(f"Results: {baseline_name}")
    print("=" * 70)
    print(f"  Best val accuracy:  {val_acc:.4f}")
    print(f"  Test accuracy:      {test_acc:.4f}")
    print(f"  Test weighted F1:   {test_f1:.4f}")
    print("=" * 70)

    print("\nClassification report:\n")
    report = classification_report(test_labels, test_preds,
                                   target_names=class_names, digits=4)
    print(report)

    # Save the classification report as a text file
    report_path = os.path.join(results_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Baseline: {baseline_name}\n")
        f.write(f"Val accuracy:  {val_acc:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1:       {test_f1:.4f}\n\n")
        f.write(report)

    # Save the confusion matrix
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        preds=test_preds,
        labels=test_labels,
        class_names=class_names,
        title=f'{baseline_name} - Test Set Confusion Matrix (acc={test_acc:.4f})',
        save_path=cm_path,
    )
