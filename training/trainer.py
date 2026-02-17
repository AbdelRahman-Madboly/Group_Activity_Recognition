"""
Core training and validation loop functions.

These functions are reused across all baselines. The only thing that changes
between baselines is the model, the dataloader, and the criterion - the loop
itself stays the same.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
from typing import Tuple, List


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    """
    Run one full pass over the training data.

    Returns the average loss and accuracy for this epoch.
    Gradients are computed and the optimizer steps after every batch.
    """
    model.train()

    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    pbar = tqdm(loader, desc='Train', leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f'{loss.item():.4f}')

    avg_loss = total_loss / len(loader.dataset)
    accuracy  = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, float, List[int], List[int]]:
    """
    Evaluate the model on a validation or test set.

    Returns average loss, accuracy, weighted F1 score, predictions, and labels.
    No gradients are computed here.
    """
    model.eval()

    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for inputs, labels in tqdm(loader, desc='Eval', leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy  = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1, all_preds, all_labels


def save_checkpoint(model: nn.Module, path: str, epoch: int, val_acc: float):
    """Save a model checkpoint with metadata."""
    torch.save({
        'epoch': epoch,
        'val_acc': val_acc,
        'model_state_dict': model.state_dict(),
    }, path)


def load_checkpoint(model: nn.Module, path: str, device: str):
    """Load a model checkpoint. Returns the checkpoint dict for metadata access."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint
