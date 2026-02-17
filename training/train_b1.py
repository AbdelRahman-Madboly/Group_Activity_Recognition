"""
Training script for Baseline B1: Frame-level image classifier.

Trains a ResNet50 on the middle frame of each clip to predict the group
activity. This is the simplest baseline and serves as the starting point.

Expected accuracy: ~77-78% on the test set.
Paper (AlexNet) baseline: ~67%.

Usage from the repo root:
    python training/train_b1.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config as cfg
from config import seed_everything, make_output_dirs
from data.datasets import VolleyballFrameDataset
from data.transforms import get_transforms
from models.baselines.b1_image_classifier import B1ImageClassifier
from training.trainer import train_one_epoch, evaluate, save_checkpoint
from evaluation.metrics import print_results_summary


def build_dataloaders(batch_size, num_workers):
    train_ds = VolleyballFrameDataset(
        cfg.VIDEOS_ROOT, cfg.TRAIN_VIDEOS, cfg.ACTIVITIES,
        transform=get_transforms('train', 'frame'),
    )
    val_ds = VolleyballFrameDataset(
        cfg.VIDEOS_ROOT, cfg.VAL_VIDEOS, cfg.ACTIVITIES,
        transform=get_transforms('val', 'frame'),
    )
    test_ds = VolleyballFrameDataset(
        cfg.VIDEOS_ROOT, cfg.TEST_VIDEOS, cfg.ACTIVITIES,
        transform=get_transforms('test', 'frame'),
    )

    pin = (cfg.DEVICE == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    print(f"Train: {len(train_ds)} clips | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


def train(num_epochs=20, batch_size=16, lr=1e-4, weight_decay=1e-4,
          dropout=0.3, num_workers=2):

    seed_everything()
    checkpoint_dir, results_dir = make_output_dirs('b1')
    device = cfg.DEVICE

    print(f"Device: {device}")
    print(f"Baseline: B1 - Frame-level image classification")
    print("=" * 70)

    train_loader, val_loader, test_loader = build_dataloaders(batch_size, num_workers)

    model = B1ImageClassifier(
        num_classes=cfg.NUM_ACTIVITY_CLASSES,
        pretrained=True,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_checkpoint = os.path.join(checkpoint_dir, 'best.pth')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 70)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(f"  Val    loss={val_loss:.4f}  acc={val_acc:.4f}  f1={val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, best_checkpoint, epoch + 1, val_acc)
            print(f"  Saved new best checkpoint (val_acc={val_acc:.4f})")

    print("\n" + "=" * 70)
    print(f"Training complete. Best val acc: {best_val_acc:.4f}")

    # Evaluate on the test set using the best checkpoint
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print_results_summary(
        baseline_name='B1',
        val_acc=best_val_acc,
        test_acc=test_acc,
        test_f1=test_f1,
        test_preds=test_preds,
        test_labels=test_labels,
        results_dir=results_dir,
    )

    return model, history


if __name__ == '__main__':
    train()
