"""
Training script for Baseline B3: Person-level feature pooling.

Three-stage pipeline:
  Stage A - Train ResNet50 on individual player crops to classify person actions.
  Stage B - Use the trained model to extract and max-pool player features per frame.
  Stage C - Train a small MLP on the pooled features to classify group activity.

Expected accuracy: ~78-82% on the test set, compared to B1's ~77-78%.
The improvement comes from learning person-level representations first.

Usage from the repo root:
    python training/train_b3.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import Counter
from PIL import Image

import config as cfg
from config import seed_everything, make_output_dirs
from data.datasets import PersonCropDataset, FeatureDataset
from data.transforms import get_transforms
from models.baselines.b3_person_pooling import PersonActionClassifier, GroupActivityClassifier
from training.trainer import train_one_epoch, evaluate, save_checkpoint
from evaluation.metrics import print_results_summary


def build_crop_dataloaders(batch_size, num_workers):
    """Build dataloaders for Stage A (person crop classification)."""
    train_ds = PersonCropDataset(
        cfg.VIDEOS_ROOT, cfg.TRAIN_VIDEOS, cfg.ACTIONS,
        transform=get_transforms('train', 'crop'),
    )
    val_ds = PersonCropDataset(
        cfg.VIDEOS_ROOT, cfg.VAL_VIDEOS, cfg.ACTIONS,
        transform=get_transforms('val', 'crop'),
    )
    test_ds = PersonCropDataset(
        cfg.VIDEOS_ROOT, cfg.TEST_VIDEOS, cfg.ACTIONS,
        transform=get_transforms('test', 'crop'),
    )

    pin = (cfg.DEVICE == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    print(f"Person crops  Train={len(train_ds):,}  Val={len(val_ds):,}  Test={len(test_ds):,}")
    return train_loader, val_loader, test_loader, train_ds


def compute_class_weights(dataset: PersonCropDataset, num_classes: int, device: str):
    """
    Compute inverse-frequency class weights to handle the severe imbalance
    in person action labels (standing is 70% of all crops).
    """
    labels = [s[2] for s in dataset.samples]
    counts = Counter(labels)
    total  = sum(counts.values())

    weights = torch.tensor([
        total / (num_classes * counts.get(i, 1))
        for i in range(num_classes)
    ], dtype=torch.float32).to(device)

    return weights


@torch.no_grad()
def extract_pooled_features(videos_root, video_ids, model, transform, device):
    """
    Stage B: for each annotated frame, crop all players, run them through
    the feature extractor, then max-pool across players to get one vector
    per frame. Returns (features, labels) as numpy arrays.
    """
    from data.annotation_parser import load_split_annotations

    model.eval()
    all_clips = load_split_annotations(videos_root, video_ids, cfg.ACTIVITIES)

    pooled_features = []
    activity_labels = []

    for clip in tqdm(all_clips, desc='Extracting features'):
        image = Image.open(clip.img_path).convert('RGB')
        img_w, img_h = image.size

        player_feats = []
        for player in clip.players:
            x, y, w, h = player.bbox
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)

            crop = image.crop((x1, y1, x2, y2))
            crop_tensor = transform(crop).unsqueeze(0).to(device)
            feat = model.extract_features(crop_tensor)  # (1, 2048)
            player_feats.append(feat)

        if player_feats:
            stacked = torch.cat(player_feats, dim=0)      # (num_players, 2048)
            pooled  = torch.max(stacked, dim=0)[0]         # (2048,)
            pooled_features.append(pooled.cpu().numpy())
            activity_labels.append(cfg.ACTIVITIES[clip.activity])

    return np.array(pooled_features), np.array(activity_labels)


def stage_a(checkpoint_dir, batch_size=32, num_epochs=15,
            lr=1e-4, weight_decay=1e-4, num_workers=2):
    """Train the person action classifier."""
    device = cfg.DEVICE
    print("\nStage A: Training person action classifier")
    print("=" * 70)

    train_loader, val_loader, test_loader, train_ds = build_crop_dataloaders(
        batch_size, num_workers
    )

    model = PersonActionClassifier(
        num_actions=cfg.NUM_ACTION_CLASSES,
        pretrained=True,
    ).to(device)

    weights   = compute_class_weights(train_ds, cfg.NUM_ACTION_CLASSES, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_acc  = 0.0
    best_ckpt_path = os.path.join(checkpoint_dir, 'b3_stage_a_best.pth')

    for epoch in range(num_epochs):
        print(f"\n  Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(f"  Val    loss={val_loss:.4f}  acc={val_acc:.4f}  f1={val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, best_ckpt_path, epoch + 1, val_acc)
            print(f"  Saved Stage A checkpoint (val_acc={val_acc:.4f})")

    print(f"\nStage A complete. Best person action val acc: {best_val_acc:.4f}")
    return model, best_ckpt_path


def stage_b(person_model, transform, device):
    """Extract and max-pool player features for all splits."""
    print("\nStage B: Extracting pooled player features")
    print("=" * 70)

    X_train, y_train = extract_pooled_features(
        cfg.VIDEOS_ROOT, cfg.TRAIN_VIDEOS, person_model, transform, device
    )
    X_val, y_val = extract_pooled_features(
        cfg.VIDEOS_ROOT, cfg.VAL_VIDEOS, person_model, transform, device
    )
    X_test, y_test = extract_pooled_features(
        cfg.VIDEOS_ROOT, cfg.TEST_VIDEOS, person_model, transform, device
    )

    print(f"Train features: {X_train.shape}")
    print(f"Val features:   {X_val.shape}")
    print(f"Test features:  {X_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def stage_c(train_data, val_data, test_data, checkpoint_dir, results_dir,
            batch_size=32, num_epochs=25, lr=1e-4, weight_decay=1e-4):
    """Train the group activity classifier on pooled features."""
    device = cfg.DEVICE
    print("\nStage C: Training group activity classifier on pooled features")
    print("=" * 70)

    X_train, y_train = train_data
    X_val, y_val     = val_data
    X_test, y_test   = test_data

    train_loader = DataLoader(FeatureDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(FeatureDataset(X_val, y_val),
                              batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(FeatureDataset(X_test, y_test),
                              batch_size=batch_size, shuffle=False)

    model = GroupActivityClassifier(
        feat_dim=cfg.FEATURE_DIM,
        hidden_dim=512,
        num_classes=cfg.NUM_ACTIVITY_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_acc   = 0.0
    best_ckpt_path = os.path.join(checkpoint_dir, 'b3_stage_c_best.pth')

    for epoch in range(num_epochs):
        print(f"\n  Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(f"  Val    loss={val_loss:.4f}  acc={val_acc:.4f}  f1={val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, best_ckpt_path, epoch + 1, val_acc)
            print(f"  Saved Stage C checkpoint (val_acc={val_acc:.4f})")

    # Final test evaluation
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print_results_summary(
        baseline_name='B3',
        val_acc=best_val_acc,
        test_acc=test_acc,
        test_f1=test_f1,
        test_preds=test_preds,
        test_labels=test_labels,
        results_dir=results_dir,
    )

    return model


def train():
    seed_everything()
    checkpoint_dir, results_dir = make_output_dirs('b3')
    device = cfg.DEVICE

    print(f"Device: {device}")
    print(f"Baseline: B3 - Person-level feature pooling")

    # Stage A
    person_model, _ = stage_a(checkpoint_dir)

    # Stage B
    val_transform = get_transforms('val', 'crop')
    train_data, val_data, test_data = stage_b(person_model, val_transform, device)

    # Stage C
    stage_c(train_data, val_data, test_data, checkpoint_dir, results_dir)


if __name__ == '__main__':
    train()
