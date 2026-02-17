"""
Sanity checks for model forward passes.

Makes sure each model runs without errors and produces the right output shape
before you spend time waiting for a full training run to fail on epoch 1.

Usage:
    python -m pytest tests/test_models.py -v
    # or just:
    python tests/test_models.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.baselines.b1_image_classifier import B1ImageClassifier
from models.baselines.b3_person_pooling import PersonActionClassifier, GroupActivityClassifier
from config import NUM_ACTIVITY_CLASSES, NUM_ACTION_CLASSES, FEATURE_DIM


def test_b1_forward():
    """B1 should map a batch of images to 8 class logits."""
    model = B1ImageClassifier(num_classes=NUM_ACTIVITY_CLASSES, pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, NUM_ACTIVITY_CLASSES), f"Wrong output shape: {out.shape}"
    print(f"test_b1_forward  passed  (output: {tuple(out.shape)})")


def test_person_classifier_forward():
    """PersonActionClassifier should output 9 class logits per crop."""
    model = PersonActionClassifier(num_actions=NUM_ACTION_CLASSES, pretrained=False)
    model.eval()

    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (4, NUM_ACTION_CLASSES), f"Wrong output shape: {out.shape}"
    print(f"test_person_classifier_forward  passed  (output: {tuple(out.shape)})")


def test_person_classifier_feature_extraction():
    """extract_features should return 2048-dim vectors."""
    model = PersonActionClassifier(num_actions=NUM_ACTION_CLASSES, pretrained=False)
    model.eval()

    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        feats = model.extract_features(x)

    assert feats.shape == (3, FEATURE_DIM), f"Wrong feature shape: {feats.shape}"
    print(f"test_person_classifier_feature_extraction  passed  (features: {tuple(feats.shape)})")


def test_group_classifier_forward():
    """GroupActivityClassifier should map pooled features to 8 class logits."""
    model = GroupActivityClassifier(
        feat_dim=FEATURE_DIM, hidden_dim=512, num_classes=NUM_ACTIVITY_CLASSES
    )
    model.eval()

    x = torch.randn(8, FEATURE_DIM)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (8, NUM_ACTIVITY_CLASSES), f"Wrong output shape: {out.shape}"
    print(f"test_group_classifier_forward  passed  (output: {tuple(out.shape)})")


if __name__ == '__main__':
    print("Running model tests...\n")
    test_b1_forward()
    test_person_classifier_forward()
    test_person_classifier_feature_extraction()
    test_group_classifier_forward()
    print("\nAll tests passed.")
