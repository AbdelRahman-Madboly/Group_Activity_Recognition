"""
Baseline B3: Person-level feature pooling.

This baseline has three stages:

  Stage A - Train a ResNet50 to classify individual player crops into
            9 person action classes. This teaches the network to recognize
            what each player is doing.

  Stage B - Use the trained Stage A model as a feature extractor. For each
            frame, crop every player, extract a 2048-dim feature vector,
            then max-pool across all players to get one frame-level vector.

  Stage C - Train a small MLP on the pooled frame features to predict the
            8 group activity classes.

The two models in this file are PersonActionClassifier (used in Stage A and B)
and GroupActivityClassifier (used in Stage C).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PersonActionClassifier(nn.Module):
    """
    ResNet50 classifier for individual player action recognition (9 classes).

    The extract_features method bypasses the classification head and returns
    the raw 2048-dim pooled feature vector. This is used in Stage B to build
    frame-level representations by aggregating player features.
    """

    def __init__(self, num_actions: int = 9, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()

        weights = 'IMAGENET1K_V1' if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        self.feat_dim = self.backbone.fc.in_features  # 2048

        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_actions),
        )

    def forward(self, x):
        return self.backbone(x)

    @torch.no_grad()
    def extract_features(self, x):
        """
        Run the backbone up to the global average pooling layer and return
        the flattened feature vector. Shape: (batch, 2048).

        This method is called during Stage B feature extraction. We use
        no_grad at the call site to avoid storing gradients for the whole
        training set worth of crops.
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class GroupActivityClassifier(nn.Module):
    """
    Small MLP that maps pooled player features to group activity classes.

    Input is the max-pooled feature vector from Stage B (shape: 2048).
    Output is a logit vector of size num_classes (8 group activities).

    We use a single hidden layer because the feature extractor (ResNet50)
    already does the heavy lifting. The MLP just needs to learn the mapping
    from pooled person features to the team activity.
    """

    def __init__(self, feat_dim: int = 2048, hidden_dim: int = 512,
                 num_classes: int = 8, dropout: float = 0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)
