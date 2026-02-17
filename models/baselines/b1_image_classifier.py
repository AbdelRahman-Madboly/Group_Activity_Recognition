"""
Baseline B1: Frame-level image classifier.

Architecture: ResNet50 pretrained on ImageNet, with the final FC layer
replaced by a dropout + linear layer that maps 2048 features to 8 classes.

This is the simplest possible baseline - no temporal modeling, no person-level
reasoning. The model just looks at a single frame and guesses the group activity.

The original paper used AlexNet and reported ~67% accuracy. Using ResNet50
and a larger batch size, we achieve around 77-78%.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class B1ImageClassifier(nn.Module):

    def __init__(self, num_classes: int = 8, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()

        weights = 'IMAGENET1K_V1' if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # The avgpool output is 2048-dimensional for ResNet50
        num_features = self.backbone.fc.in_features

        # Replace the ImageNet head with our activity classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
