"""
Image transforms for training and evaluation.

We keep transforms separate from dataset classes so they can be swapped
easily during experiments without touching the dataset code.
"""

import torchvision.transforms as T
from config import IMAGENET_MEAN, IMAGENET_STD


# Standard training transform for full frames (B1, B4).
# Uses random crop for augmentation instead of center crop.
# We deliberately disable horizontal flip here because flipping a volleyball
# frame changes the semantics (left team becomes right team), which would
# require swapping the group activity label. Easier to just not flip.
train_frame_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop((224, 224)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Deterministic transform for validation and test frames.
val_frame_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Transform for individual player crops (B3, B5).
# Crops are already tightly around the player, so we skip the crop step
# and resize directly to 224x224.
train_crop_transform = T.Compose([
    T.Resize((224, 224)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_crop_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_transforms(split: str, mode: str = 'frame'):
    """
    Return the appropriate transform for a given split and mode.

    split: 'train', 'val', or 'test'
    mode: 'frame' for full images, 'crop' for player bounding box crops
    """
    is_train = (split == 'train')

    if mode == 'frame':
        return train_frame_transform if is_train else val_frame_transform
    elif mode == 'crop':
        return train_crop_transform if is_train else val_crop_transform
    else:
        raise ValueError(f"Unknown transform mode: {mode}. Use 'frame' or 'crop'.")
