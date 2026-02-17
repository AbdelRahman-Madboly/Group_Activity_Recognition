"""
PyTorch Dataset classes for the volleyball group activity recognition project.

Three datasets are defined here to support the different baselines:

  VolleyballFrameDataset  - Used by B1 and B4. Loads the middle frame of
                            each clip as a full image.

  PersonCropDataset       - Used by B3 and B5. Crops individual players
                            from frames and returns action labels.

  FeatureDataset          - Used by B3 Stage C and B5. Wraps pre-extracted
                            numpy feature arrays for fast training of the
                            group-level classifier.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Optional

from data.annotation_parser import load_split_annotations, ClipAnnotation
from config import ACTIVITIES, ACTIONS, VIDEOS_ROOT


class VolleyballFrameDataset(Dataset):
    """
    Loads the middle (key) frame for each clip along with its group activity label.

    This is the dataset for Baseline B1. The model sees one image per clip
    and predicts which of the 8 group activities is happening.
    """

    def __init__(self, videos_root: str, video_ids: List[int],
                 activity_map: dict, transform=None):
        self.activity_map = activity_map
        self.transform    = transform

        clips = load_split_annotations(videos_root, video_ids, activity_map)
        self.samples: List[Tuple[str, int]] = [
            (clip.img_path, activity_map[clip.activity])
            for clip in clips
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class PersonCropDataset(Dataset):
    """
    Crops individual players from the middle frame and returns action labels.

    This is the dataset for Baseline B3 Stage A. Each sample is a single
    player bounding box crop paired with that player's action label.

    The samples list holds (img_path, bbox, action_label) tuples so that
    the actual cropping happens lazily in __getitem__.
    """

    def __init__(self, videos_root: str, video_ids: List[int],
                 action_map: dict, transform=None):
        self.action_map = action_map
        self.transform  = transform

        # Build the full list of (img_path, bbox, action_label)
        self.samples: List[Tuple[str, Tuple, int]] = []

        clips = load_split_annotations(videos_root, video_ids, ACTIVITIES)
        for clip in clips:
            for player in clip.players:
                if player.action not in action_map:
                    continue
                self.samples.append((
                    clip.img_path,
                    player.bbox,
                    action_map[player.action],
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        x, y, w, h = bbox
        # Clamp the crop to image boundaries to avoid errors on edge cases
        img_w, img_h = image.size
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        crop = image.crop((x1, y1, x2, y2))

        if self.transform:
            crop = self.transform(crop)

        return crop, label


class FeatureDataset(Dataset):
    """
    Wraps numpy arrays of pre-extracted features and their labels.

    Used in B3 Stage C after features have been extracted and max-pooled.
    Loading from pre-extracted arrays is much faster than re-running the
    backbone on every training step.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.labels   = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
