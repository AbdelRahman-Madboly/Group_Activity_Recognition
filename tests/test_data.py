"""
Basic sanity checks for the data pipeline.

Run these before kicking off training to make sure the dataset paths,
annotation parsing, and dataloader output shapes all look correct.

Usage:
    python -m pytest tests/test_data.py -v
    # or just:
    python tests/test_data.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from config import VIDEOS_ROOT, TRAIN_VIDEOS, VAL_VIDEOS, ACTIVITIES, ACTIONS
from data.annotation_parser import parse_annotation_line, load_video_annotations
from data.datasets import VolleyballFrameDataset, PersonCropDataset
from data.transforms import get_transforms


def test_annotation_parser():
    """Check that a known annotation line parses to the expected values."""
    line = "9575.jpg r_spike 257 446 30 92 standing 326 492 51 89 blocking"
    clip = parse_annotation_line(line, video_id=1)

    assert clip is not None, "Parser returned None"
    assert clip.clip_id == 9575
    assert clip.activity == 'r_spike'
    assert len(clip.players) == 2

    p0 = clip.players[0]
    assert p0.bbox == (257, 446, 30, 92)
    assert p0.action == 'standing'

    p1 = clip.players[1]
    assert p1.bbox == (326, 492, 51, 89)
    assert p1.action == 'blocking'

    print("test_annotation_parser  passed")


def test_annotation_parser_malformed():
    """Malformed lines should return None without raising exceptions."""
    assert parse_annotation_line("", video_id=1) is None
    assert parse_annotation_line("bad line", video_id=1) is None
    print("test_annotation_parser_malformed  passed")


def test_frame_dataset_loads():
    """Check that VolleyballFrameDataset builds and returns the right shapes."""
    small_split = TRAIN_VIDEOS[:3]
    ds = VolleyballFrameDataset(
        VIDEOS_ROOT, small_split, ACTIVITIES,
        transform=get_transforms('train', 'frame'),
    )

    assert len(ds) > 0, "Dataset is empty"

    img, label = ds[0]
    assert img.shape == (3, 224, 224), f"Unexpected image shape: {img.shape}"
    assert 0 <= label < 8, f"Label out of range: {label}"

    print(f"test_frame_dataset_loads  passed  ({len(ds)} samples)")


def test_person_crop_dataset_loads():
    """Check that PersonCropDataset builds and returns the right shapes."""
    small_split = TRAIN_VIDEOS[:3]
    ds = PersonCropDataset(
        VIDEOS_ROOT, small_split, ACTIONS,
        transform=get_transforms('train', 'crop'),
    )

    assert len(ds) > 0, "Dataset is empty"

    crop, label = ds[0]
    assert crop.shape == (3, 224, 224), f"Unexpected crop shape: {crop.shape}"
    assert 0 <= label < 9, f"Label out of range: {label}"

    print(f"test_person_crop_dataset_loads  passed  ({len(ds)} samples)")


def test_dataloader_batch_shapes():
    """Check that a DataLoader produces correctly shaped batches."""
    ds = VolleyballFrameDataset(
        VIDEOS_ROOT, TRAIN_VIDEOS[:2], ACTIVITIES,
        transform=get_transforms('train', 'frame'),
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    batch_imgs, batch_labels = next(iter(loader))

    assert batch_imgs.shape[1:] == (3, 224, 224)
    assert batch_labels.shape[0] == batch_imgs.shape[0]
    assert batch_labels.dtype == torch.int64

    print(f"test_dataloader_batch_shapes  passed  (batch shape: {tuple(batch_imgs.shape)})")


if __name__ == '__main__':
    print("Running data pipeline tests...\n")
    test_annotation_parser()
    test_annotation_parser_malformed()
    test_frame_dataset_loads()
    test_person_crop_dataset_loads()
    test_dataloader_batch_shapes()
    print("\nAll tests passed.")
