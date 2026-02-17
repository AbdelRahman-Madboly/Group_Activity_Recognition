import os
import random
import numpy as np
import torch


# Dataset paths - these point to where the data lives on Kaggle.
# When running locally, override these with your local paths.
DATASET_ROOT = '/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_'
VIDEOS_ROOT = os.path.join(DATASET_ROOT, 'videos')
ANNOT_ROOT = (
    '/kaggle/input/datasets/ahmedmohamed365/volleyball'
    '/volleyball_tracking_annotation/volleyball_tracking_annotation'
)

# Output directories for checkpoints and results
OUTPUT_ROOT = '/kaggle/working/outputs'
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(OUTPUT_ROOT, 'results')

# The official video-level train/val/test split from the paper.
# Splitting at the video level is critical - never mix clips from the same
# video across splits, or you get data leakage.
TRAIN_VIDEOS = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31,
                32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
VAL_VIDEOS   = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
TEST_VIDEOS  = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

# Group activity labels. These are the 8 team-level classes we predict.
# r = right team, l = left team.
ACTIVITIES = {
    'r_set': 0, 'r_spike': 1, 'r-pass': 2, 'r_winpoint': 3,
    'l_set': 4, 'l_spike': 5, 'l-pass': 6, 'l_winpoint': 7,
}
ACTIVITIES_INV = {v: k for k, v in ACTIVITIES.items()}
NUM_ACTIVITY_CLASSES = 8

# Person action labels. These are the 9 individual-level classes.
# Standing dominates at ~70% of all annotations.
ACTIONS = {
    'blocking': 0, 'digging': 1, 'falling': 2, 'jumping': 3,
    'moving': 4, 'setting': 5, 'spiking': 6, 'standing': 7, 'waiting': 8,
}
ACTIONS_INV = {v: k for k, v in ACTIONS.items()}
NUM_ACTION_CLASSES = 9

# When you flip an image horizontally, left and right teams swap.
# This map lets you correct the activity label after a flip augmentation.
ACTIVITY_FLIP_MAP = {
    'r_set': 'l_set',       'l_set': 'r_set',
    'r_spike': 'l_spike',   'l_spike': 'r_spike',
    'r-pass': 'l-pass',     'l-pass': 'r-pass',
    'r_winpoint': 'l_winpoint', 'l_winpoint': 'r_winpoint',
}

# Temporal window: 5 frames before the key frame, the key frame itself,
# and 4 frames after. Total = 10 frames per clip.
NUM_FRAMES_BEFORE = 5
NUM_FRAMES_AFTER  = 4
SEQUENCE_LENGTH   = NUM_FRAMES_BEFORE + 1 + NUM_FRAMES_AFTER

# ResNet50 feature dimension (output of avgpool, before the FC head)
FEATURE_DIM = 2048

# ImageNet normalization values. Required because we use pretrained ResNet50.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Training hyperparameters shared across baselines
BATCH_SIZE    = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 2
DROPOUT       = 0.3
SEED          = 42

# These high-resolution videos have a different resolution (1920x1080)
# compared to the rest (1280x720). Worth knowing when you debug.
HIGH_RES_VIDEOS = [2, 37, 38, 39, 40, 41, 44, 45]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_everything(seed=SEED):
    """Fix all random seeds so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_output_dirs(baseline_name: str):
    """Create checkpoint and result directories for a given baseline run."""
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, baseline_name)
    results_dir    = os.path.join(RESULTS_DIR, baseline_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return checkpoint_dir, results_dir
