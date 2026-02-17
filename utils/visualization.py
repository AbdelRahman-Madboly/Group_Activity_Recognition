"""
Dataset visualization utilities.

These are used for exploring the data before training, not during training.
Run these once to understand what the dataset looks like.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import Counter
from typing import List, Optional

from config import VIDEOS_ROOT, ACTIVITIES, ACTIONS, NUM_FRAMES_BEFORE, NUM_FRAMES_AFTER
from data.annotation_parser import load_split_annotations, load_video_annotations


ACTION_COLORS = {
    'standing': 'lime',
    'moving':   'cyan',
    'blocking': 'yellow',
    'spiking':  'red',
    'waiting':  'orange',
    'setting':  'magenta',
    'digging':  'pink',
    'jumping':  'blue',
    'falling':  'purple',
}


def visualize_frame(video_id: int, clip_id: int, videos_root: str = None,
                    save_path: str = None):
    """
    Draw bounding boxes and action labels on the middle frame of a clip.
    """
    if videos_root is None:
        videos_root = VIDEOS_ROOT

    clips = load_video_annotations(videos_root, video_id)
    clip = next((c for c in clips if c.clip_id == clip_id), None)

    if clip is None:
        print(f"Clip {clip_id} not found in video {video_id}")
        return

    if not os.path.exists(clip.img_path):
        print(f"Image not found: {clip.img_path}")
        return

    img = Image.open(clip.img_path).convert('RGB')
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.imshow(img)

    for idx, player in enumerate(clip.players):
        x, y, w, h = player.bbox
        color = ACTION_COLORS.get(player.action, 'white')

        rect = patches.Rectangle((x, y), w, h, linewidth=3,
                                  edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 8, f"{idx + 1}: {player.action}",
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                fontsize=9, color='black', weight='bold')

    ax.set_title(
        f"Video {video_id}, Clip {clip_id} -- {clip.activity.upper()}\n"
        f"{len(clip.players)} players | {img.size[0]}x{img.size[1]}",
        fontsize=14,
    )
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def visualize_temporal_sequence(video_id: int, clip_id: int,
                                 videos_root: str = None, save_path: str = None):
    """
    Show all frames in the temporal window for a given clip side by side.
    The key frame is highlighted with a red border.
    """
    if videos_root is None:
        videos_root = VIDEOS_ROOT

    frame_ids = list(range(clip_id - NUM_FRAMES_BEFORE,
                           clip_id + NUM_FRAMES_AFTER + 1))

    n = len(frame_ids)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(20, 8))
    axes = axes.flatten()

    for idx, fid in enumerate(frame_ids):
        img_path = os.path.join(videos_root, str(video_id), str(clip_id), f'{fid}.jpg')

        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[idx].imshow(img)

            if fid == clip_id:
                axes[idx].set_title(f'Frame {fid} (key)', fontweight='bold', color='red')
                for spine in axes[idx].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(4)
            else:
                axes[idx].set_title(f'Frame {fid}', fontsize=9)
        else:
            axes[idx].text(0.5, 0.5, 'missing', ha='center', va='center',
                           transform=axes[idx].transAxes)

        axes[idx].axis('off')

    # Hide any unused axes
    for idx in range(len(frame_ids), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Video {video_id}, Clip {clip_id} - temporal window '
                 f'({NUM_FRAMES_BEFORE} before + key + {NUM_FRAMES_AFTER} after)',
                 fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def print_dataset_statistics(video_ids: List[int], videos_root: str = None):
    """
    Print class distribution statistics for a given set of videos.
    Useful for checking class balance before training.
    """
    if videos_root is None:
        videos_root = VIDEOS_ROOT

    clips = load_split_annotations(videos_root, video_ids, ACTIVITIES)

    activity_counts = Counter(c.activity for c in clips)
    action_counts   = Counter(
        p.action for c in clips for p in c.players
    )
    total_clips   = sum(activity_counts.values())
    total_players = sum(action_counts.values())

    print(f"Videos: {len(video_ids)} | Clips: {total_clips} | "
          f"Player annotations: {total_players:,}")
    print(f"Avg players per clip: {total_players / total_clips:.1f}\n")

    print("Group activity distribution:")
    for activity, count in activity_counts.most_common():
        print(f"  {activity:12s}: {count:4d} ({100 * count / total_clips:5.1f}%)")

    print("\nPerson action distribution:")
    for action, count in action_counts.most_common():
        print(f"  {action:12s}: {count:5d} ({100 * count / total_players:5.1f}%)")
