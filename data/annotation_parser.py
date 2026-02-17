"""
Utilities for parsing volleyball dataset annotation files.

The dataset has two annotation formats:

1. Frame-level (annotations.txt inside each video folder):
   Format per line: <clip_id>.jpg <group_activity> <x> <y> <w> <h> <action> ...
   The bounding box comes before the action label for each player.

2. Tracking-level (volleyball_tracking_annotation folder):
   Format per line: <player_id> <x1> <y1> <x2> <y2> <frame_id> <lost> <grouping> <generated> <action>
   These provide per-frame tracking across the temporal window.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class PlayerAnnotation:
    """Bounding box and action label for a single player in a single frame."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height) in top-left format
    action: str


@dataclass
class ClipAnnotation:
    """All annotations for a single clip (the key frame only)."""
    video_id: int
    clip_id: int
    activity: str
    players: List[PlayerAnnotation] = field(default_factory=list)

    @property
    def img_path(self):
        """Path to the middle (key) frame image."""
        from config import VIDEOS_ROOT
        return os.path.join(VIDEOS_ROOT, str(self.video_id),
                            str(self.clip_id), f'{self.clip_id}.jpg')

    @property
    def frame_paths(self):
        """Paths to all 10 frames in the temporal window around the key frame."""
        from config import VIDEOS_ROOT, NUM_FRAMES_BEFORE, NUM_FRAMES_AFTER
        base = os.path.join(VIDEOS_ROOT, str(self.video_id), str(self.clip_id))
        frame_ids = range(self.clip_id - NUM_FRAMES_BEFORE,
                          self.clip_id + NUM_FRAMES_AFTER + 1)
        return [os.path.join(base, f'{fid}.jpg') for fid in frame_ids]


def parse_annotation_line(line: str, video_id: int) -> Optional[ClipAnnotation]:
    """
    Parse one line from annotations.txt into a ClipAnnotation.

    Returns None if the line is malformed or missing required fields.

    The annotation format is space-separated:
        <clip_id>.jpg <group_activity> <x> <y> <w> <h> <action> <x> <y> <w> <h> <action> ...
    """
    parts = line.strip().split()
    if len(parts) < 7:
        return None

    clip_id  = parts[0].replace('.jpg', '')
    activity = parts[1]

    if not clip_id.isdigit():
        return None

    players = []
    i = 2
    while i + 4 < len(parts):
        try:
            x = int(parts[i])
            y = int(parts[i + 1])
            w = int(parts[i + 2])
            h = int(parts[i + 3])
            action = parts[i + 4]
            players.append(PlayerAnnotation(bbox=(x, y, w, h), action=action))
            i += 5
        except (ValueError, IndexError):
            break

    return ClipAnnotation(
        video_id=int(video_id),
        clip_id=int(clip_id),
        activity=activity,
        players=players,
    )


def load_video_annotations(videos_root: str, video_id: int) -> List[ClipAnnotation]:
    """
    Load all clip annotations for a single video.

    Reads the annotations.txt file in the video folder and returns a list
    of ClipAnnotation objects - one per clip found in that file.
    """
    annot_file = os.path.join(videos_root, str(video_id), 'annotations.txt')
    if not os.path.exists(annot_file):
        return []

    clips = []
    with open(annot_file, 'r') as f:
        for line in f:
            clip = parse_annotation_line(line, video_id)
            if clip is not None:
                clips.append(clip)

    return clips


def load_split_annotations(videos_root: str,
                            video_ids: List[int],
                            activity_map: dict) -> List[ClipAnnotation]:
    """
    Load annotations for a list of video IDs and filter to known activity classes.

    This is the main entry point used by dataset classes. It returns only clips
    whose group activity label appears in activity_map and whose key frame image
    actually exists on disk.
    """
    all_clips = []

    for vid in video_ids:
        clips = load_video_annotations(videos_root, vid)
        for clip in clips:
            if clip.activity not in activity_map:
                continue
            # Only keep clips where the middle frame actually exists
            if os.path.exists(clip.img_path):
                all_clips.append(clip)

    return all_clips
