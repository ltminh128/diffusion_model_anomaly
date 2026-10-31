"""ShanghaiTech Campus clip dataset for frame-prediction anomaly detection.

Each item is a clip of `num_past` consecutive frames plus the next frame to predict,
following the Liu et al. (CVPR 2018) setup. Frames are resized and scaled to [-1, 1].
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".png"}


def scene_of(video_name: str) -> str:
    """Scene ID is the two-digit prefix, e.g. '01_0014' -> '01'."""
    return video_name.split("_")[0]


def list_videos(split_dir: Path, scenes=None):
    """Return {video_name: [frame paths]} for a split, optionally filtered by scene IDs."""
    videos = {}
    for vdir in sorted((split_dir / "frames").iterdir()):
        if not vdir.is_dir():
            continue
        if scenes is not None and scene_of(vdir.name) not in scenes:
            continue
        frames = sorted(p for p in vdir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if frames:
            videos[vdir.name] = frames
    return videos


class ClipDataset(Dataset):
    def __init__(self, root, split="training", num_past=4, size=(256, 256), scenes=None):
        self.split_dir = Path(root) / split
        self.num_past = num_past
        self.size = size  # (width, height) for cv2.resize
        self.videos = list_videos(self.split_dir, scenes)
        if not self.videos:
            raise FileNotFoundError(f"No frames found under {self.split_dir / 'frames'}")

        # One sample per predictable frame: indices num_past .. len-1 of every video.
        self.index = [
            (name, t)
            for name, frames in self.videos.items()
            for t in range(num_past, len(frames))
        ]

    def __len__(self):
        return len(self.index)

    def _load(self, path):
        img = cv2.imread(str(path))
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        return torch.from_numpy(img / 127.5 - 1.0).permute(2, 0, 1)  # C, H, W in [-1, 1]

    def __getitem__(self, i):
        name, t = self.index[i]
        frames = self.videos[name]
        past = torch.stack([self._load(frames[k]) for k in range(t - self.num_past, t)])
        target = self._load(frames[t])
        return {"past": past, "target": target, "video": name, "t": t}


def load_test_labels(root, video_names):
    """Frame-level ground truth: {video_name: np.ndarray of 0/1 per frame}."""
    mask_dir = Path(root) / "testing" / "test_frame_mask"
    return {v: np.load(mask_dir / f"{v}.npy").astype(np.int64) for v in video_names}
