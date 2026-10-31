"""Extract frames from the ShanghaiTech training videos.

The test set ships as frames already; the training set ships as .avi files.
This writes training/frames/<video_name>/<index>.jpg so both splits share one layout.

Usage:
    python scripts/extract_frames.py --root C:/Users/Admin/datasets/shanghaitech
"""
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def extract(video_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{idx:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        idx += 1
    cap.release()
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="ShanghaiTech root folder")
    args = parser.parse_args()

    video_dir = Path(args.root) / "training" / "videos"
    frame_dir = Path(args.root) / "training" / "frames"
    videos = sorted(video_dir.glob("*.avi"))
    if not videos:
        raise SystemExit(f"No .avi files found in {video_dir}")

    total = 0
    for v in tqdm(videos, desc="videos"):
        out = frame_dir / v.stem
        if out.exists() and any(out.iterdir()):
            continue  # already extracted; makes the script resumable
        total += extract(v, out)
    print(f"Extracted {total} frames from {len(videos)} videos into {frame_dir}")


if __name__ == "__main__":
    main()
