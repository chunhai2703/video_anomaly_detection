"""Video IO helpers optimized for inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".flv", ".gif"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe_video(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {path}")
    info = VideoInfo(
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 25.0),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    )
    cap.release()
    return info


def iter_video_frames(path: str | Path, stride: int = 1, max_frames: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    stride = max(1, int(stride))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {path}")

    read_index = 0
    yielded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if read_index % stride == 0:
                yielded += 1
                yield read_index, frame
                if max_frames is not None and yielded >= max_frames:
                    break
            read_index += 1
    finally:
        cap.release()
