"""Frame preprocessing utilities."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def resize_keep_aspect(frame: np.ndarray, max_width: int | None = None) -> np.ndarray:
    if max_width is None or frame.shape[1] <= max_width:
        return frame
    scale = max_width / float(frame.shape[1])
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)


def preprocess_frame(frame: np.ndarray, img_size: int = 128) -> torch.Tensor:
    """Convert BGR/gray frame to tensor [1, H, W] in [0, 1]."""
    if frame is None:
        raise ValueError("frame is None")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0
    return torch.from_numpy(gray).unsqueeze(0)


def preprocess_batch(frames: list[np.ndarray], img_size: int = 128) -> torch.Tensor:
    return torch.stack([preprocess_frame(frame, img_size) for frame in frames], dim=0)
