import numpy as np
from typing import List

from .optical_flow import compute_optical_flow, flow_to_magnitude_angle


def compute_motion_score(flow: np.ndarray) -> float:
    """
    Compute a scalar motion score for a frame from optical flow.

    Args:
        flow (np.ndarray): Optical flow array shape (H, W, 2).

    Returns:
        float: Motion score (mean magnitude over the frame).
    """
    magnitude, _ = flow_to_magnitude_angle(flow)
    score = float(np.mean(magnitude))
    return score


def compute_motion_map(flow: np.ndarray) -> np.ndarray:
    """
    Create a per-pixel motion map (magnitude) from optical flow.

    Args:
        flow (np.ndarray): Optical flow array shape (H, W, 2).

    Returns:
        np.ndarray: Motion map shape (H, W), float.
    """
    magnitude, _ = flow_to_magnitude_angle(flow)
    return magnitude


def normalize_motion_map(motion_map: np.ndarray) -> np.ndarray:
    """
    Normalize motion map to range [0, 1] as float32.
    """
    min_val = motion_map.min()
    max_val = motion_map.max()

    if max_val - min_val < 1e-6:
        return np.zeros_like(motion_map, dtype=np.float32)

    normalized = (motion_map - min_val) / (max_val - min_val)
    return normalized.astype(np.float32)


def compute_motion_scores_for_sequence(frames: List[np.ndarray]) -> List[float]:
    """
    Compute motion scores for a sequence of frames. Returns scores for each
    consecutive pair (len = len(frames) - 1).
    """
    scores: List[float] = []
    for i in range(len(frames) - 1):
        flow = compute_optical_flow(frames[i], frames[i + 1])
        score = compute_motion_score(flow)
        scores.append(score)
    return scores
