import numpy as np
from typing import Optional, Sequence


def _minmax_normalize(arr: Sequence[float]) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    minv = a.min()
    maxv = a.max()
    if maxv - minv < 1e-8:
        return np.zeros_like(a)
    return (a - minv) / (maxv - minv)


def combine_scores(recon_scores: Sequence[float], motion_scores: Sequence[float], alpha: float = 0.6) -> np.ndarray:
    """
    Combine reconstruction-based anomaly scores and motion-based scores.

    Args:
        recon_scores: Reconstruction error scores (higher -> more anomalous).
        motion_scores: Motion magnitude scores (higher -> more motion).
        alpha: Weight for reconstruction scores in combined score (0..1).

    Returns:
        np.ndarray: Combined scores normalized to [0, 1].
    """
    recon_n = _minmax_normalize(recon_scores)
    motion_n = _minmax_normalize(motion_scores)
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))
    combined = alpha * recon_n + (1.0 - alpha) * motion_n
    return _minmax_normalize(combined)


def apply_threshold(scores, threshold, motion_scores: Optional[Sequence[float]] = None, alpha: float = 0.6):
    """
    Convert anomaly scores to binary predictions using a threshold.

    If `motion_scores` is provided, a combined score is computed using
    `combine_scores(recon, motion, alpha)` and the threshold is applied to
    the combined score. If `motion_scores` is None, behaves like the old
    implementation (thresholding the input `scores`).
    """

    scores_arr = np.asarray(scores, dtype=np.float32)

    if motion_scores is not None:
        combined = combine_scores(scores_arr, motion_scores, alpha=alpha)
        active_scores = combined
    else:
        # Keep backward compatibility: assume scores are already comparable to threshold
        active_scores = scores_arr

    predictions = (active_scores > float(threshold)).astype(int)
    return predictions