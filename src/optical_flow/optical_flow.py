import cv2
import numpy as np


def compute_optical_flow(prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
    """
    Compute Farneback optical flow between two consecutive frames.

    Args:
        prev_frame (np.ndarray): Previous frame (grayscale or BGR, uint8).
        next_frame (np.ndarray): Next frame (grayscale or BGR, uint8).

    Returns:
        np.ndarray: Flow array with shape (H, W, 2) where flow[y, x] = (dx, dy).
    """
    if prev_frame is None or next_frame is None:
        raise ValueError("Input frames must not be None.")

    # Ensure frames are single-channel grayscale
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame

    if len(next_frame.shape) == 3:
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    else:
        next_gray = next_frame

    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_gray,
        next=next_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    return flow


def flow_to_magnitude_angle(flow: np.ndarray):
    """
    Split an optical flow array into magnitude and angle.

    Args:
        flow (np.ndarray): Flow array of shape (H, W, 2).

    Returns:
        tuple: (magnitude (H, W), angle (H, W) in radians)
    """
    if flow is None:
        raise ValueError("flow must not be None")

    fx = flow[..., 0]
    fy = flow[..., 1]
    magnitude, angle = cv2.cartToPolar(fx, fy)
    return magnitude, angle
