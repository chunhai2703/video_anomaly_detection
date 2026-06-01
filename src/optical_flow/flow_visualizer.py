import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

from .optical_flow import flow_to_magnitude_angle


def flow_to_hsv_image(flow: np.ndarray) -> np.ndarray:
    """
    Convert optical flow to an HSV visualization (BGR output).
    Hue encodes angle, Value encodes magnitude.
    """
    magnitude, angle = flow_to_magnitude_angle(flow)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def flow_to_heatmap(flow: np.ndarray) -> np.ndarray:
    """
    Create a heatmap from flow magnitude using a JET colormap.
    """
    magnitude, _ = flow_to_magnitude_angle(flow)
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)
    return heatmap


def draw_flow_arrows(frame: np.ndarray, flow: np.ndarray, step: int = 16) -> np.ndarray:
    """
    Draw arrowed vectors on the provided frame to indicate flow.
    """
    if len(frame.shape) == 2:
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        vis = frame.copy()

    h, w = flow.shape[:2]
    y_coords, x_coords = np.mgrid[step // 2:h:step, step // 2:w:step]

    fx = flow[y_coords, x_coords, 0]
    fy = flow[y_coords, x_coords, 1]

    for (x, y, dx, dy) in zip(
        x_coords.flatten(), y_coords.flatten(), fx.flatten(), fy.flatten()
    ):
        start = (int(x), int(y))
        end = (int(x + dx), int(y + dy))
        cv2.arrowedLine(vis, start, end, color=(0, 255, 0), thickness=1, tipLength=0.3)

    return vis


def plot_motion_score_timeline(scores: List[float], save_path: Optional[str] = None):
    """
    Plot motion scores over time. Optionally save to `save_path`.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(scores, color='royalblue', linewidth=1.5, label='Motion Score')
    if len(scores) > 0:
        plt.axhline(
            y=np.mean(scores) + 2 * np.std(scores),
            color='red', linestyle='--', label='Anomaly Threshold (mean + 2σ)'
        )
    plt.xlabel('Frame Index')
    plt.ylabel('Motion Score')
    plt.title('Motion Score Timeline')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved motion timeline to: {save_path}")
    else:
        plt.show()

    plt.close()
