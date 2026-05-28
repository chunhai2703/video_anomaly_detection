import os
import sys
import cv2

# ensure project root is on sys.path so `src` package is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.optical_flow.optical_flow import compute_optical_flow
from src.optical_flow.flow_visualizer import (
    flow_to_hsv_image,
    flow_to_heatmap,
    draw_flow_arrows,
)


def main(video_path: str = "test.mp4"):
    cap = cv2.VideoCapture(video_path)
    ret, f1 = cap.read()
    ret, f2 = cap.read()
    cap.release()

    if f1 is None or f2 is None:
        print(f"Could not read two frames from {video_path}")
        return 2

    flow = compute_optical_flow(f1, f2)
    print("flow.shape:", getattr(flow, "shape", None))

    hsv = flow_to_hsv_image(flow)
    heat = flow_to_heatmap(flow)
    arrows = draw_flow_arrows(f2, flow)

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/flow_hsv.png", hsv)
    cv2.imwrite("outputs/flow_heatmap.png", heat)
    cv2.imwrite("outputs/flow_arrows.png", arrows)

    print("Wrote outputs/flow_hsv.png, flow_heatmap.png, flow_arrows.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
