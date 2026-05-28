import argparse
import csv
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.models.dataset import preprocess_video_frame, tensor_to_uint8_image
from src.models.predictor import load_model, predict_tensor, is_anomaly


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run anomaly inference on video or frame folder"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input video path or folder of frames.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/model.pth",
        help="Path to trained model.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/inference",
        help="Folder to save inference results.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Anomaly threshold. If not provided, auto threshold will be calculated.",
    )

    parser.add_argument(
        "--auto_threshold_percentile",
        type=float,
        default=95.0,
        help="Percentile used when threshold is not provided.",
    )

    parser.add_argument(
        "--save_frames",
        action="store_true",
        help="Save reconstructed and marked anomaly frames.",
    )

    return parser.parse_args()


def read_frames_from_video(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    return frames


def read_frames_from_folder(folder_path: str) -> List[np.ndarray]:
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder_path}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    if len(image_paths) == 0:
        raise ValueError(f"No image files found in folder: {folder_path}")

    frames = []

    for path in image_paths:
        frame = cv2.imread(str(path))

        if frame is not None:
            frames.append(frame)

    if len(frames) == 0:
        raise ValueError(f"Cannot read images from folder: {folder_path}")

    return frames


def load_input_frames(input_path: str) -> List[np.ndarray]:
    path = Path(input_path)

    if path.is_dir():
        return read_frames_from_folder(input_path)

    return read_frames_from_video(input_path)


def draw_anomaly_label(frame: np.ndarray, score: float, threshold: float, anomaly: bool) -> np.ndarray:
    output = frame.copy()

    label = f"Score: {score:.6f}"
    status = "ANOMALY" if anomaly else "NORMAL"

    color = (0, 0, 255) if anomaly else (0, 255, 0)

    cv2.rectangle(output, (10, 10), (360, 80), (0, 0, 0), thickness=-1)
    cv2.putText(output, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(output, status, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if anomaly:
        cv2.rectangle(
            output,
            (5, 5),
            (output.shape[1] - 5, output.shape[0] - 5),
            (0, 0, 255),
            thickness=4,
        )

    return output


def save_results_csv(results, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_index", "anomaly_score", "threshold", "is_anomaly"]
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)

    print(f"CSV saved to: {csv_path}")


def run_inference(
    input_path: str,
    model_path: str,
    output_dir: str,
    threshold: Optional[float] = None,
    auto_threshold_percentile: float = 95.0,
    save_frames: bool = False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, img_size, device = load_model(model_path)
    frames = load_input_frames(input_path)

    print(f"Device: {device}")
    print(f"Frames loaded: {len(frames)}")
    print(f"Image size: {img_size}x{img_size}")

    scores = []
    reconstructed_tensors = []

    for frame in frames:
        tensor = preprocess_video_frame(frame, img_size=img_size)
        score, reconstructed = predict_tensor(tensor, model, device)

        scores.append(score)
        reconstructed_tensors.append(reconstructed)

    if threshold is None:
        threshold = float(np.percentile(scores, auto_threshold_percentile))
        print(
            f"Auto threshold = {threshold:.8f} "
            f"percentile={auto_threshold_percentile}"
        )
    else:
        print(f"Manual threshold = {threshold:.8f}")

    results = []

    marked_dir = output_dir / "marked_frames"
    reconstructed_dir = output_dir / "reconstructed_frames"

    if save_frames:
        marked_dir.mkdir(parents=True, exist_ok=True)
        reconstructed_dir.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(frames):
        score = scores[index]
        anomaly = is_anomaly(score, threshold)

        results.append(
            {
                "frame_index": index,
                "anomaly_score": score,
                "threshold": threshold,
                "is_anomaly": int(anomaly),
            }
        )

        if save_frames:
            marked = draw_anomaly_label(frame, score, threshold, anomaly)
            cv2.imwrite(str(marked_dir / f"frame_{index:05d}.jpg"), marked)

            reconstructed_image = tensor_to_uint8_image(reconstructed_tensors[index])
            cv2.imwrite(
                str(reconstructed_dir / f"reconstructed_{index:05d}.jpg"),
                reconstructed_image,
            )

    save_results_csv(results, output_dir / "anomaly_scores.csv")

    anomaly_count = sum(row["is_anomaly"] for row in results)

    print("Inference completed.")
    print(f"Total frames: {len(results)}")
    print(f"Anomaly frames: {anomaly_count}")
    print(f"Normal frames: {len(results) - anomaly_count}")

    return results


def main():
    args = parse_args()

    run_inference(
        input_path=args.input,
        model_path=args.model_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        auto_threshold_percentile=args.auto_threshold_percentile,
        save_frames=args.save_frames,
    )


if __name__ == "__main__":
    main()