import argparse
import csv
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.models.dataset import preprocess_video_frame, tensor_to_uint8_image
from src.models.predictor import load_model, predict_tensor, is_anomaly
from src.evaluation.threshold import apply_threshold
from src.evaluation.evaluator import evaluate_model


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run anomaly inference on video or frame folder"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input video path or folder of frames/videos.",
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

    parser.add_argument(
        "--labels_csv",
        type=str,
        default=None,
        help="Optional CSV file with ground truth labels for evaluation. Columns: frame_index,label",
    )

    return parser.parse_args()


def read_frames_from_video(video_path: str) -> List[np.ndarray]:
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

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


def read_frames_from_image_folder(folder_path: str) -> List[np.ndarray]:
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder_path}")

    image_paths = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
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


def read_frames_from_video_folder(folder_path: str) -> List[np.ndarray]:
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder_path}")

    video_paths = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if len(video_paths) == 0:
        raise ValueError(f"No video files found in folder: {folder_path}")

    all_frames = []

    for video_path in video_paths:
        print(f"Reading video: {video_path}")
        frames = read_frames_from_video(str(video_path))
        all_frames.extend(frames)

    if len(all_frames) == 0:
        raise ValueError(f"No frames loaded from video folder: {folder_path}")

    return all_frames


def load_input_frames(input_path: str) -> List[np.ndarray]:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if path.is_file():
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            return read_frames_from_video(str(path))

        if path.suffix.lower() in IMAGE_EXTENSIONS:
            image = cv2.imread(str(path))

            if image is None:
                raise ValueError(f"Cannot read image: {path}")

            return [image]

        raise ValueError(f"Unsupported input file type: {path.suffix}")

    image_paths = sorted(
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(image_paths) > 0:
        return read_frames_from_image_folder(str(path))

    video_paths = sorted(
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if len(video_paths) > 0:
        return read_frames_from_video_folder(str(path))

    raise ValueError(
        f"No supported images/videos found in: {input_path}\n"
        f"Supported images: {IMAGE_EXTENSIONS}\n"
        f"Supported videos: {VIDEO_EXTENSIONS}"
    )


def draw_anomaly_label(
    frame: np.ndarray,
    score: float,
    threshold: float,
    anomaly: bool,
) -> np.ndarray:
    output = frame.copy()

    label_score = f"Score: {score:.6f}"
    label_threshold = f"Threshold: {threshold:.6f}"
    label_status = "ANOMALY" if anomaly else "NORMAL"

    color = (0, 0, 255) if anomaly else (0, 255, 0)

    cv2.rectangle(
        output,
        (10, 10),
        (430, 110),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        output,
        label_score,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    cv2.putText(
        output,
        label_threshold,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    cv2.putText(
        output,
        label_status,
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

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
            fieldnames=[
                "frame_index",
                "anomaly_score",
                "threshold",
                "is_anomaly",
            ],
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)

    print(f"CSV saved to: {csv_path}")


def save_score_plot(results, output_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skip score plot.")
        return

    frame_indexes = [row["frame_index"] for row in results]
    scores = [row["anomaly_score"] for row in results]
    thresholds = [row["threshold"] for row in results]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(frame_indexes, scores, label="Anomaly score")
    plt.plot(frame_indexes, thresholds, linestyle="--", label="Threshold")
    plt.xlabel("Frame index")
    plt.ylabel("Reconstruction error")
    plt.title("Anomaly Score per Frame")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Score plot saved to: {output_path}")


def save_summary_txt(results, output_path: Path):
    total_frames = len(results)
    anomaly_count = sum(row["is_anomaly"] for row in results)
    normal_count = total_frames - anomaly_count

    scores = [row["anomaly_score"] for row in results]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("Inference Summary\n")
        f.write("=================\n")
        f.write(f"Total frames   : {total_frames}\n")
        f.write(f"Normal frames  : {normal_count}\n")
        f.write(f"Anomaly frames : {anomaly_count}\n")
        f.write(f"Min score      : {min(scores):.8f}\n")
        f.write(f"Max score      : {max(scores):.8f}\n")
        f.write(f"Mean score     : {float(np.mean(scores)):.8f}\n")
        f.write(f"Std score      : {float(np.std(scores)):.8f}\n")
        f.write(f"Threshold      : {results[0]['threshold']:.8f}\n")

    print(f"Summary saved to: {output_path}")


def load_ground_truth_labels(labels_csv: str) -> dict:
    path = Path(labels_csv)
    if not path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {path}")

    labels = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"frame_index", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Labels CSV must contain columns: frame_index,label"
            )

        for row in reader:
            index = int(row["frame_index"])
            labels[index] = int(row["label"])

    return labels


def run_inference(
    input_path: str,
    model_path: str,
    output_dir: str,
    threshold: Optional[float] = None,
    auto_threshold_percentile: float = 95.0,
    save_frames: bool = False,
    labels_csv: Optional[str] = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, img_size, device = load_model(model_path)
    frames = load_input_frames(input_path)

    print("=" * 60)
    print("Running Inference")
    print("=" * 60)
    print(f"Device       : {device}")
    print(f"Input        : {input_path}")
    print(f"Model        : {model_path}")
    print(f"Frames loaded: {len(frames)}")
    print(f"Image size   : {img_size}x{img_size}")
    print("=" * 60)

    scores = []
    reconstructed_tensors = []

    for index, frame in enumerate(frames):
        tensor = preprocess_video_frame(frame, img_size=img_size)
        score, reconstructed = predict_tensor(tensor, model, device)

        scores.append(score)
        reconstructed_tensors.append(reconstructed)

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(frames)} frames")

    if threshold is None:
        threshold = float(np.percentile(scores, auto_threshold_percentile))
        print(
            f"Auto threshold = {threshold:.8f} "
            f"(percentile={auto_threshold_percentile})"
        )
    else:
        threshold = float(threshold)
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
            marked = draw_anomaly_label(
                frame=frame,
                score=score,
                threshold=threshold,
                anomaly=anomaly,
            )

            cv2.imwrite(
                str(marked_dir / f"frame_{index:05d}.jpg"),
                marked,
            )

            reconstructed_image = tensor_to_uint8_image(
                reconstructed_tensors[index]
            )

            cv2.imwrite(
                str(reconstructed_dir / f"reconstructed_{index:05d}.jpg"),
                reconstructed_image,
            )

    save_results_csv(
        results,
        output_dir / "anomaly_scores.csv",
    )

    save_score_plot(
        results,
        output_dir / "anomaly_score_plot.png",
    )

    save_summary_txt(
        results,
        output_dir / "summary.txt",
    )

    if labels_csv is not None:
        ground_truth = load_ground_truth_labels(labels_csv)
        y_true = []
        y_scores = []
        y_pred = []

        for row in results:
            idx = row["frame_index"]
            if idx not in ground_truth:
                continue
            y_true.append(ground_truth[idx])
            y_scores.append(row["anomaly_score"])
            y_pred.append(int(is_anomaly(row["anomaly_score"], threshold)))

        if len(y_true) == 0:
            print("Warning: No matching ground truth labels found for evaluation.")
        else:
            evaluate_model(
                np.array(y_true, dtype=int),
                np.array(y_pred, dtype=int),
                np.array(y_scores, dtype=float),
            )

    anomaly_count = sum(row["is_anomaly"] for row in results)
    normal_count = len(results) - anomaly_count

    print("=" * 60)
    print("Inference completed.")
    print(f"Total frames  : {len(results)}")
    print(f"Anomaly frames: {anomaly_count}")
    print(f"Normal frames : {normal_count}")
    print(f"Output folder : {output_dir}")
    print("=" * 60)

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
        labels_csv=args.labels_csv,
    )


if __name__ == "__main__":
    main()