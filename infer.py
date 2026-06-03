import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from src.models.dataset import preprocess_video_frame, tensor_to_uint8_image
from src.models.predictor import load_model, predict_tensor, is_anomaly

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".flv", ".gif"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def parse_args():
    parser = argparse.ArgumentParser(description="Run anomaly inference on video or frame folder")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="outputs/model.pth")
    parser.add_argument("--output_dir", type=str, default="outputs/inference")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--auto_threshold_percentile", type=float, default=95.0)
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--labels_csv", type=str, default=None)
    return parser.parse_args()

def load_frames(input_path: str):
    path = Path(input_path)
    frames = []

    if path.is_file():
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            cap = cv2.VideoCapture(str(path))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            frame = cv2.imread(str(path))
            if frame is not None:
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frames.append(frame)

    elif path.is_dir():
        image_files = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        for f in image_files:
            frame = cv2.imread(str(f))
            if frame is not None:
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frames.append(frame)

        video_files = sorted(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
        for vf in video_files:
            cap = cv2.VideoCapture(str(vf))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

    return frames

def prepare_reconstructed_array(reconstructed, frame):
    reconstructed_np = reconstructed.detach().cpu().numpy()

    if reconstructed_np.ndim == 4:
        reconstructed_np = reconstructed_np[0]

    if reconstructed_np.ndim == 3:
        if reconstructed_np.shape[0] in [1, 3] and reconstructed_np.shape[-1] not in [1, 3]:
            reconstructed_np = np.transpose(reconstructed_np, (1, 2, 0))

    if reconstructed_np.ndim == 2:
        reconstructed_np = reconstructed_np[:, :, np.newaxis]

    if reconstructed_np.ndim == 1:
        side = int(np.sqrt(reconstructed_np.shape[0]))
        if side * side == reconstructed_np.shape[0]:
            reconstructed_np = reconstructed_np.reshape(side, side, 1)
        else:
            reconstructed_np = np.resize(reconstructed_np, (frame.shape[0], frame.shape[1], 1))

    if reconstructed_np.ndim != 3:
        reconstructed_np = np.squeeze(reconstructed_np)
        if reconstructed_np.ndim == 2:
            reconstructed_np = reconstructed_np[:, :, np.newaxis]
        else:
            reconstructed_np = np.resize(reconstructed_np, (frame.shape[0], frame.shape[1], 1))

    if reconstructed_np.shape[2] > 3:
        reconstructed_np = np.mean(reconstructed_np, axis=2, keepdims=True)

    reconstructed_np = cv2.resize(reconstructed_np, (frame.shape[1], frame.shape[0]))

    if reconstructed_np.ndim == 2:
        reconstructed_np = reconstructed_np[:, :, np.newaxis]

    if frame.ndim == 3 and frame.shape[2] == 3 and reconstructed_np.ndim == 3 and reconstructed_np.shape[2] == 1:
        reconstructed_np = np.repeat(reconstructed_np, 3, axis=2)

    if frame.ndim == 3 and frame.shape[2] == 3 and reconstructed_np.ndim == 3 and reconstructed_np.shape[2] > 3:
        reconstructed_np = reconstructed_np[:, :, :3]

    return reconstructed_np.astype(np.float32)

def draw_anomaly_label(frame: np.ndarray, anomaly_map: np.ndarray, score: float, threshold: float, anomaly: bool) -> np.ndarray:
    output = frame.copy()
    color = (0, 0, 255) if anomaly else (0, 255, 0)

    if anomaly:
        if anomaly_map.ndim == 3:
            heatmap = np.mean(anomaly_map, axis=2)
        else:
            heatmap = anomaly_map

        heatmap = heatmap.astype(np.float32)
        heatmap = cv2.GaussianBlur(heatmap, (9, 9), 0)
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        pixel_threshold = np.percentile(heatmap_norm, 97)
        _, thresh_img = cv2.threshold(heatmap_norm, pixel_threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((7, 7), np.uint8)
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < 300:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if w < 15 or h < 15:
                continue

            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.rectangle(output, (10, 10), (430, 110), (0, 0, 0), -1)
    cv2.putText(output, f"Score:{score:.6f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(output, f"Threshold:{threshold:.6f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(output, "ANOMALY" if anomaly else "NORMAL", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return output
def run_inference(input_path, model_path, output_dir, threshold=None, auto_threshold_percentile=95.0, save_frames=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, img_size, device = load_model(model_path)
    frames = load_frames(input_path)

    if len(frames) == 0:
        raise ValueError(f"No valid frames found in input: {input_path}")

    scores = []
    reconstructed_tensors = []

    for frame in frames:
        tensor = preprocess_video_frame(frame, img_size=img_size)
        score, reconstructed = predict_tensor(tensor, model, device)
        scores.append(float(score))
        reconstructed_tensors.append(reconstructed)

    if threshold is None:
        threshold = float(np.percentile(scores, auto_threshold_percentile))

    results = []
    marked_dir = output_dir / "marked_frames"
    reconstructed_dir = output_dir / "reconstructed_frames"

    if save_frames:
        marked_dir.mkdir(parents=True, exist_ok=True)
        reconstructed_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame in enumerate(frames):
        score = scores[idx]
        reconstructed = reconstructed_tensors[idx]
        reconstructed_np = prepare_reconstructed_array(reconstructed, frame)

        frame_float = frame.astype(np.float32) / 255.0

        if reconstructed_np.max() > 1.0:
            reconstructed_float = reconstructed_np / 255.0
        else:
            reconstructed_float = reconstructed_np

        if frame_float.shape != reconstructed_float.shape:
            if reconstructed_float.ndim == 2:
                reconstructed_float = reconstructed_float[:, :, np.newaxis]
            if frame_float.ndim == 3 and reconstructed_float.ndim == 3 and reconstructed_float.shape[2] == 1:
                reconstructed_float = np.repeat(reconstructed_float, frame_float.shape[2], axis=2)
            reconstructed_float = reconstructed_float[:, :, :frame_float.shape[2]]

        anomaly_map = np.abs(frame_float - reconstructed_float)
        anomaly = is_anomaly(score, threshold)

        results.append({
            "frame_index": idx,
            "anomaly_score": score,
            "threshold": threshold,
            "is_anomaly": int(anomaly)
        })

        if save_frames:
            marked = draw_anomaly_label(frame, anomaly_map, score, threshold, anomaly)
            cv2.imwrite(str(marked_dir / f"frame_{idx:05d}.jpg"), marked)

            reconstructed_img = tensor_to_uint8_image(reconstructed)
            cv2.imwrite(str(reconstructed_dir / f"reconstructed_{idx:05d}.jpg"), reconstructed_img)

    return results

def detect_anomalies(input_path):
    model_path = "outputs/model.pth"
    output_dir = Path("outputs/inference")

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    results = run_inference(input_path, model_path, output_dir)

    frames = []
    scores = []
    marked_dir = output_dir / "marked_frames"

    for row in results:
        idx = row["frame_index"]
        score = row["anomaly_score"]
        frame_path = marked_dir / f"frame_{idx:05d}.jpg"
        frame = cv2.imread(str(frame_path))
        frames.append(frame)
        scores.append(score)

    return list(zip(frames, scores))

def main():
    args = parse_args()

    run_inference(
        input_path=args.input,
        model_path=args.model_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        auto_threshold_percentile=args.auto_threshold_percentile,
        save_frames=args.save_frames
    )

if __name__ == "__main__":
    main()