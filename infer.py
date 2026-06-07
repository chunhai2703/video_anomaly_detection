"""Fast inference entrypoint for Video Anomaly Detection.

Key speed choices:
- streaming video decoding instead of loading the full video into RAM;
- batched PyTorch inference;
- frame stride sampling for long videos;
- no per-frame image saving unless explicitly requested;
- top-k frame rendering for UI/reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import torch

from src.models.predictor import load_model, predict_batch
from src.preprocessing import preprocess_batch, resize_keep_aspect
from src.utils.video import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, iter_video_frames, probe_video

ProgressCallback = Callable[[int, int | None], None]


@dataclass
class FrameResult:
    frame_index: int
    time_seconds: float
    score: float
    threshold: float
    is_anomaly: bool
    frame_path: str | None = None


def _safe_fps(video_path: str | Path, stride: int) -> float:
    try:
        return max(1.0, probe_video(video_path).fps / max(1, stride))
    except Exception:
        return 25.0


def _normalize_heatmap(error_map: np.ndarray) -> np.ndarray:
    error_map = cv2.GaussianBlur(error_map.astype(np.float32), (0, 0), 3)
    return cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def draw_overlay(frame: np.ndarray, score: float, threshold: float, reconstruction: np.ndarray | None = None) -> np.ndarray:
    """Draw label, heatmap, and bounding boxes on an original BGR frame."""
    out = frame.copy()
    anomaly = score > threshold
    color = (0, 0, 255) if anomaly else (0, 200, 0)

    if anomaly and reconstruction is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        recon = reconstruction.squeeze()
        recon = cv2.resize(recon, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        if recon.max() <= 1.0:
            recon = recon * 255.0
        error = np.abs(gray.astype(np.float32) - recon.astype(np.float32))
        heat = _normalize_heatmap(error)
        colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        mask = heat >= np.percentile(heat, 97.5)
        out[mask] = cv2.addWeighted(out, 0.45, colored, 0.55, 0)[mask]

        binary = (mask.astype(np.uint8) * 255)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 120:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= 10 and h >= 10:
                cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

    label = "ANOMALY" if anomaly else "NORMAL"
    cv2.rectangle(out, (8, 8), (360, 88), (0, 0, 0), -1)
    cv2.putText(out, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(out, f"score={score:.6f}", (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(out, f"threshold={threshold:.6f}", (18, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    return out


def _iter_single_image(path: str | Path) -> Iterable[tuple[int, np.ndarray]]:
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    yield 0, frame


def _frame_iterator(input_path: str | Path, stride: int, max_frames: int | None = None):
    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        yield from iter_video_frames(path, stride=stride, max_frames=max_frames)
    elif suffix in IMAGE_EXTENSIONS:
        yield from _iter_single_image(path)
    else:
        raise ValueError(f"Định dạng không hỗ trợ: {suffix}")


def _score_pass(
    input_path: str | Path,
    model,
    device: torch.device,
    img_size: int,
    batch_size: int,
    stride: int,
    max_frames: int | None,
    progress_callback: ProgressCallback | None,
) -> list[tuple[int, float]]:
    pairs: list[tuple[int, float]] = []
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []
    total = None
    if Path(input_path).suffix.lower() in VIDEO_EXTENSIONS:
        try:
            info = probe_video(input_path)
            total = int(np.ceil(info.frame_count / max(1, stride))) if info.frame_count else None
        except Exception:
            total = None

    def flush() -> None:
        nonlocal batch_frames, batch_indices
        if not batch_frames:
            return
        tensors = preprocess_batch(batch_frames, img_size=img_size)
        scores, _ = predict_batch(tensors, model, device)
        pairs.extend((idx, float(score)) for idx, score in zip(batch_indices, scores.tolist()))
        batch_frames = []
        batch_indices = []
        if progress_callback:
            progress_callback(len(pairs), total)

    for frame_index, frame in _frame_iterator(input_path, stride=stride, max_frames=max_frames):
        batch_indices.append(frame_index)
        batch_frames.append(frame)
        if len(batch_frames) >= batch_size:
            flush()
    flush()
    return pairs


def _render_pass(
    input_path: str | Path,
    model,
    device: torch.device,
    img_size: int,
    batch_size: int,
    stride: int,
    max_frames: int | None,
    scores_by_index: dict[int, float],
    threshold: float,
    output_dir: Path,
    top_k: int,
    save_video: bool,
    save_frames: bool,
    max_display_width: int | None,
) -> tuple[list[FrameResult], str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "top_frames"
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    top_indices = set(
        idx for idx, _ in sorted(scores_by_index.items(), key=lambda item: item[1], reverse=True)[:max(1, top_k)]
    )
    anomaly_indices = {idx for idx, score in scores_by_index.items() if score > threshold}
    indices_to_render = top_indices | anomaly_indices if save_frames else top_indices

    writer = None
    video_path = None
    if save_video and Path(input_path).suffix.lower() in VIDEO_EXTENSIONS:
        video_path = str(output_dir / "annotated_output.mp4")

    results: list[FrameResult] = []
    pending_frames: list[np.ndarray] = []
    pending_indices: list[int] = []

    fps = _safe_fps(input_path, stride)

    def flush() -> None:
        nonlocal pending_frames, pending_indices, writer
        if not pending_frames:
            return
        tensors = preprocess_batch(pending_frames, img_size=img_size)
        _, recon = predict_batch(tensors, model, device)
        for local_i, (frame_index, frame) in enumerate(zip(pending_indices, pending_frames)):
            score = scores_by_index[frame_index]
            frame_for_output = resize_keep_aspect(frame, max_display_width)
            recon_np = recon[local_i].numpy()
            annotated = draw_overlay(frame_for_output, score, threshold, recon_np)

            if save_video:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
                writer.write(annotated)

            frame_path = None
            if save_frames and frame_index in indices_to_render:
                frame_path = str(frames_dir / f"frame_{frame_index:06d}_score_{score:.6f}.jpg")
                cv2.imwrite(frame_path, annotated)

            results.append(
                FrameResult(
                    frame_index=frame_index,
                    time_seconds=float(frame_index / max(fps * stride, 1e-9)),
                    score=float(score),
                    threshold=float(threshold),
                    is_anomaly=bool(score > threshold),
                    frame_path=frame_path,
                )
            )
        pending_frames = []
        pending_indices = []

    for frame_index, frame in _frame_iterator(input_path, stride=stride, max_frames=max_frames):
        if frame_index not in indices_to_render and not save_video:
            score = scores_by_index[frame_index]
            results.append(
                FrameResult(frame_index, float(frame_index / max(fps * stride, 1e-9)), float(score), float(threshold), bool(score > threshold), None)
            )
            continue
        pending_indices.append(frame_index)
        pending_frames.append(frame)
        if len(pending_frames) >= batch_size:
            flush()
    flush()

    if writer is not None:
        writer.release()

    results.sort(key=lambda r: r.frame_index)
    return results, video_path


def run_inference(
    input_path: str | Path,
    model_path: str | Path = "outputs/model.pth",
    output_dir: str | Path = "outputs/inference",
    threshold: float | None = None,
    auto_threshold_percentile: float = 95.0,
    batch_size: int = 64,
    stride: int = 2,
    max_frames: int | None = None,
    top_k: int = 24,
    save_frames: bool = True,
    save_video: bool = False,
    max_display_width: int | None = 960,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    output_dir = Path(output_dir)
    model, img_size, device = load_model(model_path)

    pairs = _score_pass(
        input_path=input_path,
        model=model,
        device=device,
        img_size=img_size,
        batch_size=batch_size,
        stride=stride,
        max_frames=max_frames,
        progress_callback=progress_callback,
    )
    if not pairs:
        raise ValueError("Không tìm thấy frame hợp lệ trong input.")

    frame_indices = [idx for idx, _ in pairs]
    scores = np.array([score for _, score in pairs], dtype=np.float32)
    threshold = float(threshold) if threshold is not None else float(np.percentile(scores, auto_threshold_percentile))
    scores_by_index = dict(pairs)

    results, video_path = _render_pass(
        input_path=input_path,
        model=model,
        device=device,
        img_size=img_size,
        batch_size=batch_size,
        stride=stride,
        max_frames=max_frames,
        scores_by_index=scores_by_index,
        threshold=threshold,
        output_dir=output_dir,
        top_k=top_k,
        save_video=save_video,
        save_frames=save_frames,
        max_display_width=max_display_width,
    )

    rows = [asdict(r) for r in results]
    csv_path = output_dir / "scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input_path": str(input_path),
        "model_path": str(model_path),
        "device": str(device),
        "img_size": img_size,
        "stride": stride,
        "batch_size": batch_size,
        "processed_frames": len(scores),
        "frame_indices": frame_indices,
        "threshold": threshold,
        "auto_threshold_percentile": auto_threshold_percentile,
        "min_score": float(np.min(scores)),
        "mean_score": float(np.mean(scores)),
        "max_score": float(np.max(scores)),
        "p95_score": float(np.percentile(scores, 95)),
        "p99_score": float(np.percentile(scores, 99)),
        "anomaly_frames": int(np.sum(scores > threshold)),
        "scores_csv": str(csv_path),
        "annotated_video": video_path,
        "results": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def detect_anomalies(**kwargs) -> list[dict]:
    """Backward-compatible wrapper for the Streamlit dashboard."""
    summary = run_inference(**kwargs)
    output = []
    for item in summary["results"]:
        row = dict(item)
        row["score"] = row.pop("score")
        if row.get("frame_path"):
            row["frame"] = cv2.imread(row["frame_path"])
        output.append(row)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast anomaly inference")
    parser.add_argument("--input", required=True, help="Video/image path")
    parser.add_argument("--model_path", default="outputs/model.pth")
    parser.add_argument("--output_dir", default="outputs/inference")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--auto_threshold_percentile", type=float, default=95.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=2, help="Process every Nth frame. Increase for faster long videos.")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=24)
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--max_display_width", type=int, default=960)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = vars(args)
    kwargs["input_path"] = kwargs.pop("input")
    summary = run_inference(**kwargs)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
