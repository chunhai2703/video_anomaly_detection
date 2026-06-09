"""Model loading and fast batched prediction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from src.models.autoencoder import AutoEncoder
from typing import List, Dict
import numpy as np

from src.preprocessing import preprocess_batch
from src.optical_flow.motion_map import compute_motion_scores_for_sequence
from src.evaluation.threshold import apply_threshold


def configure_torch_threads(num_threads: int | None = None) -> None:
    """Avoid CPU oversubscription, a common cause of very slow Streamlit inference."""
    if torch.cuda.is_available():
        return
    if num_threads is None:
        num_threads = min(4, max(1, torch.get_num_threads()))
    torch.set_num_threads(max(1, int(num_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def get_device(prefer_gpu: bool = True) -> torch.device:
    configure_torch_threads()
    return torch.device("cuda" if prefer_gpu and torch.cuda.is_available() else "cpu")


def load_model(model_path: str | Path = "outputs/model.pth", device: torch.device | None = None) -> Tuple[AutoEncoder, int, torch.device]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {path.resolve()}")

    device = device or get_device()
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    img_size = int(checkpoint.get("img_size", 128)) if isinstance(checkpoint, dict) else 128

    model = AutoEncoder().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Tối ưu nhẹ cho inference CPU/GPU hiện đại.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    return model, img_size, device


@torch.inference_mode()
def predict_batch(batch: torch.Tensor, model: AutoEncoder, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-frame MSE scores and reconstructions for a tensor batch."""
    batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)
    recon = model(batch)
    scores = torch.mean((recon - batch) ** 2, dim=(1, 2, 3))
    return scores.detach().cpu(), recon.detach().cpu()


def infer_frames(
    frames: List[np.ndarray],
    model: AutoEncoder,
    img_size: int,
    device: torch.device,
    batch_size: int = 32,
    threshold: float = 0.5,
    alpha: float = 0.6,
) -> Dict[str, np.ndarray]:
    """
    Run inference on a list of raw frames and return reconstruction scores,
    motion scores, combined scores, and binary predictions.

    - `frames`: list of BGR frames (numpy arrays)
    - returns dict with keys: `recon_scores`, `motion_scores`, `combined_scores`, `predictions`
    """
    # Preprocess all frames into a tensor batch of shape (N, 1, H, W)
    batches = []
    N = len(frames)
    recon_scores_list = []

    # Process in minibatches
    for i in range(0, N, batch_size):
        batch_frames = frames[i : i + batch_size]
        batch_tensor = preprocess_batch(batch_frames, img_size)
        scores, _recon = predict_batch(batch_tensor, model, device)
        recon_scores_list.append(scores.numpy())

    recon_scores = np.concatenate(recon_scores_list, axis=0)

    # Motion scores are computed on consecutive pairs; pad to match frame count
    motion_pair_scores = compute_motion_scores_for_sequence(frames)
    if len(motion_pair_scores) == 0:
        motion_scores = np.zeros_like(recon_scores)
    else:
        motion_scores = np.concatenate([motion_pair_scores, motion_pair_scores[-1:]])

    # Compute binary predictions using combined score (backward-compatible)
    predictions = apply_threshold(recon_scores, threshold, motion_scores=motion_scores, alpha=alpha)

    return {
        "recon_scores": recon_scores,
        "motion_scores": motion_scores,
        "predictions": predictions,
    }
