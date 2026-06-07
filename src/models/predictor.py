"""Model loading and fast batched prediction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from src.models.autoencoder import AutoEncoder


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
