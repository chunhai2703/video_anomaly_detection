from pathlib import Path
import torch
import torch.nn.functional as F
from src.models.autoencoder import AutoEncoder


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path="outputs/model.pth"):
    """
    Load AutoEncoder model from checkpoint.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        # Try common backup locations
        search_dirs = [Path("."), Path("outputs")]
        candidates = []

        for directory in search_dirs:
            if directory.exists():
                candidates.extend(sorted(directory.rglob("*.pth")))

        if candidates:
            candidate_text = "\n".join(f"  - {p}" for p in candidates)
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Found these candidate model files instead:\n{candidate_text}\n"
                f"Please pass the correct --model_path or move the model checkpoint to {model_path}."
            )

        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please train the model first using: python train.py --save_path {model_path}"
        )

    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    img_size = checkpoint.get("img_size", 128)

    model = AutoEncoder().to(device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, img_size, device


def predict_tensor(frame_tensor, model, device):
    """
    Input: tensor [1,H,W] or [1,1,H,W]
    Output: MSE reconstruction loss + reconstructed tensor
    """
    if frame_tensor.dim() == 3:
        frame_tensor = frame_tensor.unsqueeze(0)  # add batch dim
    frame_tensor = frame_tensor.to(device, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(frame_tensor)
        score = F.mse_loss(reconstructed, frame_tensor, reduction="mean").item()

    return score, reconstructed.cpu()


def is_anomaly(score, threshold):
    """
    Determine if score > threshold
    """
    return score > threshold