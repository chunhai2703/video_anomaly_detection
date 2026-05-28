import torch
import torch.nn.functional as F
from src.models.autoencoder import AutoEncoder

def load_model(model_path="outputs/model.pth"):
    """
    Load pretrained AutoEncoder
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(frame, model):
    """
    Predict anomaly score for a single frame
    frame: torch tensor [1,H,W] or [1,1,H,W], normalized [0,1]
    returns: anomaly_score (MSE loss), reconstructed frame
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if frame.dim() == 3:  # [1,H,W] -> [1,1,H,W]
        frame = frame.unsqueeze(0)
    frame = frame.to(device, dtype=torch.float)

    with torch.no_grad():
        reconstructed = model(frame)
        loss = F.mse_loss(reconstructed, frame)
    
    return loss.item(), reconstructed