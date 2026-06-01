from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.autoencoder import AutoEncoder
from src.models.dataset import FrameDataset


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    data_dir,
    save_path="outputs/model.pth",
    img_size=128,
    epochs=20,
    batch_size=32,
    learning_rate=0.001,
    num_workers=0,
):
    device = get_device()

    dataset = FrameDataset(
        root_dir=data_dir,
        img_size=img_size
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    model = AutoEncoder().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = {
        "loss": []
    }

    print("=" * 60)
    print("Training AutoEncoder")
    print("=" * 60)
    print(f"Device          : {device}")
    print(f"Dataset folder  : {data_dir}")
    print(f"Training images : {len(dataset)}")
    print(f"Image size      : {img_size}x{img_size}")
    print(f"Epochs          : {epochs}")
    print(f"Batch size      : {batch_size}")
    print(f"Learning rate   : {learning_rate}")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0

        for frames in loader:
            frames = frames.to(device, dtype=torch.float32)

            reconstructed = model(frames)
            loss = criterion(reconstructed, frames)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * frames.size(0)

        epoch_loss = total_loss / len(dataset)
        history["loss"].append(epoch_loss)

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] "
            f"Loss: {epoch_loss:.8f}"
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "img_size": img_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_loss": history["loss"][-1],
    }

    torch.save(checkpoint, save_path)

    print("=" * 60)
    print(f"Model saved to: {save_path}")
    print("=" * 60)

    return history