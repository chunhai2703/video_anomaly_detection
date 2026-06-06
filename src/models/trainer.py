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
    checkpoint_every=10,
    resume_path=None,
):
    device = get_device()

    data_dir = Path(data_dir)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = save_path.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = FrameDataset(
        root_dir=data_dir,
        img_size=img_size
    )

    if len(dataset) == 0:
        raise ValueError(
            f"No training images found in: {data_dir}\n"
            "Please make sure your dataset folder contains image files."
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

    start_epoch = 1
    best_loss = float("inf")

    if resume_path is not None:
        resume_path = Path(resume_path)

        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print("=" * 60)
        print(f"Loading checkpoint from: {resume_path}")
        print("=" * 60)

        checkpoint = torch.load(resume_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "history" in checkpoint:
            history = checkpoint["history"]

        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1

        if "best_loss" in checkpoint:
            best_loss = checkpoint["best_loss"]
        elif "final_loss" in checkpoint:
            best_loss = checkpoint["final_loss"]

        print(f"Resume from epoch: {start_epoch}")
        print(f"Best loss so far : {best_loss:.8f}")

    print("=" * 60)
    print("Training AutoEncoder")
    print("=" * 60)
    print(f"Device          : {device}")
    print(f"Dataset folder  : {data_dir}")
    print(f"Training images : {len(dataset)}")
    print(f"Image size      : {img_size}x{img_size}")
    print(f"Epochs          : {epochs}")
    print(f"Start epoch     : {start_epoch}")
    print(f"Batch size      : {batch_size}")
    print(f"Learning rate   : {learning_rate}")
    print(f"Checkpoint every: {checkpoint_every} epoch(s)")
    print(f"Save path       : {save_path}")
    print("=" * 60)

    for epoch in range(start_epoch, epochs + 1):
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

        if epoch_loss < best_loss:
            best_loss = epoch_loss

            best_checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "img_size": img_size,
                "epoch": epoch,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "best_loss": best_loss,
                "final_loss": epoch_loss,
                "history": history,
            }

            best_path = save_path.parent / "best_model.pth"
            torch.save(best_checkpoint, best_path)

            print(f"Best model saved to: {best_path}")

        if checkpoint_every is not None and checkpoint_every > 0:
            if epoch % checkpoint_every == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "img_size": img_size,
                    "epoch": epoch,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "best_loss": best_loss,
                    "final_loss": epoch_loss,
                    "history": history,
                }

                torch.save(checkpoint, checkpoint_path)

                print(f"Checkpoint saved to: {checkpoint_path}")

    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "img_size": img_size,
        "epoch": epochs,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "best_loss": best_loss,
        "final_loss": history["loss"][-1],
        "history": history,
    }

    torch.save(final_checkpoint, save_path)

    print("=" * 60)
    print(f"Final model saved to: {save_path}")
    print(f"Best loss         : {best_loss:.8f}")
    print(f"Final loss        : {history['loss'][-1]:.8f}")
    print("=" * 60)

    return history