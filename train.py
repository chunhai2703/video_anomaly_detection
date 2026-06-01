import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.models.trainer import train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AutoEncoder for Video Anomaly Detection"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/train_all",
        help="Folder containing training frames."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size."
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=128,
        help="Image size."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="outputs/model.pth",
        help="Path to save model."
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers. Use 0 on Windows."
    )

    return parser.parse_args()


def save_loss_plot(losses, save_path="outputs/training_loss.png"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Training loss plot saved to: {save_path}")


def main():
    args = parse_args()

    history = train_model(
        data_dir=args.data_dir,
        save_path=args.save_path,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
    )

    save_loss_plot(history["loss"])


if __name__ == "__main__":
    main()