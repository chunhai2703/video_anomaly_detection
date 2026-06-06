import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.models.trainer import train_model


def plot_training_loss(history, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    loss = history.get("loss", [])

    if len(loss) == 0:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    save_path = save_dir / "training_loss.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Training loss plot saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/train_all"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="outputs/model.pth"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10
    )

    parser.add_argument(
        "--resume_path",
        type=str,
        default=None
    )

    return parser.parse_args()


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
        checkpoint_every=args.checkpoint_every,
        resume_path=args.resume_path,
    )

    output_dir = Path(args.save_path).parent
    plot_training_loss(history, output_dir)


if __name__ == "__main__":
    main()