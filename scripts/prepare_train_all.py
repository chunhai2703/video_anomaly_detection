from pathlib import Path
import shutil


def copy_train_folder(source: Path, destination: Path):
    if not source.exists():
        print(f"Skip, folder not found: {source}")
        return

    if destination.exists():
        shutil.rmtree(destination)

    shutil.copytree(source, destination)
    print(f"Copied: {source} -> {destination}")


def main():
    dataset_dir = Path("dataset")

    ped1_train = dataset_dir / "UCSDped1" / "Train"
    ped2_train = dataset_dir / "UCSDped2" / "Train"

    train_all_dir = dataset_dir / "train_all"
    train_all_dir.mkdir(parents=True, exist_ok=True)

    copy_train_folder(ped1_train, train_all_dir / "UCSDped1_Train")
    copy_train_folder(ped2_train, train_all_dir / "UCSDped2_Train")

    print("\nDone.")
    print(f"Train all folder: {train_all_dir}")


if __name__ == "__main__":
    main()