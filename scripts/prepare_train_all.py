from pathlib import Path
import shutil
import cv2


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def copy_train_folder(source: Path, destination: Path):
    if not source.exists():
        print(f"Skip, folder not found: {source}")
        return

    if destination.exists():
        shutil.rmtree(destination)

    shutil.copytree(source, destination)
    print(f"Copied: {source} -> {destination}")


def extract_videos_to_frames(source: Path, destination: Path):
    if not source.exists():
        print(f"Skip, folder not found: {source}")
        return

    video_paths = sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if len(video_paths) == 0:
        print(f"No videos found in: {source}")
        return

    if destination.exists():
        shutil.rmtree(destination)

    destination.mkdir(parents=True, exist_ok=True)

    for video_path in video_paths:
        clip_name = video_path.stem
        clip_output = destination / clip_name
        clip_output.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            continue

        frame_index = 1

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_path = clip_output / f"{frame_index:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_index += 1

        cap.release()

        print(f"Extracted {frame_index - 1} frames: {video_path} -> {clip_output}")


def copy_image_folder(source: Path, destination: Path):
    if not source.exists():
        print(f"Skip, folder not found: {source}")
        return

    image_paths = sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(image_paths) == 0:
        print(f"No image files found in: {source}")
        return

    if destination.exists():
        shutil.rmtree(destination)

    destination.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        relative_path = image_path.relative_to(source)
        target_path = destination / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_path)

    print(f"Copied {len(image_paths)} images: {source} -> {destination}")


def main():
    dataset_dir = Path("dataset")

    ped1_train = dataset_dir / "UCSDped1" / "Train"
    ped2_train = dataset_dir / "UCSDped2" / "Train"

    avenue_dir = dataset_dir / "Avenue Dataset"
    avenue_training_videos = avenue_dir / "training_videos"
    avenue_training_vol = avenue_dir / "training_vol"

    train_all_dir = dataset_dir / "train_all"
    train_all_dir.mkdir(parents=True, exist_ok=True)

    copy_train_folder(ped1_train, train_all_dir / "UCSDped1_Train")
    copy_train_folder(ped2_train, train_all_dir / "UCSDped2_Train")

    extract_videos_to_frames(
        avenue_training_videos,
        train_all_dir / "Avenue_Training_Videos"
    )

    copy_image_folder(
        avenue_training_vol,
        train_all_dir / "Avenue_Training_Vol"
    )

    print("\nDone.")
    print(f"Train all folder: {train_all_dir}")


if __name__ == "__main__":
    main()