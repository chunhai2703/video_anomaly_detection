from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

class FrameDataset(Dataset):

    def __init__(self, root_dir, img_size=128):
        self.root_dir = Path(root_dir)
        self.img_size = img_size

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.root_dir}")

        self.image_paths = sorted(
            path for path in self.root_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No image files found in: {self.root_dir}\n"
                f"Supported extensions: {IMAGE_EXTENSIONS}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        image = cv2.resize(
            image,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_AREA
        )

        image = image.astype(np.float32) / 255.0

        tensor = torch.from_numpy(image).unsqueeze(0)

        return tensor