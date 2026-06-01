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
    
def preprocess_video_frame(frame, img_size=128):
    """
    Resize và convert frame video sang tensor [1,H,W] normalized.
    """
    if frame is None:
        raise ValueError("Input frame is None.")
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    tensor = torch.from_numpy(frame).unsqueeze(0)
    return tensor

def tensor_to_uint8_image(tensor):
    """
    Convert tensor [1,H,W] hoặc [1,1,H,W] sang uint8 image.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    image = tensor.detach().cpu().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image