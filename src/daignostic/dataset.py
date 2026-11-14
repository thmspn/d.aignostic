"""PyTorch Dataset for NIH ChestX-ray14 (images are loaded lazily)."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from daignostic.labels import nih_labels_to_vector

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def index_images(root):
    """Map image filename -> full path for every PNG under root."""
    return {p.name: p for p in Path(root).rglob("*.png")}


def get_transform(image_size=IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ChestXrayDataset(Dataset):
    def __init__(self, df, image_paths, transform=None):
        names = df["Image Index"].tolist()
        # Fails here, immediately, if any image is missing (not hours into training)
        self.paths = [image_paths[name] for name in names]
        self.labels = torch.tensor(
            [nih_labels_to_vector(s) for s in df["Finding Labels"]],
            dtype=torch.float32,
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]