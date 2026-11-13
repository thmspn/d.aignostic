"""Training configuration, loaded from a JSON file."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    seed: int = 0
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-4
    val_fraction: float = 0.1
    image_size: int = 224
    pretrained: bool = True
    num_workers: int = 0
    max_images: int | None = None      # cap per split, for quick debugging
    data_root: str = "data/nih"
    output_dir: str = "runs/debug"


def load_config(path):
    """Read a JSON file into a TrainConfig. Unknown keys raise TypeError."""
    return TrainConfig(**json.loads(Path(path).read_text()))


def save_config(config, path):
    Path(path).write_text(json.dumps(asdict(config), indent=2))