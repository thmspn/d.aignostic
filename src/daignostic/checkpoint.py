"""Save and load training checkpoints."""

import os
from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, epoch, extra=None):
    """Write atomically. `epoch` = number of epochs completed so far."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "extra": extra or {},
        },
        tmp,
    )
    os.replace(tmp, path)           # rename: a cut-off never leaves a half-written file


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    """Load weights (and optimizer state) in place. Returns the saved epoch."""
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state["epoch"]