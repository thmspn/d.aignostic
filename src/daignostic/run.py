"""Train from a config file:  python -m daignostic.run path/to/config.json"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from daignostic.checkpoint import load_checkpoint, save_checkpoint
from daignostic.config import load_config, save_config
from daignostic.dataset import ChestXrayDataset, get_transform, index_images
from daignostic.model import build_model
from daignostic.reproducibility import set_seed
from daignostic.splits import make_splits, read_image_list
from daignostic.train import compute_pos_weight, predict_probs, train_one_epoch


def build_loaders(config):
    root = Path(config.data_root)
    df = pd.read_csv(root / "Data_Entry_2017.csv")
    paths = index_images(root)
    df = df[df["Image Index"].isin(paths)]          # only images that exist here
    if df.empty:
        raise ValueError(f"No images found under {root}")
    train_df, val_df, _test_df = make_splits(       # test: never used in training
        df,
        read_image_list(root / "train_val_list.txt"),
        read_image_list(root / "test_list.txt"),
        config.val_fraction,
        config.seed,
    )
    if config.max_images is not None:
        train_df = train_df.head(config.max_images)
        val_df = val_df.head(config.max_images)
    transform = get_transform(config.image_size)
    train_ds = ChestXrayDataset(train_df, paths, transform)
    val_ds = ChestXrayDataset(val_df, paths, transform)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              drop_last=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers)
    return train_loader, val_loader, train_ds.labels

def trim_log(log_path, last_epoch):
    """Keep only log lines for epochs the checkpoint actually contains."""
    if not log_path.exists():
        return
    kept = []
    for line in log_path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue                          # half-written line from a cut-off
        if record["epoch"] <= last_epoch:
            kept.append(line)
    log_path.write_text("".join(line + "\n" for line in kept))

def run_training(config):
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if config_path.exists():
        old = load_config(config_path)
        if replace(old, epochs=config.epochs) != config:
            raise ValueError(f"{out} holds a run with a different config; use a new output_dir")
    save_config(config, config_path)

    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, train_labels = build_loaders(config)
    model = build_model(pretrained=config.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    pos_weight = compute_pos_weight(train_labels).to(device)    # training labels only
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    start = 0
    if (out / "last.pt").exists():
        start = load_checkpoint(out / "last.pt", model, optimizer, map_location=device)
        print(f"Resuming after epoch {start}")

    trim_log(out / "log.jsonl", start)
    
    for epoch in range(start, config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        probs, labels = predict_probs(model, val_loader, device)
        val_loss = F.binary_cross_entropy(probs, labels).item()
        record = {"epoch": epoch + 1, "train_loss": round(train_loss, 4),
                  "val_loss": round(val_loss, 4)}
        print(record)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        save_checkpoint(out / "last.pt", model, optimizer, epoch + 1)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    run_training(load_config(parser.parse_args().config))

if __name__ == "__main__":
    main()