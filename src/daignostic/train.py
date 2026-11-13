"""Training and prediction helpers."""

import torch


def compute_pos_weight(labels):
    """pos_weight for BCEWithLogitsLoss: (#negatives / #positives) per finding.

    Compute from TRAINING labels only. A finding with no positives is clamped to 1.
    """
    positives = labels.sum(dim=0).clamp(min=1)
    negatives = len(labels) - labels.sum(dim=0)
    return negatives / positives


def train_one_epoch(model, loader, optimizer, loss_fn, device="cpu"):
    """One pass over the loader. Returns the mean loss per image."""
    model.train()
    total_loss, n_images = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)   # undo the per-batch mean
        n_images += len(images)
    return total_loss / n_images


@torch.no_grad()
def predict_probs(model, loader, device="cpu"):
    """Return (probabilities, labels) for the whole loader, as CPU tensors."""
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels)
    return torch.cat(all_probs), torch.cat(all_labels)