import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from daignostic.train import compute_pos_weight, predict_probs, train_one_epoch


def tiny_setup(n=16, seed=0):
    torch.manual_seed(seed)
    images = torch.randn(n, 3, 8, 8)
    labels = (torch.rand(n, 7) > 0.5).float()
    loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 7))
    return model, loader


def test_pos_weight_values():
    labels = torch.zeros(10, 2)
    labels[:2, 0] = 1                       # finding 0: 2 positives, 8 negatives
    weights = compute_pos_weight(labels)
    assert weights[0].item() == 4.0
    assert weights[1].item() == 10.0        # no positives: clamped to 1, so 10 / 1


def test_epoch_changes_weights():
    model, loader = tiny_setup()
    before = model[1].weight.clone()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    train_one_epoch(model, loader, opt, nn.BCEWithLogitsLoss())
    assert not torch.equal(before, model[1].weight)


def test_loss_goes_down_when_memorizing():
    model, loader = tiny_setup()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    first = train_one_epoch(model, loader, opt, loss_fn)
    for _ in range(100):
        last = train_one_epoch(model, loader, opt, loss_fn)
    assert last < 0.5 * first


def test_predict_probs_shape_and_range():
    model, loader = tiny_setup(n=10)        # 10 images, batches of 4: last batch has 2
    probs, labels = predict_probs(model, loader)
    assert probs.shape == (10, 7) and labels.shape == (10, 7)
    assert probs.min() >= 0 and probs.max() <= 1


def test_predict_probs_uses_eval_mode():
    model, loader = tiny_setup()
    model.train()
    predict_probs(model, loader)
    assert model.training is False 