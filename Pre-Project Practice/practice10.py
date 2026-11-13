import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from daignostic.checkpoint import load_checkpoint, save_checkpoint
from daignostic.reproducibility import set_seed
from daignostic.train import train_one_epoch

set_seed(0)
images = torch.randn(32, 3, 8, 8)
labels = (torch.rand(32, 7) > 0.5).float()
loader = DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=True)
loss_fn = nn.BCEWithLogitsLoss()


def make():
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 7))
    return model, torch.optim.Adam(model.parameters(), lr=0.02)


model, opt = make()
for epoch in range(5):
    print("run 1, epoch", epoch, round(train_one_epoch(model, loader, opt, loss_fn), 4))
save_checkpoint("runs/demo/last.pt", model, opt, epoch=5)

# "the session was cut off": start over with a brand-new model
model, opt = make()
print("fresh model:", round(train_one_epoch(model, loader, opt, loss_fn), 4))   # predict

model, opt = make()
done = load_checkpoint("runs/demo/last.pt", model, opt)
print("resumed at epoch", done)
for epoch in range(done, done + 3):
    print("run 2, epoch", epoch, round(train_one_epoch(model, loader, opt, loss_fn), 4))   # predict