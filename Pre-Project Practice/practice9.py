import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from daignostic.model import build_model
from daignostic.train import train_one_epoch

torch.manual_seed(0)
images = torch.randn(32, 3, 64, 64)               # small images so CPU is fast
labels = (torch.rand(32, 7) > 0.5).float()        # random labels: can only be memorized
loader = DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=True)

model = build_model(pretrained=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(30):
    loss = train_one_epoch(model, loader, optimizer, loss_fn)
    print(epoch, round(loss, 4))