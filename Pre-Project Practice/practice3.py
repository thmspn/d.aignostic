#mini model using a fixed, random set to train a model to predict labels given numbers
import torch
from torch import nn

torch.manual_seed(0)   # fixed seed, like our hard rule

# Fake data: 200 examples, 4 numbers each, 3 yes/no labels each.
X = torch.randn(200, 4)
true_w = torch.tensor([[1.0, -1.0,  0.0],
                       [1.0,  0.0,  1.0],
                       [0.0,  1.0, -1.0],
                       [0.0,  1.0,  1.0]])   # shape (4, 3)
y = (X @ true_w > 0).float()                 # labels, shape (200, 3)

print(X.shape, y.shape)    # predict
print(y.mean(dim=0))       # predict: what does this show?

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(200):
    logits = model(X)              # raw scores, shape (200, 3)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 40 == 0:
        print(epoch, loss.item())  # predict: roughly what is the epoch 0 loss?

probs = torch.sigmoid(model(X))    # raw scores -> probabilities
pred = (probs > 0.5).float()
print((pred == y).float().mean(dim=0))   # predict: accuracy per label