#Use of Autograd
import torch

# One weight that PyTorch should track.
w = torch.tensor(3.0, requires_grad=True)

# A tiny "model": prediction = w * x, and we want it to equal target.
x = 2.0
target = 10.0

prediction = w * x
loss = (prediction - target) ** 2

print(prediction)   # predict
print(loss)         # predict

loss.backward()     # compute gradients
print(w.grad)       # predict (hint: the gradient is 2 * (prediction - target) * x)

# Gradient descent: step the weight against its gradient.
lr = 0.1            # learning rate: how big each step is
for step in range(5):
    prediction = w * x
    loss = (prediction - target) ** 2
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
    w.grad.zero_()
    print(step, w.item(), loss.item())   # predict: does loss go down each step?

# Quick dim check from last lesson:
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(t.sum(dim=0))   # predict
print(t.sum(dim=1))   # predict