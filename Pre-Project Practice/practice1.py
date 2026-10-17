#Concept: What and how to use tensors
import torch

# A tiny 2x3 tensor: 2 rows, 3 columns.
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

print(x.shape)         # predict this
print(x.dtype)         # predict this
print(x.sum())         # predict this
print(x.sum(dim=0))    # predict this
print(x.sum(dim=1))    # predict this

# The shape of one batch of images going into our model later:
# (batch size, channels, height, width)
batch = torch.zeros(32, 3, 224, 224)
print(batch.shape)
print(batch.numel())   # total count of numbers; predict this too