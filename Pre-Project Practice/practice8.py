import time

import torch

from daignostic.model import build_model

model = build_model(pretrained=True)
print(model.classifier)                                     # predict
print(sum(p.numel() for p in model.parameters()))           # predict: roughly how many?
print(sum(p.numel() for p in model.classifier.parameters()))  # predict exactly

model.eval()
x = torch.randn(4, 3, 224, 224)
start = time.time()
with torch.no_grad():
    logits = model(x)
print(time.time() - start, "seconds")

print(logits.shape)                                         # predict
probs = torch.sigmoid(logits)
print(probs)                                                # predict: roughly where are the values?
print(probs.sum(dim=1))                                     # predict: do rows sum to 1?