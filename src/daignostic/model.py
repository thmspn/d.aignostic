"""DenseNet-121 with a 7-output multi-label head."""

import torch.nn as nn
from torchvision import models

from daignostic.labels import FINDINGS


def build_model(pretrained=True):
    """Return a DenseNet-121 whose final layer has one logit per finding."""
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, len(FINDINGS))
    return model