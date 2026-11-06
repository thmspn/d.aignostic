import torch
import torch.nn as nn

from daignostic.labels import FINDINGS
from daignostic.model import build_model


def make_model():
    torch.manual_seed(0)
    return build_model(pretrained=False)


def test_head_has_one_output_per_finding():
    assert make_model().classifier.out_features == len(FINDINGS)


def test_output_shape():
    model = make_model().eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 7)


def test_eval_mode_is_deterministic():
    model = make_model().eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_loss_backward_reaches_the_head():
    model = make_model()                       # train mode; batch of 2 so BatchNorm works
    x = torch.randn(2, 3, 224, 224)
    loss = nn.BCEWithLogitsLoss()(model(x), torch.zeros(2, 7))
    loss.backward()
    assert model.classifier.weight.grad is not None