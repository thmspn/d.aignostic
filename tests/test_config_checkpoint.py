import pytest
import torch
import torch.nn as nn

from daignostic.checkpoint import load_checkpoint, save_checkpoint
from daignostic.config import TrainConfig, load_config, save_config
from daignostic.reproducibility import set_seed


def test_load_config_fills_defaults(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"epochs": 3, "lr": 0.01}')
    cfg = load_config(path)
    assert cfg.epochs == 3 and cfg.lr == 0.01
    assert cfg.seed == 0                           # default


def test_unknown_config_key_raises(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"epocs": 3}')                # typo
    with pytest.raises(TypeError):
        load_config(path)


def test_config_roundtrip(tmp_path):
    cfg = TrainConfig(seed=5, max_images=100)
    save_config(cfg, tmp_path / "c.json")
    assert load_config(tmp_path / "c.json") == cfg


def test_same_seed_same_numbers():
    set_seed(1)
    a = torch.rand(3)
    set_seed(1)
    b = torch.rand(3)
    assert torch.equal(a, b)


def make_model_and_opt():
    model = nn.Linear(4, 2)
    return model, torch.optim.Adam(model.parameters(), lr=0.1)


def test_checkpoint_roundtrip(tmp_path):
    model, opt = make_model_and_opt()
    model(torch.randn(3, 4)).sum().backward()
    opt.step()                                     # gives Adam some state to save
    save_checkpoint(tmp_path / "ck" / "last.pt", model, opt, epoch=3)

    new_model, new_opt = make_model_and_opt()
    epoch = load_checkpoint(tmp_path / "ck" / "last.pt", new_model, new_opt)
    assert epoch == 3
    assert torch.equal(model.weight, new_model.weight)
    assert len(new_opt.state_dict()["state"]) > 0  # optimizer state came back


def test_no_temp_file_left_behind(tmp_path):
    model, opt = make_model_and_opt()
    save_checkpoint(tmp_path / "last.pt", model, opt, epoch=1)
    assert [p.name for p in tmp_path.iterdir()] == ["last.pt"]