import json

import pandas as pd
import pytest
from PIL import Image

from daignostic.config import TrainConfig
from daignostic.dataset import get_transform
from daignostic.run import run_training, trim_log

LABELS = ["No Finding", "Effusion", "Cardiomegaly|Mass", "Atelectasis"]


@pytest.fixture
def fake_nih(tmp_path):
    root = tmp_path / "nih"
    (root / "images").mkdir(parents=True)
    rows, train_val, test = [], [], []
    for i in range(24):
        name = f"{i:03d}.png"
        patient = i // 2 + 1                       # 12 patients, 2 images each
        Image.new("L", (40, 40), color=10 * i).save(root / "images" / name)
        rows.append({"Image Index": name, "Finding Labels": LABELS[i % 4], "Patient ID": patient})
        (test if patient > 10 else train_val).append(name)
    pd.DataFrame(rows).to_csv(root / "Data_Entry_2017.csv", index=False)
    (root / "train_val_list.txt").write_text("\n".join(train_val))
    (root / "test_list.txt").write_text("\n".join(test))
    return root


def config_for(root, tmp_path, **changes):
    settings = dict(epochs=2, batch_size=4, image_size=32, pretrained=False,
                    val_fraction=0.2, data_root=str(root), output_dir=str(tmp_path / "run"))
    settings.update(changes)
    return TrainConfig(**settings)


def logged_epochs(out):
    lines = (out / "log.jsonl").read_text().splitlines()
    return [json.loads(line)["epoch"] for line in lines]


def test_transform_size():
    assert get_transform(32)(Image.new("RGB", (40, 40))).shape == (3, 32, 32)


def test_run_writes_log_and_checkpoint(fake_nih, tmp_path):
    out = run_training(config_for(fake_nih, tmp_path))
    assert logged_epochs(out) == [1, 2]
    assert (out / "last.pt").exists() and (out / "config.json").exists()


def test_resume_adds_only_missing_epochs(fake_nih, tmp_path):
    run_training(config_for(fake_nih, tmp_path, epochs=1))
    out = run_training(config_for(fake_nih, tmp_path, epochs=2))
    assert logged_epochs(out) == [1, 2]


def test_different_config_same_folder_raises(fake_nih, tmp_path):
    run_training(config_for(fake_nih, tmp_path, epochs=1))
    with pytest.raises(ValueError):
        run_training(config_for(fake_nih, tmp_path, lr=0.5))

def test_resume_drops_log_lines_the_checkpoint_never_saved(fake_nih, tmp_path):
    out = run_training(config_for(fake_nih, tmp_path, epochs=2))
    with open(out / "log.jsonl", "a") as f:       # epoch 3 logged, then cut off
        f.write('{"epoch": 3, "train_loss": 9, "val_loss": 9}\n')
    run_training(config_for(fake_nih, tmp_path, epochs=3))
    assert logged_epochs(out) == [1, 2, 3]


def test_trim_log_skips_half_written_line(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text('{"epoch": 1}\n{"epoch": 2}\n{"epo')
    trim_log(log, 1)
    assert log.read_text() == '{"epoch": 1}\n'