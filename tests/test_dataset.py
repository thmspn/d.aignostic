import pandas as pd
import pytest
from PIL import Image

from daignostic.dataset import ChestXrayDataset, get_transform, index_images


@pytest.fixture
def tiny_data(tmp_path):
    (tmp_path / "sub").mkdir()
    Image.new("L", (64, 48), color=128).save(tmp_path / "a.png")      # grayscale, not square
    Image.new("L", (64, 48), color=128).save(tmp_path / "sub" / "b.png")
    df = pd.DataFrame({
        "Image Index": ["a.png", "b.png"],
        "Finding Labels": ["Effusion|Mass", "No Finding"],
    })
    return df, index_images(tmp_path)


def test_length(tiny_data):
    df, paths = tiny_data
    assert len(ChestXrayDataset(df, paths, get_transform())) == 2


def test_item_shapes_and_dtypes(tiny_data):
    df, paths = tiny_data
    image, label = ChestXrayDataset(df, paths, get_transform())[0]
    assert image.shape == (3, 224, 224)
    assert label.shape == (7,)
    assert label.dtype == image.dtype      # both float32


def test_labels_are_correct(tiny_data):
    df, paths = tiny_data
    ds = ChestXrayDataset(df, paths, get_transform())
    assert ds[0][1].tolist() == [0, 0, 1, 0, 0, 0, 1]   # Pleural effusion, Nodule/Mass
    assert ds[1][1].sum().item() == 0


def test_normalization_value(tiny_data):
    df, paths = tiny_data
    image, _ = ChestXrayDataset(df, paths, get_transform())[0]
    # gray 128 -> 128/255 = 0.50196; (0.50196 - 0.485) / 0.229 = about 0.0741
    assert image[0].mean().item() == pytest.approx(0.0741, abs=1e-3)


def test_missing_image_fails_early(tiny_data):
    df, paths = tiny_data
    bad = pd.DataFrame({"Image Index": ["zzz.png"], "Finding Labels": ["No Finding"]})
    with pytest.raises(KeyError):
        ChestXrayDataset(bad, paths)