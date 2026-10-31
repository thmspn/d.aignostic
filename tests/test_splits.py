import pytest
import pandas as pd
from daignostic.splits import make_splits
from daignostic.splits import assert_no_overlap, split_patient_ids

IDS = list(range(1, 101)) * 3   # 100 patients, 3 images each (repeated IDs)


def test_sizes():
    train, val = split_patient_ids(IDS, val_fraction=0.1, seed=0)
    assert len(val) == 10
    assert len(train) == 90


def test_no_overlap_and_nothing_lost():
    train, val = split_patient_ids(IDS, 0.1, seed=0)
    assert train.isdisjoint(val)
    assert train | val == set(IDS)


def test_same_seed_same_split():
    assert split_patient_ids(IDS, 0.1, seed=0) == split_patient_ids(IDS, 0.1, seed=0)


def test_different_seed_different_split():
    assert split_patient_ids(IDS, 0.1, seed=0) != split_patient_ids(IDS, 0.1, seed=1)


def test_assert_no_overlap_catches_leak():
    with pytest.raises(ValueError):
        assert_no_overlap({1, 2, 3}, {3, 4})


def test_bad_fraction_raises():
    with pytest.raises(ValueError):
        split_patient_ids(IDS, 1.5, seed=0)

def make_df():
    """12 images, 6 patients, 2 images each."""
    return pd.DataFrame({
        "Image Index": [f"img{i}.png" for i in range(12)],
        "Patient ID": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
    })


def images(first, last):
    return {f"img{i}.png" for i in range(first, last + 1)}


def test_make_splits_sizes_and_no_loss():
    df = make_df()
    train, val, test = make_splits(df, images(0, 7), images(8, 11), 0.25, seed=0) 
    assert (len(train), len(val), len(test)) == (6, 2, 4)
    assert len(train) + len(val) + len(test) == len(df)


def test_make_splits_no_patient_in_two_splits():
    train, val, test = make_splits(make_df(), images(0, 7), images(8, 11), 0.25, seed=0)
    patients = [set(t["Patient ID"]) for t in (train, val, test)]
    assert patients[0].isdisjoint(patients[1])
    assert patients[0].isdisjoint(patients[2])
    assert patients[1].isdisjoint(patients[2])


def test_make_splits_catches_leaky_official_lists():
    # img8 (patient 5) is in train+val, img9 (also patient 5) is in test
    with pytest.raises(ValueError):
        make_splits(make_df(), images(0, 8), images(9, 11), 0.25, seed=0)