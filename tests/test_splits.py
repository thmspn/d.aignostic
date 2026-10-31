import pytest

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