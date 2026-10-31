"""Patient-level splitting: all images from one patient go to the same split."""
from pathlib import Path
import random


def split_patient_ids(patient_ids, val_fraction, seed):
    """Split unique patient IDs into (train_ids, val_ids), reproducibly."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    ids = sorted(set(patient_ids))   # unique, in a fixed order before shuffling
    rng = random.Random(seed)        # own generator: same seed, same shuffle
    rng.shuffle(ids)
    n_val = round(len(ids) * val_fraction)
    return set(ids[n_val:]), set(ids[:n_val])


def assert_no_overlap(first, second):
    """Raise if any patient appears in both groups."""
    shared = set(first) & set(second)
    if shared:
        raise ValueError(
            f"{len(shared)} patients in both splits, e.g. {sorted(shared)[:5]}"
        )

def read_image_list(path):
    """Read an NIH list file (one image filename per line) into a set."""
    return set(Path(path).read_text().split())


def make_splits(df, train_val_images, test_images, val_fraction, seed):
    """Build (train_df, val_df, test_df) from the metadata table.

    The official lists decide train+val vs test. Train+val is then split
    again by patient. Raises if any patient appears in two splits.
    """
    train_val_df = df[df["Image Index"].isin(train_val_images)]
    test_df = df[df["Image Index"].isin(test_images)]

    train_ids, val_ids = split_patient_ids(
        train_val_df["Patient ID"], val_fraction, seed
    )
    test_ids = set(test_df["Patient ID"])

    assert_no_overlap(train_ids, val_ids)
    assert_no_overlap(train_ids, test_ids)
    assert_no_overlap(val_ids, test_ids)

    train_df = train_val_df[train_val_df["Patient ID"].isin(train_ids)]
    val_df = train_val_df[train_val_df["Patient ID"].isin(val_ids)]
    return train_df, val_df, test_df