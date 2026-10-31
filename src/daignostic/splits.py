"""Patient-level splitting: all images from one patient go to the same split."""

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