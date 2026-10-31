from pathlib import Path

import pandas as pd

from daignostic.labels import FINDINGS, nih_labels_to_vector
from daignostic.splits import make_splits, read_image_list

NIH = Path.home() / "dev/data/nih"
df = pd.read_csv(NIH / "Data_Entry_2017.csv")

train_df, val_df, test_df = make_splits(
    df,
    read_image_list(NIH / "train_val_list.txt"),
    read_image_list(NIH / "test_list.txt"),
    val_fraction=0.1,
    seed=0,
)

print(len(train_df) + len(val_df) + len(test_df), len(df))   # predict

for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print("\n==", name, "==")
    print("images:", len(part), " patients:", part["Patient ID"].nunique())
    print(part["View Position"].value_counts(normalize=True).round(3).to_dict())
    vectors = pd.DataFrame(
        part["Finding Labels"].apply(nih_labels_to_vector).tolist(),
        columns=FINDINGS,
    )
    print(vectors.mean().round(4))                      # share of images with each finding
    print("all-zero rows:", round((vectors.sum(axis=1) == 0).mean(), 3))