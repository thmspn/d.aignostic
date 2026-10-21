from pathlib import Path

import pandas as pd

from daignostic.splits import assert_no_overlap, split_patient_ids

df = pd.read_csv(Path.home() / "dev/data/nih/Data_Entry_2017.csv")

train_ids, val_ids = split_patient_ids(df["Patient ID"], val_fraction=0.1, seed=0)
assert_no_overlap(train_ids, val_ids)

train_df = df[df["Patient ID"].isin(train_ids)]
val_df = df[df["Patient ID"].isin(val_ids)]

print(len(train_ids), len(val_ids))   # predict
print(len(train_df), len(val_df))     # predict: is val exactly 10% of IMAGES?
print(len(val_df) / len(df))