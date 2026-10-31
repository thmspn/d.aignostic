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

train_val_list = set(Path.home().joinpath("dev/data/nih/train_val_list.txt").read_text().split())
test_list = set(Path.home().joinpath("dev/data/nih/test_list.txt").read_text().split())
tv_patients = set(df.loc[df["Image Index"].isin(train_val_list), "Patient ID"])
test_patients = set(df.loc[df["Image Index"].isin(test_list), "Patient ID"])
print(len(tv_patients), len(test_patients), len(tv_patients & test_patients))   # predict the last number