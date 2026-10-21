from pathlib import Path

import pandas as pd

from daignostic.labels import NIH_IGNORED, NIH_TO_FINDING

path = Path.home() / "dev/data/nih/Data_Entry_2017.csv"
df = pd.read_csv(path)

print(df.shape)                  # predict
print(df.columns.tolist())       # the REAL column names
print(df.head(3))

# These two column names are from memory. If you get a KeyError,
# use the real names printed above.
print(df["Patient ID"].nunique())                         # predict
print(df.groupby("Patient ID").size().describe())         # images per patient

multi = df["Finding Labels"].str.contains("|", regex=False)
print(df.loc[multi, "Finding Labels"].head())             # separator check

labels = set()
for s in df["Finding Labels"]:
    labels.update(s.split("|"))
print(sorted(labels))

known = set(NIH_TO_FINDING) | NIH_IGNORED
print("In file, not in our code:", labels - known)
print("In our code, not in file:", known - labels)