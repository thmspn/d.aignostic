import json
from pathlib import Path

import pandas as pd
from PIL import Image

root = Path.home() / "dev/data/fake_nih"
(root / "images").mkdir(parents=True, exist_ok=True)
labels = ["No Finding", "Effusion", "Cardiomegaly|Mass", "Atelectasis"]
rows, train_val, test = [], [], []
for i in range(80):
    name = f"{i:03d}.png"
    patient = i // 2 + 1                           # 40 patients
    Image.new("L", (128, 128), color=(3 * i) % 256).save(root / "images" / name)
    rows.append({"Image Index": name, "Finding Labels": labels[i % 4], "Patient ID": patient})
    (test if patient > 32 else train_val).append(name)
pd.DataFrame(rows).to_csv(root / "Data_Entry_2017.csv", index=False)
(root / "train_val_list.txt").write_text("\n".join(train_val))
(root / "test_list.txt").write_text("\n".join(test))

Path("runs").mkdir(exist_ok=True)
config = {"epochs": 6, "batch_size": 8, "image_size": 64, "pretrained": False,
          "data_root": str(root), "output_dir": "runs/fake"}
Path("runs/fake_config.json").write_text(json.dumps(config, indent=2))
print("made", len(rows), "images")