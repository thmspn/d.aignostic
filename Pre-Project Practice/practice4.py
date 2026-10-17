#mini model that uses fake X-ray "image" to predict fake labels for 7 findings
import torch
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)

class FakeXrayDataset(Dataset):
    def __init__(self, n=70, size=64, n_findings=7):
        # n fake grayscale "X-rays": shape (n, 1, size, size)
        self.images = torch.rand(n, 1, size, size)
        # one 0/1 entry per finding, about 20% positive
        self.labels = (torch.rand(n, n_findings) > 0.8).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]        # (1, size, size)
        image = image.repeat(3, 1, 1)   # grayscale -> 3 channels
        label = self.labels[idx]        # (7,)
        return image, label

ds = FakeXrayDataset()
print(len(ds))                    # predict
img, lab = ds[0]
print(img.shape, lab.shape)       # predict
print(lab)                        # predict: what kind of values?

loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
print(len(loader))                # predict

for images, labels in loader:
    print(images.shape, labels.shape)   # predict
    break

sizes = [images.shape[0] for images, _ in loader]
print(sizes)                      # predict