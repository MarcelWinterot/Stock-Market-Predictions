import torch
import numpy as np
from torch.utils.data import Dataset

NUM_PREVIOUS_DAYS = 14

X = torch.load('src/dataset/X.pt')
y = torch.load('src/dataset/y.pt')

X = X.reshape(-1, 1, 10)

print(X.shape)
print(y.shape)

dataset_np = X.numpy()

new_dataset = []

for i in range(len(dataset_np)):
    if i >= NUM_PREVIOUS_DAYS:
        current_and_last_4_days = dataset_np[i-NUM_PREVIOUS_DAYS:i+1, 0, :]

        new_dataset.append(current_and_last_4_days)

    else:
        y = torch.cat((y[:i], y[i+1:]))

new_dataset = torch.tensor(np.array(new_dataset), dtype=torch.float32)

new_dataset = new_dataset.permute(0, 2, 1)

print(new_dataset.shape)


class HistoricalDataset(Dataset):
    def __init__(self, X, y):
        super(HistoricalDataset, self).__init__()
        self.X = X
        self.y = y
        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = HistoricalDataset(new_dataset, y)

torch.save(dataset, 'src/dataset/dataset.pt')
