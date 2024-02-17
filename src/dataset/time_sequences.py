# import torch
# import numpy as np
# from torch.utils.data import Dataset
# import pandas as pd

# NUM_PREVIOUS_DAYS = 29

# X = torch.load('src/dataset/X.pt')
# y = torch.load('src/dataset/y.pt')

# df = pd.DataFrame(X.numpy())

# unique_stocks = df[0].unique()

# print(f"Num unique stocks: {len(unique_stocks)}")

# stock_indexes = []

# for stock in df[0].unique():
#     stock_indexes.append(df[df[0] == stock].index[0])  # Starting index
#     stock_indexes.append(df[df[0] == stock].index[-1])  # Ending index

# X = X.reshape(-1, 1, 10)

# print(f"X shape: {X.shape}")
# print(f"y shape: {y.shape}")


# dataset_np = X.numpy()

# new_dataset = []


# for i in range(len(stock_indexes) // 2):
#     starting_index = stock_indexes[i*2]
#     ending_index = stock_indexes[i*2+1]

#     for j in range(starting_index, ending_index):
#         if j >= NUM_PREVIOUS_DAYS + starting_index and j < ending_index - 1:
#             current_and_last_n_days = dataset_np[j-NUM_PREVIOUS_DAYS:j+1, 0, :]

#             new_dataset.append(current_and_last_n_days)

#         else:
#             y = torch.cat((y[:j], y[j+1:]))


# new_dataset = torch.tensor(np.array(new_dataset), dtype=torch.float32)

# new_dataset = new_dataset.permute(0, 2, 1)

# print(f"Final X shape: {new_dataset.shape}")
# print(f"Final y shape: {y.shape}")


# class HistoricalDataset(Dataset):
#     def __init__(self, X, y):
#         super(HistoricalDataset, self).__init__()
#         self.X = X
#         self.y = y
#         self.len = len(self.y)

#     def __len__(self):
#         return self.len

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]


# dataset = HistoricalDataset(new_dataset, y)

# torch.save(dataset, 'src/dataset/dataset.pt')
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd

NUM_PREVIOUS_DAYS = 29

X = torch.load('src/dataset/X.pt')
y = torch.load('src/dataset/y.pt')

df = pd.DataFrame(X.numpy())

unique_stocks = df[0].unique()

print(f"Num unique stocks: {len(unique_stocks)}")

stock_indexes = []

for stock in df[0].unique():
    stock_indexes.append(df[df[0] == stock].index[0])  # Starting index
    stock_indexes.append(df[df[0] == stock].index[-1])  # Ending index

X = X.reshape(-1, 1, 10)

print(X.shape)
print(y.shape)

dataset_np = X.numpy()

new_dataset = []
mask = []

for i in range(len(stock_indexes) // 2):
    starting_index = stock_indexes[i * 2]
    ending_index = stock_indexes[i * 2 + 1]

    for j in range(starting_index, ending_index):
        if j >= NUM_PREVIOUS_DAYS + starting_index and j < ending_index - 1:
            current_and_last_n_days = dataset_np[j -
                                                 NUM_PREVIOUS_DAYS:j + 1, 0, :]

            new_dataset.append(current_and_last_n_days)
            mask.append(j)

new_dataset = torch.tensor(np.array(new_dataset), dtype=torch.float32)
mask = torch.tensor(mask, dtype=torch.long)

y = y[mask]

new_dataset = new_dataset.permute(0, 2, 1)

print(new_dataset.shape)
print(y.shape)


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
