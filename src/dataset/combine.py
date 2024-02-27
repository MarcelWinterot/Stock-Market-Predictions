import torch
import pandas as pd
from utils import HistoricalDataset


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, economic_indicators):
        super(CombinedDataset, self).__init__()
        self.X = X
        self.y = y
        self.economic_indicators = economic_indicators

        self.len = len(self.y)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]

        year = X[-1, 8]
        month = X[-1, 7]

        economic_indicators = self.economic_indicators[(
            self.economic_indicators[:, -1, -1] == year) & (self.economic_indicators[:, -1, -2] == month)]

        return {"X": X, "y": y, "economic_indicators": economic_indicators[-1, :, :-2]}

    def __len__(self):
        return self.len


dataset = torch.load('src/dataset/dataset.pt')
economic_indicators = torch.load('src/dataset/economic_dataset.pt')

combined_dataset = CombinedDataset(dataset.X, dataset.y, economic_indicators)


# Testing if we don't get any errors when loading the data
for i in range(combined_dataset.len):
    try:
        data = combined_dataset[i]
    except Exception as e:
        print(i)
        exit()
        # print(e)

data = combined_dataset[0]

print(data['X'].shape, data['y'].shape, data['economic_indicators'].shape)

torch.save(combined_dataset, 'src/dataset/combined_dataset.pt')
