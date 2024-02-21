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

        year = X[-1, -2]
        month = X[-1, -3]

        economic_indicators = self.economic_indicators[(
            self.economic_indicators[:, -1, -1] == year) & (self.economic_indicators[:, -1, -2] == month)]

        return {"X": X, "y": y, "economic_indicators": economic_indicators[-1, :, :-2]}

    def __len__(self):
        return self.len


dataset = torch.load('src/dataset/dataset.pt')
economic_indicators = torch.load('src/dataset/economic_dataset.pt')

combined_dataset = CombinedDataset(dataset.X, dataset.y, economic_indicators)


torch.save(combined_dataset, 'src/dataset/combined_dataset.pt')


combined_dataset[0]
