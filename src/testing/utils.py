import torch
from torch.utils.data import Dataset
import pandas as pd


class CombinedDataset(Dataset):
    def __init__(self, X, y, economic_indicators):
        super(CombinedDataset, self).__init__()
        self.X = X
        self.y = y
        self.economic_indicators = economic_indicators

        self.len = len(self.y)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]

        year = X[-1, 9]
        month = X[-1, 8]

        economic_indicators = self.economic_indicators[(
            self.economic_indicators[:, -1, -1] == year) & (self.economic_indicators[:, -1, -2] == month)]

        return {"X": X, "y": y, "economic_indicators": economic_indicators[-1, :, :-2]}

    def __len__(self):
        return self.len

    def get_date(self, year: float, month: float, day: float) -> dict[pd.DataFrame]:
        X_df = pd.DataFrame(self.X.cpu().numpy()[:, -1, :])

        year = year.repeat(X_df.shape[0])
        month = month.repeat(X_df.shape[0])
        day = day.repeat(X_df.shape[0])

        filtered_X = X_df[(X_df.iloc[:,  9] == year[0]) & (
            X_df.iloc[:,  8] == month[0]) & (X_df.iloc[:,  7] == day[0])]

        year = self.X[filtered_X.index][-1, -1, 9]
        month = self.X[filtered_X.index][-1, -1, 8]

        economic_indicators = self.economic_indicators[(
            self.economic_indicators[:, -1, -1] == year) & (self.economic_indicators[:, -1, -2] == month)]

        return {"X": self.X[filtered_X.index], "y": self.y[filtered_X.index], "economic_indicators": economic_indicators[-1, :, :-2]}
