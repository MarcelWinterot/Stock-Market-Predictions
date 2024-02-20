import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class RNNBlock(nn.Module):
    def __init__(self, activation, in_channels, in_between_channels, out_channels, num_layers=1, bidirectional=False, dropout=0.0, use_norm=True, using_time2vec=False):
        super(RNNBlock, self).__init__()
        self.activation = activation
        self.use_norm = use_norm

        self.bidirectional_modification = 2 if bidirectional else 1
        self.time2vec_modification = 2 if using_time2vec else 1

        self.lstm = nn.LSTM(
            in_channels * self.time2vec_modification, in_between_channels // self.bidirectional_modification, num_layers, batch_first=True, bidirectional=bidirectional)
        self.gru = nn.GRU(
            in_between_channels, out_channels // self.bidirectional_modification, num_layers, batch_first=True, bidirectional=bidirectional)

        if use_norm:
            self.norm = nn.LayerNorm(out_channels)

        self.drop = nn.Dropout(dropout)

    def forward(self, X: torch.tensor) -> torch.tensor:
        X, _ = self.lstm(X)
        X = self.activation(X)
        X = self.drop(X)

        X, _ = self.gru(X)
        X = self.activation(X)
        X = self.drop(X)

        if self.use_norm:
            X = self.norm(X)

        return X


class Time2Vec(nn.Module):
    def __init__(self, in_features, out_features, activation=torch.sin):
        super(Time2Vec, self).__init__()

        self.linear_1 = nn.Linear(in_features, out_features)
        self.linear_2 = nn.Linear(in_features, out_features)
        self.f = activation

        v1 = self.f(self.linear_1(X))
        v2 = self.linear_2(X)

        X = torch.cat([v1, v2])
        return X


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
