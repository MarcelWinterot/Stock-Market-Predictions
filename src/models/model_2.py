"""
Simple LSTM with attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMWithAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0, bidirectional: bool = True) -> None:
        super(LSTMWithAttention, self).__init__()
        self.bidirectional_modification = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size, hidden_size // self.bidirectional_modification, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)

        self.attn = nn.MultiheadAttention(
            hidden_size, 1, dropout=dropout, batch_first=True)

    def forward(self, X: torch.tensor) -> torch.tensor:
        X, _ = self.lstm(X)

        X = self.attn(X, X, X)[0]

        return X


class Model_2(nn.Module):
    def __init__(self, hidden_size: int = 30, num_layers: int = 5, dropout: float = 0.0, bidirectional: bool = True, num_stocks: int = 10) -> None:
        super(Model_2, self).__init__()
        self.name_embedding = nn.Embedding(num_stocks, 1)
        self.activation = nn.PReLU()
        self.drop = nn.Dropout(dropout)
        self.flatten = nn.Flatten()

        self.lstms = nn.ModuleList([])

        for _ in range(num_layers):
            self.lstms.append(LSTMWithAttention(hidden_size,
                                                hidden_size, 1, dropout, bidirectional))

        self.fc_1 = nn.Linear(hidden_size * 10, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, 10)
        self.fc_3 = nn.Linear(10, 1)

        self.fcs = nn.ModuleList([self.fc_1, self.fc_2, self.fc_3])

    def forward(self, X: torch.tensor) -> torch.tensor:
        X[:, 0] = self.name_embedding(
            X[:, 0].long()).squeeze(2)

        for lstm in self.lstms:
            X = lstm(X)

        X = self.flatten(X)

        for i, fc in enumerate(self.fcs):
            if i != 2:
                X = self.activation(self.drop(fc(X)))

            else:
                X = fc(X)

        X = F.sigmoid(X)

        return X
