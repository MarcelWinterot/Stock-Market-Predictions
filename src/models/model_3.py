"""
Transformer, mamba tests
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model_2 import LSTMWithAttention


class EconomyModel(nn.Module):
    def __init__(self, hidden_size: int = 6, num_layers: int = 3, dropout: float = 0.0, bidirectional: bool = True):
        super(EconomyModel, self).__init__()
        self.activation = nn.PReLU()
        self.drop = nn.Dropout(dropout)
        self.flatten = nn.Flatten()

        self.lstms = nn.ModuleList([])

        for _ in range(num_layers):
            self.lstms.append(LSTMWithAttention(hidden_size,
                                                hidden_size, 1, dropout, bidirectional))

        self.fc_1 = nn.Linear(hidden_size * 2 * 6, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, 6)
        self.fc_3 = nn.Linear(6, 1)

        self.fcs = nn.ModuleList([self.fc_1, self.fc_2, self.fc_3])

    def forward(self, X: torch.tensor) -> torch.tensor:
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


class Model_3(nn.Module):
    def __init__(self, hidden_size: int = 30, n_heads: int = 8, dropout: float = 0.1, n_layers: int = 5, num_stocks: int = 10):
        super(Model_3, self).__init__()
        self.name_embedding = nn.Embedding(num_stocks, 1)
        self.activation = nn.PReLU()
        self.drop = nn.Dropout(dropout)
        self.flatten = nn.Flatten()

        self.economy = EconomyModel(6, 3, dropout, True)

        d_ff = hidden_size * 10
        self.hidden_size = hidden_size

        encoder = nn.TransformerEncoderLayer(
            hidden_size, n_heads, d_ff, dropout, self.activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, n_layers)

        self.fc_1 = nn.Linear(hidden_size * 11, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, 11)
        self.fc_3 = nn.Linear(11, 1)

        self.fcs = nn.ModuleList([self.fc_1, self.fc_2, self.fc_3])

    def forward(self, X, economic_indicators):
        economic = self.economy(economic_indicators)

        economic = economic.unsqueeze(1).expand(-1, self.hidden_size, -1)

        X = torch.cat((X, economic), dim=2).permute(0, 2, 1)

        X[:, 0] = self.name_embedding(
            X[:, 0].long()).squeeze(2)

        X = self.encoder(X)

        X = self.flatten(X)

        for i, fc in enumerate(self.fcs):
            if i != 2:
                X = self.activation(self.drop(fc(X)))

            else:
                X = fc(X)

        X = F.sigmoid(X)

        return X
