import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNBlock(nn.Module):
    def __init__(self, activation, in_channels, in_between_channels, out_channels, num_layers=1, bidirectional=False, dropout=0.0, use_norm=True):
        super(RNNBlock, self).__init__()
        self.activation = activation
        self.use_norm = use_norm

        if bidirectional:
            self.lstm = nn.LSTM(
                in_channels, in_between_channels // 2, num_layers, batch_first=True, bidirectional=bidirectional)
            self.gru = nn.GRU(
                in_between_channels, out_channels // 2, num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(
                in_channels, in_between_channels, num_layers, batch_first=True, bidirectional=bidirectional)
            self.gru = nn.GRU(
                in_between_channels, out_channels, num_layers, batch_first=True, bidirectional=bidirectional)

        if use_norm:
            self.norm = nn.LayerNorm(out_channels)

        self.drop = nn.Dropout(dropout)

    def forward(self, X):
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

        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(
            torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(
            torch.randn(in_features, out_features-1))
        self.f = activation

    def forward(self, X):
        return X


class Model_1(nn.Module):
    def __init__(self, num_stocks: int, starting_channels: int = 365) -> None:
        super(Model_1, self).__init__()
        self.name_embedding = nn.Embedding(num_stocks, 1)

        # Assuming we are using starting_channels equal to 365
        self.rnn_1 = RNNBlock(starting_channels, 730, 1200)
        self.rnn_2 = RNNBlock(starting_channels, 1200, 365)
