import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


class EconomyModel(nn.Module):
    def __init__(self, hidden_size: int = 8, num_layers: int = 3, dropout: float = 0.0, bidirectional: bool = True):
        super(EconomyModel, self).__init__()
        self.activation = nn.PReLU()
        self.drop = nn.Dropout(dropout)
        self.flatten = nn.Flatten()

        self.lstms = nn.ModuleList([])

        for _ in range(num_layers):
            self.lstms.append(LSTMWithAttention(hidden_size,
                                                hidden_size, 1, dropout, bidirectional))

        self.fc_1 = nn.Linear(hidden_size * 12, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, 1)

    def forward(self, X: torch.tensor) -> torch.tensor:
        for lstm in self.lstms:
            X = lstm(X)

        X = self.flatten(X)

        X = self.activation(self.drop(self.fc_1(X)))

        X = self.fc_2(X)

        X = F.sigmoid(X)

        return X


class Model(nn.Module):
    def __init__(self, hidden_size: int = 30, n_heads: int = 8, dropout: float = 0.1, n_layers: int = 5, num_stocks: int = 10):
        super(Model, self).__init__()
        self.name_embedding = nn.Embedding(num_stocks, 1)
        self.activation = nn.PReLU()
        self.drop = nn.Dropout(dropout)
        self.flatten = nn.Flatten()

        self.economy = EconomyModel(8, 3, dropout, True)

        d_ff = hidden_size * 15
        self.hidden_size = hidden_size

        encoder = nn.TransformerEncoderLayer(
            hidden_size, n_heads, d_ff, dropout, self.activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, n_layers)

        decoder = nn.TransformerDecoderLayer(
            hidden_size, n_heads, d_ff, dropout, self.activation, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder, n_layers)

        self.fc_1 = nn.Linear(hidden_size * 15, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, 15)
        self.fc_3 = nn.Linear(15, 1)

        self.fcs = nn.ModuleList([self.fc_1, self.fc_2, self.fc_3])

    def forward(self, X, economic_indicators):
        economic = self.economy(economic_indicators)

        economic = economic.unsqueeze(1).expand(-1, self.hidden_size, -1)

        X = torch.cat((X, economic), dim=2).permute(0, 2, 1)

        X[:, 0] = self.name_embedding(
            X[:, 0].long()).squeeze(2)

        encoder_out = self.encoder(X)
        X = self.decoder(X, encoder_out)

        X = self.flatten(X)

        for i, fc in enumerate(self.fcs):
            if i != 2:
                X = self.activation(self.drop(fc(X)))

            else:
                X = fc(X)

        X = F.sigmoid(X)

        return X
