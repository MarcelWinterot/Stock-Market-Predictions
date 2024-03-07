"""
Architecture inspired by the paper N-Beats 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.HMM import HMM


class MLP(nn.Module):
    def __init__(self, d_ff: int, hidden_size: int, num_variables: int, activation: callable, use_norm: bool = False) -> None:
        super(MLP, self).__init__()
        self.use_norm = use_norm
        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.fc_1 = nn.Linear(d_ff, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, num_variables)
        self.fc_3 = nn.Linear(num_variables, 1)

        if self.use_norm:
            self.norm_1 = nn.LayerNorm(hidden_size)
            self.norm_2 = nn.LayerNorm(num_variables)

    def forward(self, X: torch.tensor) -> torch.tensor:
        if self.use_norm:
            X = self.activation(self.norm_1(self.fc_1(X)))
            X = self.activation(self.norm_2(self.fc_2(X)))
        else:
            X = self.activation(self.fc_1(X))
            X = self.activation(self.fc_2(X))

        X = self.fc_3(X)
        X = self.hmm(X, torch.tensor([X.shape[1]]))

        print(X.shape)

        X = self.sigmoid(X)

        return X


class Stack(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 4, dropout: float = 0.0, activation: callable = None, norm: callable = None) -> None:
        super(Stack, self).__init__()
        self.rnns = nn.ModuleList([])
        self.activation = activation
        self.norm = norm
        self.num_layers = num_layers

        for _ in range(num_layers):
            self.rnns.append(nn.LSTM(hidden_size, hidden_size, 1,
                                     batch_first=True, dropout=dropout))

    def forward(self, X: torch.tensor) -> tuple[torch.tensor]:
        Xs = []

        for layer in self.rnns:
            layer_out = layer(X)[0]
            Xs.append(layer_out)

            X = X - layer_out

            if self.norm is not None:
                X = self.norm(X)

            if self.activation is not None:
                X = self.activation(X)

        Xs = sum(Xs)

        return (X, Xs)


class EconomyModel(nn.Module):
    def __init__(self, hidden_size: int = 8, num_stacks: int = 2, dropout: float = 0.0) -> None:
        super(EconomyModel, self).__init__()
        self.activation = nn.PReLU()
        self.flatten = nn.Flatten()

        self.stacks = nn.ModuleList([])
        for _ in range(num_stacks):
            self.stacks.append(
                Stack(hidden_size, 4, dropout, self.activation, nn.LayerNorm(hidden_size)))

        self.fc_1 = nn.Linear(hidden_size * 12, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, 1)

    def forward(self, X: torch.tensor) -> torch.tensor:
        Xs = []
        for stack in self.stacks:
            X, X2 = stack(X)

            Xs.append(X2)

        Xs = sum(Xs)

        Xs = self.flatten(Xs)

        Xs = self.activation(self.fc_1(Xs))

        Xs = self.fc_2(Xs)

        Xs = F.sigmoid(Xs)

        return Xs


class StackedRNNs(nn.Module):
    def __init__(self, num_stacks: int = 5, num_layers_per_stack: int = 4, hidden_size: int = 30, dropout: float = 0.0, num_stocks: int = 10) -> None:
        super(StackedRNNs, self).__init__()
        self.name_embedding = nn.Embedding(num_stocks, 1)
        self.activation = nn.PReLU()
        self.flatten = nn.Flatten()

        num_variables = 16
        d_ff = hidden_size * num_variables
        self.hidden_size = hidden_size

        self.stacks = nn.ModuleList([])
        for _ in range(num_stacks):
            self.stacks.append(
                Stack(hidden_size, num_layers_per_stack, dropout, self.activation, nn.LayerNorm(hidden_size)))

        self.num_stacks = num_stacks

        self.economy = EconomyModel(8, 2, dropout)

        self.mlp = MLP(d_ff, hidden_size, num_variables, self.activation)

    def forward(self, X, economic_indicators):
        economic = self.economy(economic_indicators)

        economic = economic.unsqueeze(1).expand(-1, self.hidden_size, -1)

        X = torch.cat((X, economic), dim=2).permute(0, 2, 1)

        X[:, 6] = self.name_embedding(
            X[:, 6].long()).squeeze(2)

        Xs = []
        for stack in self.stacks:
            X, X2 = stack(X)

            Xs.append(X2)

        Xs = sum(Xs)

        Xs = self.flatten(Xs)

        Xs = self.mlp(Xs)

        return Xs
