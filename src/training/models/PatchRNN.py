"""
Architecture inspired by N-Beats and PatchTST models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.NBeats import NBeats


class MLP(nn.Module):
    def __init__(self, d_ff: int, hidden_size: int, num_variables: int, activation: callable) -> None:
        super(MLP, self).__init__()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.fc_1 = nn.Linear(d_ff, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, num_variables)
        self.fc_3 = nn.Linear(num_variables, 1)

    def forward(self, X: torch.tensor) -> torch.tensor:
        X = self.activation(self.fc_1(X))
        X = self.activation(self.fc_2(X))

        X = self.fc_3(X)

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
        result = []

        for i, layer in enumerate(self.rnns):
            layer_out = layer(X)[0]
            result.append(layer_out)

            X = X - layer_out

            if self.norm is not None:
                X = self.norm(X)

            if self.activation is not None:
                X = self.activation(X)

        result = sum(result)

        return (X, result)


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
        result = []
        for stack in self.stacks:
            X, X2 = stack(X)

            result.append(X2)

        X = sum(result)

        X = self.flatten(X)

        X = self.activation(self.fc_1(X))

        X = self.fc_2(X)

        X = F.tanh(X)

        return X


class PatchRNNs(nn.Module):
    def __init__(self, num_patches: int, num_stacks_per_patch: int, num_layers_per_stack: int, hidden_size: int, dropout: float, num_stocks: int = 10) -> None:
        super(PatchRNNs, self).__init__()
        assert hidden_size % num_patches == 0, "Hidden size must be divisible by the number of patches"

        self.name_embedding = nn.Embedding(num_stocks, 1)
        self.activation = nn.PReLU()
        self.flatten = nn.Flatten()

        num_variables = 16
        d_ff = hidden_size * num_variables
        self.hidden_size = hidden_size
        self.num_patches = num_patches

        self.patches = nn.ModuleList([nn.ModuleList([Stack(hidden_size // self.num_patches, num_layers_per_stack, dropout, self.activation, nn.LayerNorm(hidden_size // self.num_patches))
                                      for _ in range(num_stacks_per_patch)]) for _ in range(num_patches)])

        self.num_stacks = num_stacks_per_patch

        self.economy = EconomyModel(8, 2, dropout)

        # self.mlp = MLP(d_ff, hidden_size, num_variables, self.activation)
        self.fc_1 = nn.Linear(d_ff, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, num_variables)
        self.fc_3 = nn.Linear(num_variables, 1)
        self.mlp_1 = NBeats(2, 4, hidden_size, self.activation)

    def forward(self, X, economic_indicators):
        economic = self.economy(economic_indicators)

        economic = economic.unsqueeze(1).expand(-1, self.hidden_size, -1)

        X = torch.cat((X, economic), dim=2).permute(0, 2, 1)

        X[:, 6] = self.name_embedding(
            X[:, 6].long()).squeeze(2)

        X = torch.chunk(X, self.num_patches, dim=2)

        new_X = [torch.zeros_like(patch) for patch in X]

        for i, patch in enumerate(X):
            result = [torch.zeros_like(patch) for _ in range(self.num_patches)]
            for j, stack in enumerate(self.patches[i]):
                patch, patch2 = stack(patch)

                result[j] = patch2

            new_X[i] = sum(result)

        X = torch.cat(new_X, dim=2)

        X = self.flatten(X)

        X = self.activation(self.fc_1(X))

        X = self.mlp_1(X)

        X = self.activation(self.fc_2(X))

        X = F.sigmoid(self.fc_3(X))

        return X
