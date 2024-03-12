import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
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


class DAIN_Layer(nn.Module):
    def __init__(self, mode='adaptive_avg', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, input_dim=144):
        super(DAIN_Layer, self).__init__()
        print("Mode = ", mode)

        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(
            data=np.eye(input_dim, input_dim))

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(
            data=np.eye(input_dim, input_dim))

        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def forward(self, x):
        # Expecting  (n_samples, dim,  n_feature_vectors)

        # Nothing to normalize
        if self.mode == None:
            pass

        # Do simple average normalization
        elif self.mode == 'avg':
            avg = torch.mean(x, 2)
            avg = avg.resize(avg.size(0), avg.size(1), 1)
            x = x - avg

        # Perform only the first step (adaptive averaging)
        elif self.mode == 'adaptive_avg':
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.resize(
                adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

        # Perform the first + second step (adaptive averaging + adaptive scaling )
        elif self.mode == 'adaptive_scale':

            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.resize(
                adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

            # Step 2:
            std = torch.mean(x ** 2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.resize(
                adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / (adaptive_std)

        elif self.mode == 'full':

            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.resize(
                adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

            # # Step 2:
            std = torch.mean(x ** 2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.resize(
                adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / adaptive_std

            # Step 3:
            avg = torch.mean(x, 2)
            gate = F.sigmoid(self.gating_layer(avg))
            gate = gate.resize(gate.size(0), gate.size(1), 1)
            x = x * gate

        else:
            assert False

        return x


class Block(nn.Module):
    def __init__(self, hidden_size: int, activation: callable, num_layers: int = 4) -> None:
        super().__init__()
        self.activation = activation
        self.fcs = nn.ModuleList([])

        for _ in range(num_layers):
            self.fcs.append(nn.Linear(hidden_size, hidden_size))

    def forward(self, X: torch.tensor) -> torch.tensor:
        for fc in self.fcs:
            X = self.activation(fc(X))

        return X


class Stack(nn.Module):
    def __init__(self, num_blocks: int, hidden_size: int, activation: callable, num_layers: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([])

        for _ in range(num_blocks):
            self.blocks.append(Block(hidden_size, activation, num_layers))

    def forward(self, X: torch.tensor) -> torch.tensor:
        for block in self.blocks:
            X = X - block(X)

        return X


class NBeats(nn.Module):
    def __init__(self, num_stacks: int, num_blocks: int, hidden_size: int, activation: callable, num_layers: int = 4) -> None:
        super().__init__()
        self.stacks = nn.ModuleList([])
        for _ in range(num_stacks):
            self.stacks.append(
                Stack(num_blocks, hidden_size, activation, num_layers))

    def forward(self, X: torch.tensor) -> torch.tensor:
        results = []
        for stack in self.stacks:
            X = stack(X)
            results.append(X)

        return torch.stack(results).sum(dim=0)
