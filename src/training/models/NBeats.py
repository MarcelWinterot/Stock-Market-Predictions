import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
