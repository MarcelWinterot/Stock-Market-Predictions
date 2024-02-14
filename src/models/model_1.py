"""
Model_1 will be divided into 2 different types, small and large versions
Small version is for quick tests on the architecture
Large version is to test the performance overall and will be the version used for all practical purposes

Also
Small version is set to have a starting_channels set to 15, meaning it has only the past 14 days of data
Large version will have the last year of data
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import RNNBlock, Time2Vec


class MLP_Small(nn.Module):
    def __init__(self, activation, dropout_rate) -> None:
        super(MLP_Small, self).__init__()
        self.lienar_1 = nn.Linear(250, 100)
        self.lienar_2 = nn.Linear(100, 25)
        self.lienar_3 = nn.Linear(25, 1)

        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X):
        X = self.dropout(self.activation(self.lienar_1(X)))
        X = self.dropout(self.activation(self.lienar_2(X)))
        X = self.lienar_3(X)

        return X


class Model_1_Small(nn.Module):
    def __init__(self, num_stocks: int, starting_channels: int = 15, use_time2vec: bool = True) -> None:
        super(Model_1_Small, self).__init__()
        self.activation = nn.ReLU()
        self.bidirectional = False
        self.dropout = 0.0
        self.use_norm = True

        self.num_stocks = num_stocks
        self.starting_channels = starting_channels

        self.use_time2vec = use_time2vec

        self.rnn_1 = RNNBlock(
            self.activation, starting_channels, 30, 60, 1, self.bidirectional, self.dropout, self.use_norm, self.use_time2vec)
        self.rnn_2 = RNNBlock(
            self.activation, 60, 120, 240, 1, self.bidirectional, self.dropout, self.use_norm, self.use_time2vec)
        self.rnn_3 = RNNBlock(
            self.activation, 240, 300, 300, 1, self.bidirectional, self.dropout, self.use_norm, self.use_time2vec)
        self.rnn_4 = RNNBlock(
            self.activation, 300, 300, 300, 2, self.bidirectional, self.dropout, self.use_norm, self.use_time2vec)
        self.rnn_5 = RNNBlock(
            self.activation, 300, 100, 25, 1, self.bidirectional, self.dropout, self.use_norm, self.use_time2vec)

        self.RNNS = nn.ModuleList(
            [self.rnn_1, self.rnn_2, self.rnn_3, self.rnn_4, self.rnn_5])

        self.time2vec_1 = Time2Vec(15, 15)
        self.time2vec_2 = Time2Vec(60, 60)
        self.time2vec_3 = Time2Vec(240, 240)
        self.time2vec_4 = Time2Vec(300, 300)
        self.time2vec_5 = Time2Vec(300, 300)

        self.time2vecs = nn.ModuleList(
            [self.time2vec_1, self.time2vec_2, self.time2vec_3, self.time2vec_4, self.time2vec_5])

        self.flatten = nn.Flatten()

        self.mlp = MLP_Small(self.activation, self.dropout)

        self.name_embedding = nn.Embedding(num_stocks, 1)

    def forward(self, X):
        X[:, 0] = self.name_embedding(
            X[:, 0].long()).squeeze(2)

        for rnn, time2vec in zip(self.RNNS, self.time2vecs):
            if self.use_time2vec:
                X = time2vec(X)
            X = rnn(X)

        X = self.flatten(X)

        X = self.mlp(X)

        return X
