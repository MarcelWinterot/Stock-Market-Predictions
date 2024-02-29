import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from calendar import monthrange
import pandas as pd

from model import Model
from utils import HistoricalDataset, CombinedDataset


dataset = torch.load('src/dataset/combined_dataset.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


HIDDEN_SIZE = 30
NUM_LAYERS = 5
DROPOUT = 0.0
BIDIRECTIONAL = True
NUM_STOCKS = 10

N_HEADS = 6

model = Model(HIDDEN_SIZE, N_HEADS, DROPOUT,
              NUM_LAYERS, NUM_STOCKS).to(device)

try:
    model.load_state_dict(torch.load('src/testing/model.pt'))
except:
    pass


class TradingStrategy:
    def __init__(self, starting_price: int, dataset: CombinedDataset, model: Model, device: torch.device) -> None:
        self.money: float = starting_price
        self.dataset: CombinedDataset = dataset
        self.model: Model = model
        self.device: torch.device = device
        self.stock = {"name": None, "number": 0}

        self.price_scaler = torch.load('src/dataset/scalers/price_scaler.pkl')
        self.year_scaler = torch.load('src/dataset/scalers/year_scaler.pkl')
        self.month_scaler = torch.load('src/dataset/scalers/month_scaler.pkl')
        self.day_scaler = torch.load('src/dataset/scalers/day_scaler.pkl')

    def get_data(self, year: int, month: int, day: int) -> torch.tensor:
        year = self.year_scaler.transform(np.array(year).reshape(-1, 1))
        month = self.month_scaler.transform(np.array(month).reshape(-1, 1))
        day = self.day_scaler.transform(np.array(day).reshape(-1, 1))

        data = self.dataset.get_date(year, month, day)

        return data

    def predict(self, data: torch.tensor) -> torch.tensor:
        X = data["X"].to(self.device)
        economic_indicators = data["economic_indicators"].to(self.device)
        predictions = []

        economic_indicators = economic_indicators.view(
            1, economic_indicators.shape[0], economic_indicators.shape[1])

        for i in range(X.shape[0]):
            x = X[i].view(1, X[i].shape[0], X[i].shape[1])
            out = self.model(x, economic_indicators)
            predictions.append(out)

        return torch.tensor(predictions)

    def calculate_profits(self, year: int, month: int, day: int) -> dict[list, torch.tensor]:
        data = self.get_data(year, month, day)
        predictions = self.predict(data)

        y = data["y"].to(self.device)

        profits = []
        for i in range(predictions.shape[0]):
            profit = predictions[i] - y[i]
            profits.append(profit)

        return {"profits": profits, "data": data["X"]}

    def strategy(self, year: int, month: int, day: int) -> None:
        raise NotImplementedError("You need to implement the strategy method")

    def loop(self, year: int) -> float:
        for month in range(1, 13):
            for day in range(5, monthrange(year, month)[1] + 1):
                try:
                    self.strategy(year, month, day)
                except Exception as e:
                    pass

            print(
                f"Money after month {month}: {self.money}")  # Will show a small number as it's the price after buying the stocks

        return self.money


class MomentumTrading(TradingStrategy):
    def __init__(self, starting_price: int, dataset: CombinedDataset, model: Model, device: torch.device) -> None:
        super().__init__(starting_price, dataset, model, device)

    def buy_stock(self, profits: list[torch.tensor], X: torch.tensor) -> None:
        best_profit = max(profits)
        best_company = profits.index(best_profit)

        name = X[best_company, -1, 0]
        price = self.price_scaler.inverse_transform([[X[best_company, -1, 1]]])
        number = self.money // price

        self.stock = {"name": name, "number": number}

        self.money -= number * price

    def sell_stock(self, X: torch.tensor) -> None:
        name = self.stock["name"].to(torch.long)

        price = self.price_scaler.inverse_transform([[X[name, -1, 1]]])

        self.money += self.stock["number"] * price

    def strategy(self, year: int, month: int, day: int) -> None:
        data = self.calculate_profits(year, month, day)

        profits: list[torch.tensor] = data['profits']
        X: torch.tensor = data['data']

        if self.stock["name"] is None:
            self.buy_stock(profits, X)

        else:
            self.sell_stock(X)

            self.buy_stock(profits, X)


strategy = MomentumTrading(100_000, dataset, model, device)

strategy.loop(2023)
