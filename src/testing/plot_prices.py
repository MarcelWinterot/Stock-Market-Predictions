import matplotlib.pyplot as plt
from model import Model
from utils import HistoricalDataset, CombinedDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch


dataset = torch.load('src/dataset/combined_dataset.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X = dataset.X
y = dataset.y

dataloader = DataLoader(dataset, 1024, False)

HIDDEN_SIZE = 30
NUM_LAYERS = 5
DROPOUT = 0.0
BIDIRECTIONAL = True
NUM_STOCKS = 10

N_HEADS = 6

model = Model(HIDDEN_SIZE, N_HEADS, DROPOUT, NUM_LAYERS, NUM_STOCKS).to(device)
model.load_state_dict(torch.load('src/testing/model.pt'))

model.eval()

predictions = []

for data in dataloader:
    X, economic_indicators = data['X'].to(
        device), data['economic_indicators'].to(device)

    out = model(X, economic_indicators)

    predictions.extend(out.cpu().detach().numpy())


scaler = torch.load('src/dataset/scalers/price_scaler.pkl')

y = y.cpu()

predictions = scaler.inverse_transform(predictions)

y = scaler.inverse_transform(y)


plt.plot(predictions, label='Predictions')

plt.plot(y, label='True')

plt.legend()

plt.show()
