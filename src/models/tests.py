import matplotlib.pyplot as plt
from model_2 import Model_2
from utils import HistoricalDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
dataset = torch.load('src/dataset/dataset.pt')


dataset = torch.load('src/dataset/dataset.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


X = dataset.X
y = dataset.y

dataloader = DataLoader(dataset, 1024, False)

model = Model_2(num_stocks=10).to(device)
model.load_state_dict(torch.load('src/models/model_fold_1.pt'))

predictions = []

for X, _ in dataloader:
    X = X.to(device)

    out = model(X)

    predictions.extend(out.cpu().detach().numpy())


plt.plot(predictions, label='Predictions')

plt.plot(y.cpu().detach().numpy(), label='True')

plt.legend()

plt.show()
