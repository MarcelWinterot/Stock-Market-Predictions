import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from pytorch_forecasting import MAPE

from tqdm import tqdm
from sklearn.model_selection import KFold, TimeSeriesSplit

from models.utils import HistoricalDataset, CombinedDataset
from models.StackedRNNs import StackedRNNs

torch.autograd.set_detect_anomaly(True)


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Training variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 8
EPOCHS = 10
BATCH_SIZE = 64
TEST_BATCH_SIZE = 512
K = 10
PATIENCE = 20
LR = 1e-4
BETAS = (0.9, 0.999)

# Model variables
HIDDEN_SIZE = 300
NUM_LAYERS = 5
DROPOUT = 0.0
NUM_STACKS = 5
NUM_LAYERS_PER_STACK = 4
NUM_STOCKS = 10

model = StackedRNNs(NUM_STACKS, NUM_LAYERS_PER_STACK,
                    HIDDEN_SIZE, DROPOUT, NUM_STOCKS).to(device)

try:
    model.load_state_dict(torch.load('src/training/model.pt'))
    print("Model loaded")
except:
    pass


dataset = torch.load('src/dataset/combined_dataset.pt')

criterion = torch.nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LR, betas=BETAS)
mape = MAPE()

# lr_scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer, step_size=5, gamma=0.5)
lr_scheduler = None


def train(epoch, dataloader, model, optimizer, criterion):
    model.train()

    for epoch in range(epoch):
        running_loss = 0.0
        running_mape_loss = 0.0
        for data in tqdm(dataloader):
            X, y, economic_indicators = data['X'].to(device), data['y'].to(
                device), data['economic_indicators'].to(device)
            optimizer.zero_grad()

            # out = model(X)
            out = model(X, economic_indicators)

            loss = criterion(out, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            with torch.no_grad():
                running_mape_loss += mape(out, y).item()

        optimizer.zero_grad()

    return (running_loss / len(dataloader), running_mape_loss / len(dataloader))


def test(dataloader, model, criterion):
    model.eval()
    running_loss = 0.0
    running_mape_loss = 0.0
    for data in tqdm(dataloader):
        X, y, economic_indicators = data['X'].to(device), data['y'].to(
            device), data['economic_indicators'].to(device)

        # out = model(X)
        out = model(X, economic_indicators)

        loss = criterion(out, y)

        running_loss += loss.item()

        with torch.no_grad():
            running_mape_loss += mape(out, y).item()

    return (running_loss / len(dataloader), running_mape_loss / len(dataloader))


def lr_fidner(start_lr: float, end_lr: float, num_iterations: int) -> list[float]:
    return np.logspace(np.log10(start_lr), np.log10(end_lr), num_iterations)


lrs = lr_fidner(1e-6, 1e-2, 25)
losses = {}


folds = KFold(n_splits=K, shuffle=True)
train_id, test_id = folds.split(dataset).__next__()

train_loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_id))

test_loader = DataLoader(
    dataset, batch_size=TEST_BATCH_SIZE, sampler=SubsetRandomSampler(test_id))

for lr in lrs:
    print(f"Learning rate: {lr}")
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=BETAS)
    train(1, train_loader, model, optimizer, criterion)
    loss, mape_loss = test(test_loader, model, criterion)

    losses[lr] = mape_loss

    print(f"Loss: {loss}, MAPE: {mape_loss}")

    optimizer.zero_grad()

plt.plot(list(losses.keys()), list(losses.values()))

plt.xscale('log')
plt.xlabel('Learning rate')
plt.ylabel('Loss')

plt.show()
