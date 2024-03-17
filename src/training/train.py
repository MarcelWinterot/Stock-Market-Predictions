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
from models.PatchRNN import PatchRNNs

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
EPOCHS = 30
BATCH_SIZE = 64
TEST_BATCH_SIZE = 512
K = 10
PATIENCE = 20
LR = 2e-4
BETAS = (0.9, 0.999)

# Model variables
HIDDEN_SIZE = 300
DROPOUT = 0.0
NUM_PATCHES = 5
NUM_STACKS = 5
NUM_LAYERS = 5
NUM_LAYERS_PER_STACK = 4
NUM_STOCKS = 10

# model = StackedRNNs(NUM_STACKS, NUM_LAYERS_PER_STACK,
#                     HIDDEN_SIZE, DROPOUT, NUM_STOCKS).to(device)
model = PatchRNNs(NUM_PATCHES, NUM_STACKS, NUM_LAYERS, HIDDEN_SIZE,
                  DROPOUT, NUM_STOCKS).to(device)

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
        print(f"Epoch: {epoch}")
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

        print(
            f"Loss: {running_loss / len(dataloader)}, MAPE: {running_mape_loss / len(dataloader)}")

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


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def k_fold_cv(k: int, dataset: torch.utils.data.Dataset, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, patience: int = None, lrs: optim.lr_scheduler = None):
    folds = KFold(n_splits=k, shuffle=True)
    # folds = TimeSeriesSplit(n_splits=k)

    for fold, (train_ids, test_ids) in enumerate(folds.split(dataset), 1):
        current_patience = 0
        best_loss = float("inf")
        best_model_state = model.state_dict()
        print(f"Fold {fold}")

        train_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_ids))
        test_loader = DataLoader(
            dataset, batch_size=TEST_BATCH_SIZE, sampler=SubsetRandomSampler(test_ids))

        for epoch in range(EPOCHS):
            train(1, train_loader, model, optimizer, criterion)

            loss, mape_loss = test(test_loader, model, criterion)

            if loss < best_loss:
                best_loss = loss
                current_patience = 0
                best_model_state = model.state_dict()
            else:
                current_patience += 1

            if patience is not None and current_patience >= patience:
                print(
                    f"Stopping training due to model not improving over {patience} epochs")
                break

            print(
                F"Epoch {epoch} loss: {loss} mape loss: {mape_loss} in test set")

            if lrs is not None:
                lrs.step()

        torch.save(best_model_state, f'./model_fold_{fold}.pt')

        model.apply(reset_weights)


k_fold_cv(K, dataset, model, optimizer, criterion, PATIENCE, lr_scheduler)
