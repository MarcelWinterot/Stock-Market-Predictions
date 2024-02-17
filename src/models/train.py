import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from tqdm import tqdm
from sklearn.model_selection import KFold

from model_2 import Model_2
from utils import HistoricalDataset

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 8
EPOCHS = 10
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1000
K = 10

torch.backends.cudnn.enabled = True

model = Model_2(num_stocks=5).to(device)


dataset = torch.load('src/dataset/dataset.pt')


criterion = torch.nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(epoch, dataloader, model, optimizer, criterion):
    model.train()

    for epoch in range(epoch):
        running_loss = 0.0
        print(f"Epoch: {epoch}")
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(X)

            loss = criterion(out, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(f"Loss: {running_loss / len(dataloader)}")


def test(dataloader, model, criterion):
    model.eval()

    running_loss = 0.0
    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)

        out = model(X)

        loss = criterion(out, y)

        running_loss += loss.item()

    return running_loss


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def k_fold_cv(k: int, dataset: torch.utils.data.Dataset, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module):
    folds = KFold(n_splits=k, shuffle=True)

    for fold, (train_ids, test_ids) in enumerate(folds.split(dataset), 1):
        print(f"Fold {fold}")

        model.apply(reset_weights)

        train_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_ids))
        test_loader = DataLoader(
            dataset, batch_size=TEST_BATCH_SIZE, sampler=SubsetRandomSampler(test_ids))

        for epoch in range(EPOCHS):
            train(1, train_loader, model, optimizer, criterion)

            loss = test(test_loader, model, criterion)

            print(
                F"Epoch {epoch} loss: {loss:.8f} in test set")

        torch.save(model.state_dict(), f'./model_fold_{fold}.pt')
        exit()


k_fold_cv(K, dataset, model, optimizer, criterion)
