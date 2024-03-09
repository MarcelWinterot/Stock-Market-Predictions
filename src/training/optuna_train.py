import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import optuna
from pytorch_forecasting import MAPE

from tqdm import tqdm
from sklearn.model_selection import KFold, TimeSeriesSplit

from models.utils import HistoricalDataset, CombinedDataset
from models.StackedRNNs import Stack, StackedRNNs

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
BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
K = 10
PATIENCE = 20
LR = 2e-4
BETAS = (0.9, 0.999)
NUM_TRIALS = 100

# Model variables
HIDDEN_SIZE = 300
DROPOUT = 0.0
NUM_STACKS = 25
NUM_LAYERS_PER_STACK = 4
NUM_STOCKS = 10

model = StackedRNNs(NUM_STACKS, NUM_LAYERS_PER_STACK,
                    HIDDEN_SIZE, DROPOUT, NUM_STOCKS).to(device)

try:
    model.load_state_dict(torch.load('src/models/model.pt'))
except:
    pass


dataset = torch.load('src/dataset/combined_dataset.pt')

criterion = torch.nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LR, betas=BETAS)
mape = MAPE()

# lr_scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer, step_size=5, gamma=0.5)
lr_scheduler = None


def train(epoch, dataloader, model, optimizer, criterion, use_tdqm=True):
    model.train()

    for epoch in range(epoch):
        running_loss = 0.0
        running_mape_loss = 0.0
        if use_tdqm:
            range_ = tqdm(dataloader)
            print(f"Epoch: {epoch}")
        else:
            range_ = dataloader

        for data in range_:
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

        if use_tdqm:
            print(
                f"Loss: {running_loss / len(dataloader)}, MAPE: {running_mape_loss / len(dataloader)}")

        optimizer.zero_grad()

    return running_loss / len(dataloader), running_mape_loss / len(dataloader)


def test(dataloader, model, criterion, use_tdqm=True):
    model.eval()
    running_loss = 0.0
    running_mape_loss = 0.0

    range_ = tqdm(dataloader) if use_tdqm else dataloader

    for data in range_:
        X, y, economic_indicators = data['X'].to(device), data['y'].to(
            device), data['economic_indicators'].to(device)

        # out = model(X)
        out = model(X, economic_indicators)

        loss = criterion(out, y)

        running_loss += loss.item()

        with torch.no_grad():
            running_mape_loss += mape(out, y).item()

    return running_loss / len(dataloader), running_mape_loss / len(dataloader)


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def objective(trial):
    # Parameters to optimize: lr, NUM_STACKS, NUM_LAYERS_PER_STACK
    # LR = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    NUM_STACKS = trial.suggest_int("NUM_STACKS", 1, 25)

    model = StackedRNNs(NUM_STACKS, NUM_LAYERS_PER_STACK,
                        HIDDEN_SIZE, DROPOUT, NUM_STOCKS).to(device)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=BETAS)

    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(
        dataset, batch_size=TEST_BATCH_SIZE)

    for epoch in tqdm(range(EPOCHS)):
        train(1, train_loader, model, optimizer, criterion, False)

        loss, mape = test(test_loader, model, criterion, False)

        trial.report(mape, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return mape


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name="StackedRNNs")
    study.optimize(objective, NUM_TRIALS)

    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
