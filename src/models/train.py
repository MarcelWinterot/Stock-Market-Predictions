import torch
from model_1 import Model_1_Small, Model_1_Large
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = 8

model = Model_1_Small(6).to(device)


class HistoricalDataset(Dataset):
    def __init__(self, X, y):
        super(HistoricalDataset, self).__init__()
        self.X = X
        self.y = y
        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = torch.load('src/dataset/dataset.pt')

print(dataset.X.shape)


criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, foreach=False)


def train(epoch, dataset, model, optimizer, criterion):
    model.train()

    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=num_workers)

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


train(10, dataset, model, optimizer, criterion)
