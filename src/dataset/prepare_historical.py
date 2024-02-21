from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
import torch
import pandas as pd
import os
from utils import HistoricalDataset


def load_data(file):
    new_df = pd.read_csv(file)
    new_df['name'] = os.path.basename(file).split('.')[0]
    date = pd.to_datetime(new_df['Date'])
    new_df['day'] = date.dt.day
    new_df['month'] = date.dt.month
    new_df['year'] = date.dt.year
    new_df['weekday'] = date.dt.weekday
    new_df = new_df.drop('Date', axis=1)
    return new_df


def remove_bad_stocks():
    max_price = 5000

    for i, file in enumerate(os.listdir('src/dataset/data')):
        if i % 100 == 0:
            print(f"Processed {i} files")

        df = load_data(f'src/dataset/data/{file}')
        if (df["Open"] > max_price).any() or (df["Close"] > max_price).any() or (df["High"] > max_price).any() or (df["Low"] > max_price).any():
            with open("to_remove.txt", "a") as f:
                f.write(f"{file}\n")


def prepare():
    selected_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN",
                       "FB", "TSLA", "NVDA", "INTC", "CSCO", "ADBE"]
    # files = [file.split('.')[0] for file in os.listdir(data_dir)]
    # selected_stocks = random.sample(files, 10)

    files = [f'src/dataset/data/{file}' for file in os.listdir(
        'src/dataset/data') if file.split('.')[0] in selected_stocks]

    dfs = []

    for file in files:
        dfs.append(load_data(file))

    df = pd.concat(dfs, axis=0)

    return df


df = prepare()


price_scaler = MinMaxScaler()
volume_scaler = MinMaxScaler()
day_scaler = MinMaxScaler()
month_scaler = MinMaxScaler()
year_scaler = MinMaxScaler()
weekday_scaler = MinMaxScaler()
label_encoder = LabelEncoder()

df['name'] = label_encoder.fit_transform(df['name'])
prices = pd.concat([df['Open'], df['High'], df['Low'], df['Close']], axis=0)

price_scaler.fit(prices.values.reshape(-1, 1))

df['Open'] = price_scaler.transform(df['Open'].values.reshape(-1, 1))
df['High'] = price_scaler.transform(df['High'].values.reshape(-1, 1))
df['Low'] = price_scaler.transform(df['Low'].values.reshape(-1, 1))
df['Close'] = price_scaler.transform(df['Close'].values.reshape(-1, 1))
df['Adj Close'] = price_scaler.transform(df['Adj Close'].values.reshape(-1, 1))
df['Volume'] = volume_scaler.fit_transform(df['Volume'].values.reshape(-1, 1))
df['day'] = day_scaler.fit_transform(df['day'].values.reshape(-1, 1))
df['month'] = month_scaler.fit_transform(df['month'].values.reshape(-1, 1))
df['year'] = year_scaler.fit_transform(df['year'].values.reshape(-1, 1))
df['weekday'] = weekday_scaler.fit_transform(
    df['weekday'].values.reshape(-1, 1))


print(df.head())


def remove_stocks_older_than_1986(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df['year'] > 0.16) & (df['year'] < 0.925)]  # TODO Remove this


df = remove_stocks_older_than_1986(df)


with open('src/dataset/scalers/price_scaler.pkl', 'wb') as f:
    torch.save(price_scaler, f)

with open('src/dataset/scalers/volume_scaler.pkl', 'wb') as f:
    torch.save(volume_scaler, f)

with open('src/dataset/scalers/day_scaler.pkl', 'wb') as f:
    torch.save(day_scaler, f)

with open('src/dataset/scalers/month_scaler.pkl', 'wb') as f:
    torch.save(month_scaler, f)

with open('src/dataset/scalers/year_scaler.pkl', 'wb') as f:
    torch.save(year_scaler, f)

with open('src/dataset/scalers/weekday_scaler.pkl', 'wb') as f:
    torch.save(weekday_scaler, f)

with open('src/dataset/scalers/label_encoder.pkl', 'wb') as f:
    torch.save(label_encoder, f)

print(df.head(100))

X = df[['name', 'Open', 'High', 'Low', 'Adj Close',
        'Volume', 'day', 'month', 'year', 'weekday']]
y = df['Close']

y = y.fillna(y.mean())
X = X.fillna(method='ffill')
X = X.fillna(method='bfill')

X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32)

y = y.reshape(-1, 1)


NUM_PREVIOUS_DAYS = 29

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.DataFrame(X.numpy())

unique_stocks = df[0].unique()

print(f"Num unique stocks: {len(unique_stocks)}")

stock_indexes = []

for stock in df[0].unique():
    stock_indexes.append(df[df[0] == stock].index[0])  # Starting index
    stock_indexes.append(df[df[0] == stock].index[-1])  # Ending index

X = X.reshape(-1, 1, 10)

print(X.shape)
print(y.shape)

dataset_np = X.numpy()

new_dataset = []
mask = []

for i in range(len(stock_indexes) // 2):
    starting_index = stock_indexes[i * 2]
    ending_index = stock_indexes[i * 2 + 1]

    for j in range(starting_index, ending_index):
        if j >= NUM_PREVIOUS_DAYS + starting_index and j < ending_index - 1:
            current_and_last_n_days = dataset_np[j -
                                                 NUM_PREVIOUS_DAYS:j + 1, 0, :]

            new_dataset.append(current_and_last_n_days)
            mask.append(j)

new_dataset = torch.tensor(np.array(new_dataset), dtype=torch.float32)
mask = torch.tensor(mask, dtype=torch.long)

y = y[mask]

# new_dataset = new_dataset.permute(0, 2, 1)

print(new_dataset.shape)
print(y.shape)

new_dataset = new_dataset.to(device)
y = y.to(device)


dataset = HistoricalDataset(new_dataset, y)

torch.save(dataset, 'src/dataset/dataset.pt')
