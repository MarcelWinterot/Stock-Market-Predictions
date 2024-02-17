import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

df = pd.read_pickle('src/dataset/df.pkl')

print(df.head(100))


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

print(X['name'].unique())

X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32)

y = y.reshape(-1, 1)

torch.save(X, 'src/dataset/X.pt')
torch.save(y, 'src/dataset/y.pt')
