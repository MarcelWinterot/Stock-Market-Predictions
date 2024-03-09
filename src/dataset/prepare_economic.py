from torch.utils.data import Dataset
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

NUM_PREVIOUS_MONTHS = 11

CPI = pd.read_csv('src/dataset/economic_data/CPI.csv')
NONFARM_PAYROLL = pd.read_csv('src/dataset/economic_data/NONFARM_PAYROLL.csv')
REAL_GDP = pd.read_csv('src/dataset/economic_data/REAL_GDP.csv')
UNEMPLOYMENT = pd.read_csv('src/dataset/economic_data/UNEMPLOYMENT.csv')
WTI = pd.read_csv('src/dataset/economic_data/WTI.csv')
GOLD = pd.read_csv('src/dataset/economic_data/GOLD.csv')
CCI = pd.read_csv('src/dataset/economic_data/CCI.csv')
PPI = pd.read_csv('src/dataset/economic_data/PPI.csv')


def process_alpha_vantage_df(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['timestamp'])
    df = df.drop(columns=['timestamp'])
    df['value'] = df['value'].astype(float)
    df = df.iloc[::-1]
    df = df.set_index('date')
    return df


# def process_gold(df: pd.DataFrame) -> pd.DataFrame:
#     df['date'] = pd.to_datetime(df['DATE'])
#     df['value'] = df['PCU2122212122210'].astype(float)
#     df = df.drop(columns=['DATE', 'PCU2122212122210'])
#     df = df.set_index('date')
#     return df

def process_gold(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['Time'])
    df['date'] = df['date'].apply(lambda x: x.replace(day=1))
    df['value'] = df['Last'].astype(float)

    df = df.drop(columns=['Exp Date', 'Symbol', 'Contract Name', '52W High',
                 '52W High Date', '52W Low', '52W Low Date', '52W %Chg', 'Time', 'Last'])

    df = df.set_index('date')

    return df


CPI = process_alpha_vantage_df(CPI)
NONFARM_PAYROLL = process_alpha_vantage_df(NONFARM_PAYROLL)
REAL_GDP = process_alpha_vantage_df(REAL_GDP)
UNEMPLOYMENT = process_alpha_vantage_df(UNEMPLOYMENT)
WTI = process_alpha_vantage_df(WTI)
GOLD = process_gold(GOLD)
CCI = process_alpha_vantage_df(CCI)
PPI = process_alpha_vantage_df(PPI)


def process_gdp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.resample('MS').ffill()
    return df


REAL_GDP = process_gdp(REAL_GDP)


def combine_datasets(CPI, NONFARM_PAYROLL, REAL_GDP, UNEMPLOYMENT, WTI, GOLD, CCI, PPI):
    df = CPI
    df['CPI'] = df['value']
    df = df.drop(columns=['value'])

    df['NONFARM_PAYROLL'] = NONFARM_PAYROLL['value']
    df['REAL_GDP'] = REAL_GDP['value']
    df['UNEMPLOYMENT'] = UNEMPLOYMENT['value']
    df['WTI'] = WTI['value']
    df['GOLD'] = GOLD['value']
    df['CCI'] = CCI['value']
    df['PPI'] = PPI['value']

    df = df.dropna(axis=0, how='any')

    return df


df = combine_datasets(CPI, NONFARM_PAYROLL, REAL_GDP,
                      UNEMPLOYMENT, WTI, GOLD, CCI, PPI)

print(df.head())


def scale_df(df: pd.DataFrame) -> pd.DataFrame:
    scalers = {}

    for column in df.columns:
        scaler = MinMaxScaler()
        df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
        scalers[column] = scaler

    return df, scalers


df, scalers = scale_df(df)

print(df.head())

with open('src/dataset/economic_data/scalers.pkl', 'wb') as f:
    pickle.dump(scalers, f)


def add_time(df: pd.DataFrame) -> pd.DataFrame:
    df['month'] = df.index.month
    df['year'] = df.index.year

    with open('src/dataset/scalers/month_scaler.pkl', 'rb') as f:
        month_scaler = torch.load(f)

    with open('src/dataset/scalers/year_scaler.pkl', 'rb') as f:
        year_scaler = torch.load(f)

    df['month'] = month_scaler.transform(df['month'].values.reshape(-1, 1))
    df['year'] = year_scaler.transform(df['year'].values.reshape(-1, 1))

    return df


df = add_time(df)

print(df.head())
print(df.tail())

X = torch.tensor(df.values).float()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.DataFrame(X.numpy())

X = X.reshape(-1, 1, 10)

print(X.shape)

dataset_np = X.numpy()

new_dataset = []

for i in range(len(dataset_np)):
    if i >= NUM_PREVIOUS_MONTHS and i < len(dataset_np) - 1:
        current_and_last_n_days = dataset_np[i -
                                             NUM_PREVIOUS_MONTHS:i + 1, 0, :]

        new_dataset.append(current_and_last_n_days)

new_dataset = torch.tensor(np.array(new_dataset), dtype=torch.float32)

print(new_dataset.shape)

new_dataset = new_dataset

torch.save(new_dataset, 'src/dataset/economic_dataset.pt')
