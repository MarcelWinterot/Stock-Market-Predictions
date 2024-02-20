import pandas as pd

CPI = pd.read_csv('src/dataset/economic_data/CPI.csv')
NONFARM_PAYROLL = pd.read_csv('src/dataset/economic_data/NONFARM_PAYROLL.csv')
REAL_GDP = pd.read_csv('src/dataset/economic_data/REAL_GDP.csv')
UNEMPLOYMENT = pd.read_csv('src/dataset/economic_data/UNEMPLOYMENT.csv')
WTI = pd.read_csv('src/dataset/economic_data/WTI.csv')
GOLD = pd.read_csv('src/dataset/economic_data/GOLD.csv')


def process_alpha_vantage_df(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['timestamp'])
    df = df.drop(columns=['timestamp'])
    df['value'] = df['value'].astype(float)
    df = df.iloc[::-1]
    df = df.set_index('date')
    return df


def process_gold(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['DATE'])
    df['value'] = df['PCU2122212122210'].astype(float)
    df = df.drop(columns=['DATE', 'PCU2122212122210'])
    df = df.set_index('date')
    return df


CPI = process_alpha_vantage_df(CPI)
NONFARM_PAYROLL = process_alpha_vantage_df(NONFARM_PAYROLL)
REAL_GDP = process_alpha_vantage_df(REAL_GDP)
UNEMPLOYMENT = process_alpha_vantage_df(UNEMPLOYMENT)
WTI = process_alpha_vantage_df(WTI)
GOLD = process_gold(GOLD)


def process_gdp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.resample('MS').pad()
    return df


REAL_GDP = process_gdp(REAL_GDP)

print(CPI.head())
print(NONFARM_PAYROLL.head())
print(REAL_GDP.head())
print(UNEMPLOYMENT.head())
print(WTI.head())
print(GOLD.head())
