# Code taken from: https://www.kaggle.com/code/jacksoncrow/download-nasdaq-historical-data/notebook
# Be sure to create the folders hist, etfs, stocks before running this script

from os.path import isfile, join
import shutil
import contextlib
import os
import yfinance as yf
import pandas as pd

offset = 0
limit = 3000
period = 'max'  # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max


data = pd.read_csv(
    "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt", sep='|')
data_clean = data[data['Test Issue'] == 'N']
symbols = data_clean['NASDAQ Symbol'].tolist()
print('total number of symbols traded = {}'.format(len(symbols)))

# I'm testing the model using only 10 stocks
# If you want to have all the stocks remove the next 2 lines
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN",
           "META", "TSLA", "NVDA", "INTC", "CSCO", "ADBE"]
data_clean = data_clean[data_clean['NASDAQ Symbol'].isin(symbols)]

limit = limit if limit else len(symbols)
end = min(offset + limit, len(symbols))
is_valid = [False] * len(symbols)
# force silencing of verbose API
with open(os.devnull, 'w') as devnull:
    with contextlib.redirect_stdout(devnull):
        for i in range(offset, end):
            s = symbols[i]
            data = yf.download(s, period=period)
            if len(data.index) == 0:
                continue

            is_valid[i] = True
            data.to_csv('hist/{}.csv'.format(s))

print('Total number of valid symbols downloaded = {}'.format(sum(is_valid)))

valid_data = data_clean[is_valid]
valid_data.to_csv('symbols_valid_meta.csv', index=False)


etfs = valid_data[valid_data['ETF'] == 'Y']['NASDAQ Symbol'].tolist()
stocks = valid_data[valid_data['ETF'] == 'N']['NASDAQ Symbol'].tolist()


def move_symbols(symbols, dest):
    for s in symbols:
        filename = '{}.csv'.format(s)
        shutil.move(join('hist', filename), join(dest, filename))


move_symbols(etfs, "etfs")
move_symbols(stocks, "stocks")
