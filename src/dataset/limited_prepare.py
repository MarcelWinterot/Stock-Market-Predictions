# This version of prepare works with only the selected stocks

import pandas as pd
import os
import pickle as pkl
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')


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

    for i, file in enumerate(os.listdir(data_dir)):
        if i % 100 == 0:
            print(f"Processed {i} files")

        df = load_data(os.path.join(data_dir, file))
        if (df["Open"] > max_price).any() or (df["Close"] > max_price).any() or (df["High"] > max_price).any() or (df["Low"] > max_price).any():
            with open("to_remove.txt", "a") as f:
                f.write(f"{file}\n")


def main():
    selected_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN",
                       "FB", "TSLA", "NVDA", "INTC", "CSCO", "ADBE"]
    # files = [file.split('.')[0] for file in os.listdir(data_dir)]
    # selected_stocks = random.sample(files, 10)

    files = [os.path.join(data_dir, file) for file in os.listdir(
        data_dir) if file.split('.')[0] in selected_stocks]

    dfs = []

    for file in files:
        dfs.append(load_data(file))

    df = pd.concat(dfs, axis=0)

    with open(os.path.join(current_dir, 'df.pkl'), 'wb') as f:
        pkl.dump(df, f)


if __name__ == '__main__':
    main()
    # remove_bad_stocks()
