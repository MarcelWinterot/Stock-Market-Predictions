# This version of prepare works with only the selected stocks

import pandas as pd
import os
import pickle as pkl

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


def main():
    selected_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "FB"]
    files = [os.path.join(data_dir, file) for file in os.listdir(
        data_dir) if file.split('.')[0] in selected_stocks]

    dfs = []

    for file in files:
        dfs.append(load_data(file))

    df = pd.concat(dfs, axis=0)

    print(df['name'].unique())

    with open(os.path.join(current_dir, 'df.pkl'), 'wb') as f:
        pkl.dump(df, f)


if __name__ == '__main__':
    main()
