import pandas as pd
import os
import pickle as pkl
import multiprocessing as mp

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


def process_files(files):
    dfs = []

    for i, file in enumerate(files):
        if file.endswith(".csv"):
            dfs.append(load_data(file))

        if i % 100 == 0:
            print(f'Processed {i} files')

    return pd.concat(dfs, axis=0)


def main():
    files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]

    pool = mp.Pool()
    chunk_size = len(files) // mp.cpu_count()
    result = pool.map(process_files, [files[i:i+chunk_size]
                      for i in range(0, len(files), chunk_size)])
    pool.close()
    pool.join()

    df = pd.concat(result, axis=0)

    with open(os.path.join(current_dir, 'df.pkl'), 'wb') as f:
        pkl.dump(df, f)


if __name__ == '__main__':
    main()