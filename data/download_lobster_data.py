# from settings.default import ALL_QUANDL_CODES
LOBSTER_DATETIME_COLUMNS = ['datetime', 'date', 'timestamp']
LOBSTER_TICKERS = ['AAPL', 'AMZN', 'GOOG', 'INTC', 'MSFT']

import os
import pandas as pd
from multiprocessing import Pool
import glob

LEVELS = 1


def read_lobster_file(filepath: str, levels: int=LEVELS) -> pd.DataFrame:
    df = pd.read_csv(filepath, header=0)

    columns = ['timestamp']
    for level in range(1, levels + 1): # TODO: decide if this is necessary
        columns.extend([
            f'ask_price_{level}', f'ask_size_{level}',
            f'bid_price_{level}', f'bid_size_{level}'
        ])
    df.columns = columns

    df['date'] = pd.to_datetime(filepath.split('/')[-1].split('_')[1])
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['timestamp'].astype(str))

    price_cols = [col for col in columns if 'price' in col]
    for col in price_cols:
        df[col] = (pd.to_numeric(df[col], errors='coerce') / 10000.0).astype('float64') # un-scale prices

    size_cols = [col for col in columns if 'size' in col]
    for col in size_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('int64')

    df = df[LOBSTER_DATETIME_COLUMNS + columns[1:]] # reorder columns to have datetime first

    return df


def read_lobster_folder(folderpath, workers: int=8) -> pd.DataFrame:
    files = [os.path.join(folderpath, f) for f in os.listdir(folderpath) if f.endswith('.csv')]
    with Pool(processes=workers) as pool:
        dfs = pool.map(read_lobster_file, files)

    try:
        return pd.concat(dfs, ignore_index=True).sort_values(by='datetime').reset_index(drop=True)
    except Exception as e:
        pass # TODO: log error or handle it appropriately


def main():
    for t in LOBSTER_TICKERS:
        print(t)
        try:
            output = read_lobster_folder(f'data/raw/_data_dwn_43_456_transformer_{t}_2024-01-01_2025-01-01_1_1') # TODO: change folder name to match data source

            if not os.path.exists(os.path.join('data', 'processed', t)):
                os.makedirs(os.path.join('data', 'processed', t))

            output['mid_price'] = ((output['ask_price_1'] + output['bid_price_1']) / 2.0).astype('float64')

            output.to_parquet(f'data/processed/{t}/prices.parquet', index=False)
        except BaseException as ex:
            print(ex)


if __name__ == "__main__":
    main()
