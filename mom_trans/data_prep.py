
import os
import numpy as np
import pandas as pd
from mom_trans.classical_strategies import (
    MACDStrategy,
    calc_returns,
    calc_second_vol,
    calc_vol_scaled_returns,
)

VOL_THRESHOLD = 5
HALFLIFE_WINSORISE = 252
SEC_PER_DAY = 23400

def read_changepoint_results_and_fill_na(
    file_path: str, lookback_window_length: int
) -> pd.DataFrame:
    """Read output data from changepoint detection module into a dataframe.
    For rows where the module failed, information for changepoint location and severity is
    filled using the previous row.


    Args:
        file_path (str): the file path of the csv containing the results
        lookback_window_length (int): lookback window length - necessary for filling in the blanks for norm location

    Returns:
        pd.DataFrame: changepoint severity and location information
    """

    return (
        pd.read_csv(file_path, index_col=0, parse_dates=True)
        .ffill()
        .dropna()  # if first values are na
        .assign(
            cp_location_norm=lambda row: (row["t"] - row["cp_location"])
            / lookback_window_length
        )  # fill by assigning the previous cp and score, then recalculate norm location
    )


def prepare_cpd_features(folder_path: str, lookback_window_length: int) -> pd.DataFrame:
    """Read output data from changepoint detection module for all assets into a dataframe.


    Args:
        file_path (str): the folder path containing csvs with the CPD the results
        lookback_window_length (int): lookback window length

    Returns:
        pd.DataFrame: changepoint severity and location information for all assets
    """

    return pd.concat(
        [
            read_changepoint_results_and_fill_na(
                os.path.join(folder_path, f), lookback_window_length
            ).assign(ticker=os.path.splitext(f)[0])
            for f in os.listdir(folder_path)
        ]
    )


def deep_momentum_strategy_features(df_asset: pd.DataFrame) -> pd.DataFrame:
    """prepare input features for deep learning model

    Args:
        df_asset (pd.DataFrame): time-series for asset with column mid

    Returns:
        pd.DataFrame: input features
    """
    df_asset = df_asset[
        ~df_asset["mid"].isna()
        | ~df_asset["mid"].isnull()
        | (df_asset["mid"] > 1e-8)  # price is zero
    ].copy()

    # winsorize using rolling 5X standard deviations to remove outliers
    df_asset["srs"] = df_asset["mid"]
    ewm = df_asset["srs"].ewm(halflife=HALFLIFE_WINSORISE)
    means = ewm.mean()
    stds = ewm.std()
    df_asset["srs"] = np.minimum(df_asset["srs"], means + VOL_THRESHOLD * stds)
    df_asset["srs"] = np.maximum(df_asset["srs"], means - VOL_THRESHOLD * stds)

    df_asset["second_returns"] = calc_returns(df_asset["srs"])
    df_asset["second_vol"] = calc_second_vol(df_asset["second_returns"])
    df_asset["target_returns"] = calc_vol_scaled_returns(
        df_asset["second_returns"], df_asset["second_vol"]
    ).shift(-1)

    def calc_normalised_returns(sec_offset):
        return (
            calc_returns(df_asset["srs"], sec_offset)
            / df_asset["second_vol"]
            / np.sqrt(sec_offset)
        )
    df_asset["norm_second_return"] = calc_normalised_returns(1)
    df_asset["norm_minute_return"] = calc_normalised_returns(60)
    df_asset["norm_hourly_return"] = calc_normalised_returns(3600)
    df_asset["norm_daily_return"] = calc_normalised_returns(SEC_PER_DAY)
    df_asset["norm_monthly_return"] = calc_normalised_returns(21*SEC_PER_DAY)
    df_asset["norm_quarterly_return"] = calc_normalised_returns(63*SEC_PER_DAY)
    df_asset["norm_biannual_return"] = calc_normalised_returns(126*SEC_PER_DAY)
    df_asset["norm_annual_return"] = calc_normalised_returns(252*SEC_PER_DAY)

    trend_combinations = [
        (60, 300),         # 1m / 5m
        (300, 900),        # 5m / 15m
        (600, 1800),       # 10m / 30m
        (1800, 7200),      # 30m / 2h
        (3600, 14400),     # 1h / 4h
        (7200, 18000),     # 2h / 5h
        (14400, 23400),    # 4h / 1d
        (23400, 117000),   # 1d / 5d
        (187200, 561600),  # 8d / 24d
        (374400, 1123200), # 16d / 48d
        (748800, 2246400), # 32d / 96d
    ]
    for short_window, long_window in trend_combinations:
        df_asset[f"macd_{short_window}_{long_window}"] = MACDStrategy.calc_signal(
            df_asset["srs"], short_window, long_window
        )

    # date features
    if isinstance(df_asset.index, pd.MultiIndex):
        # Drop one level of MultiIndex
        df_asset.index = df_asset.index.droplevel(0)
        # If still MultiIndex, reset completely
        if isinstance(df_asset.index, pd.MultiIndex):
            df_asset = df_asset.reset_index()
        df_asset.index = pd.to_datetime(df_asset.index)

    if len(df_asset):
        df_asset_index = pd.to_datetime(df_asset.index)
        df_asset['hour_of_day'] = df_asset_index.hour
        df_asset['minute_of_hour'] = df_asset_index.minute
        df_asset['second_of_minute'] = df_asset_index.second
        df_asset['day_of_week'] = df_asset_index.dayofweek
        df_asset['day_of_month'] = df_asset_index.day
        df_asset['week_of_year'] = df_asset_index.isocalendar().week
        df_asset['month_of_year'] = df_asset_index.month
        df_asset['year'] = df_asset_index.year
        df_asset['date'] = df_asset_index  # duplication but sometimes makes life easier
    else:
        df_asset['hour_of_day'] = []
        df_asset['minute_of_hour'] = []
        df_asset['second_of_minute'] = []
        df_asset['day_of_week'] = []
        df_asset['day_of_month'] = []
        df_asset['week_of_year'] = []
        df_asset['month_of_year'] = []
        df_asset['year'] = []
        df_asset['date'] = []

    return df_asset.dropna()


def include_changepoint_features(
    features: pd.DataFrame, cpd_folder_name: pd.DataFrame, lookback_window_length: int
) -> pd.DataFrame:
    """combine CP features and DMN featuress

    Args:
        features (pd.DataFrame): features
        cpd_folder_name (pd.DataFrame): folder containing CPD results
        lookback_window_length (int): LBW used for the CPD

    Returns:
        pd.DataFrame: features including CPD score and location
    """
    features = features.merge(
        prepare_cpd_features(cpd_folder_name, lookback_window_length)[
            ["ticker", "cp_location_norm", "cp_score"]
        ]
        .rename(
            columns={
                "cp_location_norm": f"cp_rl_{lookback_window_length}",
                "cp_score": f"cp_score_{lookback_window_length}",
            }
        )
        .reset_index(),  # for date column
        on=["date", "ticker"],
    )

    features.index = features["date"]

    return features
