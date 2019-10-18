import pandas as pd
import numpy as np
import datetime as dt
from loguru import logger


def simple_moving_average(frame: pd.DataFrame, column_name: str = 'Adj Close',
                          window: int = 14, add_to_frame: bool = 'True') -> pd.Series:
    """

    Calculates SMA (Simple Moving Average).

    :param frame: DataFrame with Yahoo Finance data.
    :param column_name: Column name.
    :param window: Period for the RSI (days).
    :param add_to_frame: "True" adds a new column to frame, "False" returns a pd.Series object.
    :return: pandas Series.
    """

    df = frame.copy()
    roll_avg = df[column_name].rolling(window).mean()
    if add_to_frame:
        col_str = 'SMA_' + str(window)
        frame[col_str] = roll_avg
    else:
        return roll_avg


def relative_strength_index(frame: pd.DataFrame, column_name: str = 'Adj Close',
                            window: int = 20, technique: str = 'SMA', add_to_frame: bool = 'True') -> pd.Series:
    """
    Calculates RSI (Relative Strength Index) with either SMA or EWMA:
        * "SMA" for Simple Moving Average
        * "EWMA" for Exponential Moving Average).

    :param frame: DataFrame with Yahoo Finance data.
    :param column_name: Column name.
    :param window: int: Period for the RSI (days).
    :param technique: str: Options are "SMA" or EWMA".
    :param add_to_frame: "True" adds a new column to frame, "False" returns a pd.Series object.
    :return: pandas Series.
    """
    df = frame.copy()
    deltas = df[column_name].diff()
    deltas = deltas[1:]
    d_up, d_down = deltas.copy(), deltas.copy()
    d_up[d_up < 0] = 0
    d_down[d_down > 0] = 0
    rsi = 50

    if technique == 'EWMA':
        roll_up = d_up.ewm(span=window, freq='D').mean()
        roll_down = d_down.ewm(span=window, freq='D').mean()
        rs = roll_up / roll_down.abs()
        rsi = 100.0 - (100.0 / (1.0 + rs))
    elif technique == 'SMA':
        roll_up = d_up.rolling(window).mean()
        roll_down = d_down.rolling(window).mean()
        rs = roll_up / roll_down.abs()
        rsi = 100.0 - (100.0 / (1.0 + rs))
    else:
        logger.critical('Function was passed "' + technique +
                        '" as parameter. Needs to be either "EWMA" or "SMA". Aborted.')
        quit()

    if add_to_frame:
        col_str = 'RSI_' + str(window)
        frame[col_str] = rsi
    else:
        return rsi


def price_rise(frame: pd.DataFrame, column_name: str = 'Adj Close'):
    frame['Price_Rise'] = np.where(frame[column_name].shift(-1) > frame[column_name], 1, 0)


def add_fixed_features(df) -> pd.DataFrame:
    df['H-L'] = df['High'] - df['Low']
    df['O-C'] = df['Close'] - df['Open']
    relative_strength_index(frame=df, window=9)
    simple_moving_average(frame=df, window=3)
    simple_moving_average(frame=df, window=10)
    simple_moving_average(frame=df, window=30)
    df['Std_dev'] = df['Close'].rolling(5).std()
    price_rise(frame=df)
    return df
