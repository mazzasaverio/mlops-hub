from numba import njit, prange
import numpy as np
import pandas as pd
from itertools import combinations, product


def generate_macd(df):
    # Define lists of price and size-related column names
    prices = [
        "reference_price",
        "far_price",
        "near_price",
        "ask_price",
        "bid_price",
        "wap",
    ]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]

    for stock_id, values in df.groupby(["stock_id"])[prices]:
        macd_values, signal_line_values, histogram_values = calculate_macd(
            values.values
        )
        col_macd = [f"macd_{col}" for col in values.columns]
        col_signal = [f"macd_sig_{col}" for col in values.columns]
        col_hist = [f"macd_hist_{col}" for col in values.columns]

        df.loc[values.index, col_macd] = macd_values
        df.loc[values.index, col_signal] = signal_line_values
        df.loc[values.index, col_hist] = histogram_values

    return df


@njit(fastmath=True)
def rolling_average(arr, window):
    """
    Calculate the rolling average for a 1D numpy array.

    Parameters:
    arr (numpy.ndarray): Input array to calculate the rolling average.
    window (int): The number of elements to consider for the moving average.

    Returns:
    numpy.ndarray: Array containing the rolling average values.
    """
    n = len(arr)
    result = np.empty(n)
    result[
        :window
    ] = np.nan  # Padding with NaN for elements where the window is not full
    cumsum = np.cumsum(arr)

    for i in range(window, n):
        result[i] = (cumsum[i] - cumsum[i - window]) / window

    return result


@njit(parallel=True)
def compute_rolling_averages(df_values, window_sizes):
    """
    Calculate the rolling averages for multiple window sizes in parallel.

    Parameters:
    df_values (numpy.ndarray): 2D array of values to calculate the rolling averages.
    window_sizes (List[int]): List of window sizes for the rolling averages.

    Returns:
    numpy.ndarray: A 3D array containing the rolling averages for each window size.
    """
    num_rows, num_features = df_values.shape
    num_windows = len(window_sizes)
    rolling_features = np.empty((num_rows, num_features, num_windows))

    for feature_idx in prange(num_features):
        for window_idx, window in enumerate(window_sizes):
            rolling_features[:, feature_idx, window_idx] = rolling_average(
                df_values[:, feature_idx], window
            )

    return rolling_features


@njit(parallel=True)
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    rows, cols = data.shape
    macd_values = np.empty((rows, cols))
    signal_line_values = np.empty((rows, cols))
    histogram_values = np.empty((rows, cols))

    for i in prange(cols):
        short_ema = np.zeros(rows)
        long_ema = np.zeros(rows)

        for j in range(1, rows):
            short_ema[j] = (data[j, i] - short_ema[j - 1]) * (
                2 / (short_window + 1)
            ) + short_ema[j - 1]
            long_ema[j] = (data[j, i] - long_ema[j - 1]) * (
                2 / (long_window + 1)
            ) + long_ema[j - 1]

        macd_values[:, i] = short_ema - long_ema

        signal_line = np.zeros(rows)
        for j in range(1, rows):
            signal_line[j] = (macd_values[j, i] - signal_line[j - 1]) * (
                2 / (signal_window + 1)
            ) + signal_line[j - 1]

        signal_line_values[:, i] = signal_line
        histogram_values[:, i] = macd_values[:, i] - signal_line

    return macd_values, signal_line_values, histogram_values


def generate_rsi(df):
    # Define lists of price and size-related column names
    prices = [
        "reference_price",
        "far_price",
        "near_price",
        "ask_price",
        "bid_price",
        "wap",
    ]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]

    for stock_id, values in df.groupby(["stock_id"])[prices]:
        columns = [f"rsi_{col}" for col in values.columns]
        data = calculate_rsi(values.values)
        df.loc[values.index, columns] = data

    return df


@njit(parallel=True)
def calculate_rsi(prices, period=14):
    rsi_values = np.zeros_like(prices)

    for col in prange(prices.shape[1]):
        price_data = prices[:, col]
        delta = np.zeros_like(price_data)
        delta[1:] = price_data[1:] - price_data[:-1]
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])

        if avg_loss != 0:
            rs = avg_gain / avg_loss
        else:
            rs = 1e-9  # or any other appropriate default value

        rsi_values[:period, col] = 100 - (100 / (1 + rs))

        for i in prange(period - 1, len(price_data) - 1):
            avg_gain = (avg_gain * (period - 1) + gain[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss[i]) / period
            if avg_loss != 0:
                rs = avg_gain / avg_loss
            else:
                rs = 1e-9  # or any other appropriate default value
            rsi_values[i + 1, col] = 100 - (100 / (1 + rs))

    return rsi_values


@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))

    for i in prange(num_combinations):
        a, b, c = comb_indices[i]
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = (
                df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
            )
            if mid_val == min_val:  # Prevent division by zero
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features


def calculate_triplet_imbalance_numba(price, df):
    # Convert DataFrame to numpy array for Numba compatibility
    df_values = df[price].values
    comb_indices = [
        (price.index(a), price.index(b), price.index(c))
        for a, b, c in combinations(price, 3)
    ]

    # Calculate the triplet imbalance
    features_array = compute_triplet_imbalance(df_values, comb_indices)

    # Create a DataFrame from the results
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)

    return features


def convert_weights_to_dict():
    weights = [
        0.004,
        0.001,
        0.002,
        0.006,
        0.004,
        0.004,
        0.002,
        0.006,
        0.006,
        0.002,
        0.002,
        0.008,
        0.006,
        0.002,
        0.008,
        0.006,
        0.002,
        0.006,
        0.004,
        0.002,
        0.004,
        0.001,
        0.006,
        0.004,
        0.002,
        0.002,
        0.004,
        0.002,
        0.004,
        0.004,
        0.001,
        0.001,
        0.002,
        0.002,
        0.006,
        0.004,
        0.004,
        0.004,
        0.006,
        0.002,
        0.002,
        0.04,
        0.002,
        0.002,
        0.004,
        0.04,
        0.002,
        0.001,
        0.006,
        0.004,
        0.004,
        0.006,
        0.001,
        0.004,
        0.004,
        0.002,
        0.006,
        0.004,
        0.006,
        0.004,
        0.006,
        0.004,
        0.002,
        0.001,
        0.002,
        0.004,
        0.002,
        0.008,
        0.004,
        0.004,
        0.002,
        0.004,
        0.006,
        0.002,
        0.004,
        0.004,
        0.002,
        0.004,
        0.004,
        0.004,
        0.001,
        0.002,
        0.002,
        0.008,
        0.02,
        0.004,
        0.006,
        0.002,
        0.02,
        0.002,
        0.002,
        0.006,
        0.004,
        0.002,
        0.001,
        0.02,
        0.006,
        0.001,
        0.002,
        0.004,
        0.001,
        0.002,
        0.006,
        0.006,
        0.004,
        0.006,
        0.001,
        0.002,
        0.004,
        0.006,
        0.006,
        0.001,
        0.04,
        0.006,
        0.002,
        0.004,
        0.002,
        0.002,
        0.006,
        0.002,
        0.002,
        0.004,
        0.006,
        0.006,
        0.002,
        0.002,
        0.008,
        0.006,
        0.004,
        0.002,
        0.006,
        0.002,
        0.004,
        0.006,
        0.002,
        0.004,
        0.001,
        0.004,
        0.002,
        0.004,
        0.008,
        0.006,
        0.008,
        0.002,
        0.004,
        0.002,
        0.001,
        0.004,
        0.004,
        0.004,
        0.006,
        0.008,
        0.004,
        0.001,
        0.001,
        0.002,
        0.006,
        0.004,
        0.001,
        0.002,
        0.006,
        0.004,
        0.006,
        0.008,
        0.002,
        0.002,
        0.004,
        0.002,
        0.04,
        0.002,
        0.002,
        0.004,
        0.002,
        0.002,
        0.006,
        0.02,
        0.004,
        0.002,
        0.006,
        0.02,
        0.001,
        0.002,
        0.006,
        0.004,
        0.006,
        0.004,
        0.004,
        0.004,
        0.004,
        0.002,
        0.004,
        0.04,
        0.002,
        0.008,
        0.002,
        0.004,
        0.001,
        0.004,
        0.006,
        0.004,
    ]

    weights = {int(k): v for k, v in enumerate(weights)}
    return weights


def global_stock_id_feats(df_train):
    return {
        "median_size": df_train.groupby("stock_id")["bid_size"].median()
        + df_train.groupby("stock_id")["ask_size"].median(),
        "std_size": df_train.groupby("stock_id")["bid_size"].std()
        + df_train.groupby("stock_id")["ask_size"].std(),
        "ptp_size": df_train.groupby("stock_id")["bid_size"].max()
        - df_train.groupby("stock_id")["bid_size"].min(),
        "median_price": df_train.groupby("stock_id")["bid_price"].median()
        + df_train.groupby("stock_id")["ask_price"].median(),
        "std_price": df_train.groupby("stock_id")["bid_price"].std()
        + df_train.groupby("stock_id")["ask_price"].std(),
        "ptp_price": df_train.groupby("stock_id")["bid_price"].max()
        - df_train.groupby("stock_id")["ask_price"].min(),
    }
