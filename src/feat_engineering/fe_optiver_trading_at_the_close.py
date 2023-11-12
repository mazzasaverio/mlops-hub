from numba import njit, prange
import numpy as np
import pandas as pd
from itertools import combinations, product


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
