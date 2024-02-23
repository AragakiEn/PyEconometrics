# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 02:26:18 2024

@author: Yuan Hu
"""


import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap, CircularBlockBootstrap, IIDBootstrap, optimal_block_length

def calculate_statistic(bootstrapped_data: pd.DataFrame) -> float:
    """
    Calculates the test statistics using bootstrapped data.
    Example here is to calculate the max value of included financial variable.

    Parameters:
    - bootstrapped_data 

    Returns:
    - float: the test statistics.
    """
    
    return bootstrapped_data.mean().max()


def bs_fun(df: pd.DataFrame, var_names: list, n_sample: int=10000, bs_type: str='s') -> pd.DataFrame:
    """
    Applies a bootstrap method to a DataFrame containing financial time series data and computes confidence intervals.

    Parameters:
    - df (pd.DataFrame)
    - var_names (list): column names of the financial variables
    - n_sample (int, optional): The number of bootstrap samples to draw. Defaults to 10.
    - bs_type (str, optional): The type of bootstrap to perform ('s' for Stationary, 'c' for Circular, 'i' for IID).

    Returns:
    - pd.DataFrame: Output table for confidence interval.
    """
    # Determine optimal block length for the stationary bootstrap.
    (length_stationary, length_circular) = tuple(optimal_block_length(df[var_names]).max().values)
    
    # Perform bootstrap depending on the specified type and calculate confidence intervals.
    if bs_type == 's':
        stationary_bs = StationaryBootstrap(length_stationary, df[var_names],
                                            random_state=np.random.RandomState(len(var_names) * [2021]))
        results = stationary_bs.apply(calculate_statistic, n_sample)
    elif bs_type == 'c':
        circular_bs = CircularBlockBootstrap(length_circular, df[var_names],
                                             random_state=np.random.RandomState(len(var_names) * [2021]))
        results = circular_bs.apply(calculate_statistic, n_sample)
    elif bs_type == 'i':
        iid_bs = IIDBootstrap(df[var_names], random_state=np.random.RandomState(len(var_names) * [2021]))
        results = iid_bs.apply(calculate_statistic, n_sample)
    else:
        raise ValueError("Invalid bootstrap type specified: choose 's', 'c', or 'i'.")

    confidence_interval = np.percentile(results, [2.5, 97.5], axis=0)
    return pd.DataFrame({'lower': confidence_interval[0], 'upper': confidence_interval[1]})

# Example usage 
# df is a DataFrame with columns ['X1', 'X2'], type is a group index variable
# stationary_bs_df = df.groupby(['type']).apply(lambdf x: bs_fun(x, var_names=['X1', 'X2'], n_sample=10000, bs_type='s'))
# stationary_bs_df = stationary_bs_df.reset_index().drop(columns='level_1') 
