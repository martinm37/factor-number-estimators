

import pandas as pd
import numpy as np


def daily_returns_fun(data_price):

    # calculating daily returns - V2

    data_daily_returns = data_price.iloc[1:, :].copy()

    time_lenght = len(data_daily_returns)

    column_names = data_daily_returns.columns.values.tolist()

    # columns_names = data_daily_returns.columns
    column_names.remove("UTCTIME") # prior to conversion to numpy

    # flushing
    for column_name in column_names:
        data_daily_returns[column_name] = np.zeros(time_lenght)  # flushing

    # creating 2 price df

    data_price_t0 = data_price.iloc[:-1, :].copy()
    data_price_t0_np = data_price_t0.iloc[:, 1:]
    data_price_t0_np = data_price_t0_np.to_numpy()

    data_price_t1 = data_price.iloc[1:, :].copy()
    data_price_t1_np = data_price_t1.iloc[:, 1:]
    data_price_t1_np = data_price_t1_np.to_numpy()

    daily_returns_np = (data_price_t1_np - data_price_t0_np) / data_price_t0_np * 100  # !!!!!!!!!

    # storing in the pandas df
    data_daily_returns.iloc[:, 1:] = daily_returns_np

    # removing the zero rows -> these are bank holidays

    dr_column_names = data_daily_returns.columns.values.tolist()

    dr_column_names_stocks = dr_column_names.copy()

    dr_column_names_stocks.remove("UTCTIME")

    data_daily_returns = data_daily_returns.loc[~(data_daily_returns[dr_column_names_stocks] == 0.0).all(axis=1)]
    # dropping all rows for which all stocks are zero on the given day

    return data_daily_returns




def correlations_list(data,cutoff):

    pairwise_correlations = data.corr()
    pairwise_correlations_np = pairwise_correlations.to_numpy()

    np.fill_diagonal(pairwise_correlations_np, 0)


    selection = pairwise_correlations[(pairwise_correlations >= cutoff)]
    selection = selection.dropna(axis="columns", how="all")
    selection = selection.dropna(axis="rows", how="all")


    corr_list = selection.columns.tolist()

    return corr_list


def max_correlation(data):

    pairwise_correlations = data.corr()
    pairwise_correlations_np = pairwise_correlations.to_numpy()

    np.fill_diagonal(pairwise_correlations_np, 0)

    return np.max(pairwise_correlations_np, axis=None)