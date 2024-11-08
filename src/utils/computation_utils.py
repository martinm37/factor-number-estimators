

import numpy as np


def time_demeaning(data):

    N, T = data.shape

    T_mean = np.mean(data, axis=1).reshape(-1, 1)  # time_mean -> column vector (one time mean for each of N observations)
    T_mean_matrix = np.ones([N, T]) * T_mean

    return data - T_mean_matrix


def double_demeaning(data):

    N, T = data.shape

    T_mean = np.mean(data, axis=1).reshape(-1, 1)  # time_mean - column vector -> one time mean for each N observation
    N_mean = np.mean(data, axis=0).reshape(1, -1)  # cross_sectional_mean - row vector -> one cross sectional mean for each time point
    NT_mean = 1 / (N * T) * np.sum(data, axis=None)  # mean across both dimensions

    T_mean_matrix = np.ones([N, T]) * T_mean
    N_mean_matrix = np.ones([N, T]) * N_mean
    NT_mean_matrix = np.ones([N, T]) * NT_mean

    # tmean_vec = np.mean(data, axis=1).reshape(-1, 1)
    # variance_vec = np.var(data, axis=1)

    return data - T_mean_matrix - N_mean_matrix + NT_mean_matrix



def time_demeaning_unit_variance(data):

    N, T = data.shape

    T_mean = np.mean(data, axis=1).reshape(-1, 1)  # time_mean -> column vector (one time mean for each of N observations)
    T_mean_matrix = np.ones([N, T]) * T_mean

    data = data - T_mean_matrix # time de meaning

    var_vec = np.var(data, axis=1)

    data = data / np.sqrt(var_vec).reshape(-1, 1) # standardizing to unit variance

    # tmean_vec = np.mean(data, axis=1).reshape(-1, 1)
    # variance_vec = np.var(data, axis=1)

    return data






"""
T_mean = np.mean(data, axis=0).reshape(1, -1)  # time_mean -> row vector
this should be axis=1 !!!! but it doesnt matter above as i substract both, but here it matters!!!
"""
