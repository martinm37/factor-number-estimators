

import numpy as np

from src.estimators.covariance_matrix.old.poet_numba import r_k_threshold_fun


def m_calculation(covariance_matrix, N, T, K, C_min, C_grid_precision):

    m_test = C_min

    while True: # or some prespecified limit on m

        #POET_K = poet_fun(covariance_matrix, N, T, K, m_test)
        R_K_threshold = r_k_threshold_fun(covariance_matrix, N, T, K, m_test)
        m_test += C_grid_precision

        if np.all(R_K_threshold == np.diag(np.diag(R_K_threshold))): # matrix is diagonal
            break

    return np.round(m_test, decimals = 2) # so that I dont have 3.300000016
