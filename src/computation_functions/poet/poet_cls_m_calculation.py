
"""
computes the maximum possible value of C* - denoted my m
"""

import numpy as np

from src.estimators.covariance_matrix.poet_cls_thresholder import poet_cls_thresholder


def poet_cls_m_calculation(u_matrix, C_min, C_grid_precision):

    m_test = C_min

    while True:  # or some pre-specified limit on m

        Sigma_u_thresholded_matrix = poet_cls_thresholder(u_matrix=u_matrix, C=m_test)
        m_test += C_grid_precision

        if np.all(Sigma_u_thresholded_matrix == np.diag(np.diag(Sigma_u_thresholded_matrix))):  # matrix is diagonal
            break

    return np.round(m_test, decimals=2)  # so that I don't have 3.300000016







