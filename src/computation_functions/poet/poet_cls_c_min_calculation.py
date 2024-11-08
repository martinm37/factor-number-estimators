
"""
computes the minimum possible value of C* - denoted my c_min
"""

import numpy as np

from src.estimators.covariance_matrix.poet_cls_thresholder import poet_cls_thresholder

def poet_cls_c_min_calc(u_matrix, C_min_start, C_min_end, C_grid_precision):


    C_vector = np.arange(C_min_start,C_min_end,C_grid_precision)
    C_matrix = np.concatenate([C_vector.reshape(-1,1), np.zeros(len(C_vector)).reshape(-1,1)],axis = 1)

    for c in range(len(C_vector)):

        Sigma_u_thresholded_matrix = poet_cls_thresholder(u_matrix=u_matrix, C=C_matrix[c, 0])

        eigen_vals = np.linalg.eigvalsh(Sigma_u_thresholded_matrix)
        index = np.argsort(eigen_vals)[::-1]
        eigen_vals_sort = eigen_vals[index]

        lambda_min = np.min(eigen_vals_sort)

        C_matrix[c,1] = lambda_min

    C_matrix_filter = C_matrix[C_matrix[:,1] > 0]

    argmin = np.argmin(C_matrix_filter[:, 1])
    C_min = np.round(C_matrix_filter[argmin,0],decimals = 2)


    return C_min
