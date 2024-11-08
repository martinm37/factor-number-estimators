

"""
POET - principal orthogonal complement thresholding method
- estimator of the covariance matrix by Fan, J., Liao, Y., & Mincheva, M. (2013)

- based on the constrained least squares method

this part of code thresholds the covariance matrix
"""


import numpy as np
import numba


@numba.jit(nopython=True)
def poet_cls_thresholder(u_matrix, C):

    """
    :param u_matrix: matrix of errors
    :param C: thresholding constant (C*)
    :return: Sigma_u_thresholded_matrix : thresholded covariance matrix of errors
    """

    """
    this function receives error matrix from the constrained least squares (cls) regression,
    creates the covariance matrix out of it, and thresholds it    
    -> it does not receive a covariance matrix !!!!!! just the error matrix
    """

    N_dim, T_dim = u_matrix.shape

    Sigma_u_matrix = (1/T_dim) * u_matrix @ u_matrix.T #(N x N)

    Theta_ij_matrix = (1/T_dim) * (u_matrix * u_matrix) @ (u_matrix * u_matrix).T - Sigma_u_matrix * Sigma_u_matrix

    omega_T = 1 / np.sqrt(N_dim) + np.sqrt(np.log(N_dim) / T_dim)

    Tau_ij_matrix = C * omega_T * np.sqrt(Theta_ij_matrix)


    sign_matrix = np.sign(Sigma_u_matrix)
    difference_matrix = np.abs(Sigma_u_matrix) - Tau_ij_matrix
    censored_matrix = np.where(difference_matrix >= 0, difference_matrix, 0)  # returns just values >= 0

    s_ij_matrix = sign_matrix * censored_matrix

    Sigma_u_thresholded_matrix = s_ij_matrix

    for i in range(N_dim):
        Sigma_u_thresholded_matrix[i,i] = Sigma_u_matrix[i,i]


    return Sigma_u_thresholded_matrix



# diag_indeces = np.diag_indices_from(Sigma_u_matrix)
# Sigma_u_thresholded_matrix[diag_indeces] = Sigma_u_matrix[diag_indeces]





