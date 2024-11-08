
"""
POET - principal orthogonal complement thresholding method
- estimator of the covariance matrix by Fan, J., Liao, Y., & Mincheva, M. (2013)

- old file, using the spectral decomposition /
  principal component analysis approach to POET
- depreciated in favor of constrained least squares approach
"""

import numpy as np


def poet_fun(covariance_matrix,N,T,K,C):

    # singular value decomposition
    # -----------------------

    U, S, Vh = np.linalg.svd(covariance_matrix, full_matrices=True)

    """
    S (Nx1) contains the ordered eigenvalues of the covariance_matrix
    columns of U (NxN) are the corresponding eigenvectors of the covariance_matrix
    """

    # creating two sums
    # -----------------------

    Sum_K = np.zeros([N, N], dtype=np.float64)
    for i in range(0, K):
        # this selects the first K decompositions
        Sum_K += S[i] * U[:, [i]] @ U[:, [i]].T


    R_K = np.zeros([N, N], dtype=np.float64)
    for i in range(K, N):
        # this selects the rest
        R_K += S[i] * U[:, [i]] @ U[:, [i]].T

    # thresholding R_K
    # -----------------------

    omega_T = 1 / np.sqrt(N) + np.sqrt(np.log(N) / T)

    """
    i do not care about the on diagonal elements anyway
    -> i can just generate for j < i, and then transpose it and sum it
    -> then i save half of the operations
    """

    rii_rjj_matrix = np.zeros([N, N], dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if j < i :
                rii_rjj_matrix[i, j] = np.sqrt(R_K[i, i] * R_K[j, j])

    rii_rjj_matrix = rii_rjj_matrix + rii_rjj_matrix.T # filling the top rows as we are symmetric

    # rii_rjj_matrix_OLD = np.zeros([N, N], dtype=np.float64)
    # for i in range(N):
    #     for j in range(N):
    #         rii_rjj_matrix_OLD[i, j] = np.sqrt(R_K[i, i] * R_K[j, j])


    thresholding_matrix = C * omega_T * rii_rjj_matrix

    difference_matrix = np.abs(R_K) - thresholding_matrix

    censored_matrix = np.where(difference_matrix >= 0, difference_matrix, 0) # returns just values >= 0

    sgn_matrix = np.sign(R_K)

    R_K_thresholded = np.multiply(sgn_matrix, censored_matrix)  # hadamard product

    """
    this can be much improved, just the diagonal
    elements of R_K_thresholded_offdiagonal need 
    to be rewritten
    """

    diag_indeces = np.diag_indices_from(R_K_thresholded)
    # this is basically a tuple of np.arrays

    #diag_indeces_V2 = (np.arange(N),np.arange(N)) # a tuple

    R_K_thresholded[diag_indeces] = R_K[diag_indeces]


    # POET estimator
    # -----------------------

    POET_K = Sum_K + R_K_thresholded

    return POET_K



def r_k_threshold_fun(covariance_matrix, N, T, K, C):

    # singular value decomposition
    # -----------------------

    U, S, Vh = np.linalg.svd(covariance_matrix, full_matrices=True)

    """
    S (Nx1) contains the ordered eigenvalues of the covariance_matrix
    columns of U (NxN) are the corresponding eigenvectors of the covariance_matrix
    """


    R_K = np.zeros([N, N], dtype=np.float64)
    for i in range(K, N):
        # this selects the rest
        R_K += S[i] * U[:, [i]] @ U[:, [i]].T

    # thresholding R_K
    # -----------------------

    omega_T = 1 / np.sqrt(N) + np.sqrt(np.log(N) / T)



    rii_rjj_matrix = np.zeros([N, N], dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if j < i:
                rii_rjj_matrix[i, j] = np.sqrt(R_K[i, i] * R_K[j, j])

    rii_rjj_matrix = rii_rjj_matrix + rii_rjj_matrix.T  # filling the top rows as we are symmetric


    thresholding_matrix = C * omega_T * rii_rjj_matrix

    difference_matrix = np.abs(R_K) - thresholding_matrix

    censored_matrix = np.where(difference_matrix >= 0, difference_matrix, 0)  # returns just values >= 0

    sgn_matrix = np.sign(R_K)

    R_K_thresholded = np.multiply(sgn_matrix, censored_matrix)  # hadamard product


    diag_indeces = np.diag_indices_from(R_K_thresholded)

    R_K_thresholded[diag_indeces] = R_K[diag_indeces]


    return R_K_thresholded







