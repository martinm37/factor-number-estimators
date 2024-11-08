
"""
POET - principal orthogonal complement thresholding method
- estimator of the covariance matrix by Fan, J., Liao, Y., & Mincheva, M. (2013)

- old file, using the spectral decomposition /
  principal component analysis approach to POET
- depreciated in favor of constrained least squares approach
"""

import numpy as np
import numba


@numba.jit(nopython = True, fastmath=True)
def numba_matmul(A,B):
    return A @ B


@numba.jit(nopython = True)
def poet_fun(covariance_matrix, N, T, K, C):

    # singular value decomposition
    # -----------------------

    U, S, Vh = np.linalg.svd(covariance_matrix, full_matrices=True)

    """
    S (Nx1) contains the ordered eigenvalues of the covariance_matrix
    columns of U (NxN) are the corresponding eigenvectors of the covariance_matrix
    """

    # creating two sums
    # -----------------------

    Sum_K = np.zeros((N, N), dtype=numba.float64)
    for i in range(0, K):
        # this selects the first K decompositions
        vec1 = U[:, i].copy()
        vec1 = np.reshape(vec1, (-1, 1))
        vec2 = U[:, i].copy()
        vec2 = np.reshape(vec2, (1, -1))
        Sum_K += S[i] * numba_matmul(vec1,vec2)


    R_K = np.zeros((N, N), dtype=numba.float64)
    for i in range(K, N):
        # this selects the rest
        vec1 = U[:, i].copy()
        vec1 = np.reshape(vec1, (-1, 1))
        vec2 = U[:, i].copy()
        vec2 = np.reshape(vec2, (1, -1))
        R_K += S[i] * numba_matmul(vec1,vec2)

    # thresholding R_K
    # -----------------------

    omega_T = 1 / np.sqrt(N) + np.sqrt(np.log(N) / T)


    rii_rjj_matrix = np.zeros((N, N), dtype=numba.float64)
    for i in range(N):
        for j in range(N):
            if j < i :
                rii_rjj_matrix[i, j] = np.sqrt(R_K[i, i] * R_K[j, j])

    rii_rjj_matrix = rii_rjj_matrix + rii_rjj_matrix.T # filling the top rows as we are symmetric


    thresholding_matrix = C * omega_T * rii_rjj_matrix

    difference_matrix = np.abs(R_K) - thresholding_matrix

    censored_matrix = np.where(difference_matrix >= 0, difference_matrix, 0) # returns just values >= 0

    sgn_matrix = np.sign(R_K)

    R_K_thresholded = np.multiply(sgn_matrix, censored_matrix)  # hadamard product


    for i in range(N):
        R_K_thresholded[i,i] = R_K[i,i]


    # POET estimator
    # -----------------------

    POET_K = Sum_K + R_K_thresholded

    return POET_K


@numba.jit(nopython = True)
def r_k_threshold_fun(covariance_matrix, N, T, K, C):

    # singular value decomposition
    # -----------------------

    U, S, Vh = np.linalg.svd(covariance_matrix, full_matrices=True)

    """
    S (Nx1) contains the ordered eigenvalues of the covariance_matrix
    columns of U (NxN) are the corresponding eigenvectors of the covariance_matrix
    """

    R_K = np.zeros((N, N), dtype=numba.float64)
    for i in range(K, N):
        # this selects the rest
        vec1 = U[:, i].copy()
        vec1 = np.reshape(vec1, (-1, 1))
        vec2 = U[:, i].copy()
        vec2 = np.reshape(vec2, (1, -1))
        R_K += S[i] * numba_matmul(vec1,vec2)


    # thresholding R_K
    # -----------------------

    omega_T = 1 / np.sqrt(N) + np.sqrt(np.log(N) / T)

    rii_rjj_matrix = np.zeros((N, N), dtype=numba.float64)
    for i in range(N):
        for j in range(N):
            if j < i:
                rii_rjj_matrix[i, j] = np.sqrt(R_K[i, i] * R_K[j, j])

    """
    i do not care about the on diagonal elements anyway
    -> i can just generate for j < i, and then transpose it and sum it
    -> i should save half of the oprations
    """

    rii_rjj_matrix = rii_rjj_matrix + rii_rjj_matrix.T  # filling the top rows as we are symmetric

    thresholding_matrix = C * omega_T * rii_rjj_matrix

    difference_matrix = np.abs(R_K) - thresholding_matrix

    censored_matrix = np.where(difference_matrix >= 0, difference_matrix, 0)  # returns just values >= 0

    sgn_matrix = np.sign(R_K)

    R_K_thresholded = np.multiply(sgn_matrix, censored_matrix)  # hadamard product


    for i in range(N):
        R_K_thresholded[i,i] = R_K[i,i]

    return R_K_thresholded




@numba.jit(nopython = True)
def r_k_raw_fun(covariance_matrix, N, K):

    # singular value decomposition
    # -----------------------

    U, S, Vh = np.linalg.svd(covariance_matrix, full_matrices=True)

    """
    S (Nx1) contains the ordered eigenvalues of the covariance_matrix
    columns of U (NxN) are the corresponding eigenvectors of the covariance_matrix
    """

    R_K = np.zeros((N, N), dtype=numba.float64)
    for i in range(K, N):
        # this selects the rest
        vec1 = U[:, i].copy()
        vec1 = np.reshape(vec1, (-1, 1))
        vec2 = U[:, i].copy()
        vec2 = np.reshape(vec2, (1, -1))
        R_K += S[i] * numba_matmul(vec1,vec2)



    return R_K

