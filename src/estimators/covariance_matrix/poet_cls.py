
"""
POET - principal orthogonal complement thresholding method
- estimator of the covariance matrix by Fan, Liao and Mincheva, 2013

"""

import numpy as np


def poet_fun_cls(data,N,T,K,C):

    # constrained least squares
    # -------------------------
    gram_matrix = data.T @ data

    eigen_vals, eigen_vecs = np.linalg.eigh(gram_matrix)

    index = np.argsort(eigen_vals)[::-1]
    # eigen_val_sort = eigen_vals[index]
    eigen_vecs_sort = eigen_vecs[:, index]

    F_hat = np.sqrt(T) * eigen_vecs_sort[:, :K] # (T x k)

    L_hat = (1/T) * data @ F_hat                # (N x k)

    error_mat = data - L_hat @ F_hat.T

    #Sigma_u = (1/T) * error_mat


    # thresholding Sigma_u
    # -----------------------

    omega_T = 1 / np.sqrt(N) + np.sqrt(np.log(N) / T)

    """
    i do not care about the on diagonal elements anyway
    -> i can just generate for j < i, and then transpose it and sum it
    -> i should save half of the oprations
    """

    Sigma_ij_matrix = (1/T) * error_mat @ error_mat.T

    Theta_ij_matrix = (1/T) * (error_mat * error_mat) @ (error_mat * error_mat).T \
    - (1/T)**2 * (error_mat @ error_mat.T ) * (error_mat @ error_mat.T)



    thresholding_matrix = C * omega_T * np.sqrt(Theta_ij_matrix)

    difference_matrix = np.abs(Sigma_ij_matrix) - thresholding_matrix

    censored_matrix = np.where(difference_matrix >= 0, difference_matrix, 0) # returns just values >= 0

    sgn_matrix = np.sign(Sigma_ij_matrix)

    Sigma_u_thresholded = np.multiply(sgn_matrix, censored_matrix)  # hadamard product

    """
    this can be much improved, just the diagonal
    elements of R_K_thresholded_offdiagonal need 
    to be rewritten
    """

    diag_indeces = np.diag_indices_from(Sigma_u_thresholded)
    # this is basically a tuple of np.arrays

    #diag_indeces_V2 = (np.arange(N),np.arange(N)) # a tuple

    Sigma_u_thresholded[diag_indeces] = Sigma_ij_matrix[diag_indeces]


    # POET estimator
    # -----------------------

    POET_K = L_hat @ L_hat.T + Sigma_u_thresholded

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




    # R_K_thresholded_offdiagonal = np.multiply(sgn_matrix, censored_matrix)  # hadamard product
    #
    # R_K_thresholded_OLD = np.zeros([N, N])
    #
    # for i in range(N):
    #     for j in range(N):
    #         if i == j:
    #             R_K_thresholded_OLD[i, j] = R_K[i, j]
    #         else:
    #             R_K_thresholded_OLD[i, j] = R_K_thresholded_offdiagonal[i, j]


    # result = np.allclose(R_K_thresholded_NEW, R_K_thresholded_OLD, atol=1e-8, rtol=1e-5)






