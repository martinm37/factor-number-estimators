
"""
python code for Wei and Chen estimator
using parallel loops by numba
"""

"""
***
do a time benchmarking to see if the
changes you want to implement are worth it,
or if its not gonna reduce computation time a lot
"""


"""
current issue:
NumbaPerformanceWarning: '@' is faster on contiguous arrays, 
called on (Array(float64, 2, 'C', False, aligned=True), Array(float64, 2, 'A', False, aligned=True))
  XRjk_hat = LAMBDAj_kR @ F_jRk.T

"""




# importing packages
# ------------------
import numpy as np
import numba

# function for CV(R)
# ------------------


@numba.jit(nopython = True)
def CV_R_fun(X, j, k, R):
    """
    X - panel data
    j - width of cross sectional folds
    k - width of temporal folds
    R - assumed number of true factors
    """

    # setting dimensions and preallocating the output matrix
    # ****************************
    N, T = X.shape
    CV_R_mat = np.zeros((int(N / j), int(T / k)))

    # twice k-fold CV
    # ****************************
    for n in range(1, int(N / j) + 1):

        # loop over cross section

        """
        *** this loop runs in parallel ***        
        """

        # Step 1
        # -------------------------

        """
        X_j should be created at once in a smart way
        """

        # creating matrices Xj and X_j
        Xj = X[j * (n - 1):j * (n), :]
        X_j1 = X[0:j * (n - 1), :]
        X_j2 = X[j * (n):, :]
        X_j = np.concatenate((X_j1, X_j2), axis=0)

        # estimating factors from X_j.T @ X_j
        Cov_XX = X_j.T @ X_j
        eigenval, eigenvec = np.linalg.eigh(Cov_XX)

        # sorting eigenvectors in descending order
        index = np.argsort(eigenval)[::-1]
        eigenvec_sort = eigenvec[:, index]

        F_jR = np.sqrt(T) * eigenvec_sort[:, 0:R]

        for t in range(1, int(T / k) + 1):
            # loop over time

            # Step 2
            # -------------------------

            """
            Xj_k should be created at once in a smart way
            """

            # creating matrices Xjk and Xj_k
            Xjk = Xj[:, k * (t - 1):k * (t)]

            Xj_k1 = Xj[:, 0:k * (t - 1)]
            Xj_k2 = Xj[:, k * (t):]
            Xj_k = np.concatenate((Xj_k1, Xj_k2), axis=1)

            # creating matrices F_jRk and F_jR_k
            F_jRk = F_jR[k * (t - 1):k * (t), :]

            F_jR_k1 = F_jR[0:k * (t - 1), :]
            F_jR_k2 = F_jR[k * (t):, :]
            F_jR_k = np.concatenate((F_jR_k1, F_jR_k2), axis=0)

            """
            here use a faster OLS method without inv, but with solve
            """
            """
            xA = B -> x = B @ np.linalg.inv(A)
            B = Xj_k @ F_jR_k
            A = F_jR_k.T @ F_jR_k
            .......
            x = np.linalg.solve(A.T,B.T).T
            x = np.linalg.solve((F_jR_k.T @ F_jR_k).T , (Xj_k @ F_jR_k).T ).T
            """

            # estimating factor loading matrix LAMBDAj_kR by OLS

            LAMBDAj_kR = Xj_k @ F_jR_k @ np.linalg.inv(F_jR_k.T @ F_jR_k)
            # LAMBDAj_kR = np.linalg.solve((F_jR_k.T @ F_jR_k).T, (Xj_k @ F_jR_k).T).T

            # Step 3
            # -------------------------

            # factor estimation
            # fixing the numba issue just in this place
            # for my use case .copy() is faster than np.ascontiguousarray()
            LAMBDAj_kR = LAMBDAj_kR.copy()
            F_jRk = F_jRk.copy()

            XRjk_hat = LAMBDAj_kR @ F_jRk.T

            # error of the estimation
            diff = Xjk - XRjk_hat

            # squared frobenius norm
            CVijR = np.trace(diff @ diff.T)
            # np.trace((Xjk - XRjk_hat) @ (Xjk - XRjk_hat).T)

            # export
            CV_R_mat[n - 1, t - 1] = CVijR
        
    
    # summing over the two sums and exporting
    # ****************************
    #CV_R = np.sum(CV_R_mat,axis=None) # numba supports numpy sum only along a single axis, not a tuple of them

    return CV_R_mat


# function for the estimator
# --------------------------

"""
potentially this could be parallelized as well, but I am not sure
if this would not be too much paralellizations
"""

def TKCV_fun(X, fold_number, k_max):

    # default for the fold number is 5

    N, T = X.shape

    j = int(N / fold_number)
    k = int(T / fold_number)

    output_vec = np.zeros((k_max, 1))
    for i in range(0, k_max):
        CV_R_mat_i = CV_R_fun(X, j, k, i + 1)
        output_vec[i] = np.sum(CV_R_mat_i,axis=None)

    return np.argmin(output_vec) + 1 , output_vec