
"""
Onatski (2010) factor number estimator
ED - edge distribution
"""

# importing packages
# ------------------
import numpy as np


# estimator code
# --------------


def ED_fun(X, eigen_vals_sort, k_max):

    """
    editions:
    eig_vals_sort -> now modified within this file, not outside of it!!!
    """

    """
    inputs:    
    -------
    eig_vals_sort - sorted eigenvalues of (1/T)*X@X.T or (1/T)*X.T@X matrix of data X
    r_max - assumed maximum number of factors
    outputs:
    -------
    [0] - estimated factor number
    [1] - number of iterations
    """

    # step 1 - adjusting the eigenvalues -> no longer : eigenvalues computed and sorted outside of this function
    # ------
    N,T = X.shape

    eigen_vals_sort = (1 / T) * eigen_vals_sort
    eigen_vals_sort = eigen_vals_sort.reshape(-1, 1) # converting 1D array to 2D vector
    j = k_max + 1

    # step 2 - OLS
    # lambda_vec = j_mat * beta
    # ------------
    lambda_vec = eigen_vals_sort[j - 1:j + 4, :]
    j_vec = np.array([j-1,j,j+1,j+2,j+3]).reshape(-1,1)
    j_vec = j_vec ** (2/3)
    j_mat = np.concatenate((np.ones((len(j_vec),1)),j_vec),axis=1)

    # print(f"lambda vec shape: {lambda_vec.shape}")
    # print(f"jmat shape: {j_mat.shape}")

    beta_hat = np.linalg.inv(j_mat.T@j_mat) @ j_mat.T @ lambda_vec

    delta = 2 * abs(beta_hat[1][0])
    # [1] selects 2nd element of the 2x1 vector, returns 1x1 array
    # [0] selects the element of 1x1 array - returns a float

    # step 3 - computing r_hat
    # ------------------------
    evs_first = eigen_vals_sort[0:k_max, :]
    evs_second = eigen_vals_sort[1:k_max + 1, :]
    diff_vec = evs_first - evs_second

    # test condition
    if (np.all(diff_vec < delta)): 
        return 0,0

    else:
        diff_vec_delta = np.where(diff_vec >= delta,diff_vec,0)
        k_hat = np.argmax(diff_vec_delta) + 1

        # step 4 - repeat steps 2 and 3 until convergence in r_hat
        # --------------------------------------------------------
        j = k_hat + 1
        
        k_hat_iter = 0
        iter_count = 0

        while (k_hat != k_hat_iter):
                
                # initialization updating:
                # -----------------------
                k_hat = k_hat_iter # updating the old one to the current iterate

                # step 2 - OLS
                # lambda_vec = j_mat * beta
                # ------------
                lambda_vec = eigen_vals_sort[j - 1:j + 4, :]
                j_vec = np.array([j-1,j,j+1,j+2,j+3]).reshape(-1,1)
                j_vec = j_vec ** (2/3)
                j_mat = np.concatenate((np.ones((len(j_vec),1)),j_vec),axis=1)

                beta_hat = np.linalg.inv(j_mat.T@j_mat) @ j_mat.T @ lambda_vec

                delta = 2 * abs(beta_hat[1][0])
                # [1] selects 2nd element of the 2x1 vector, returns 1x1 array
                # [0] selects just the given number

                # step 3 - computing r_hat
                # ------------------------
                evs_first = eigen_vals_sort[0:k_max, :]
                evs_second = eigen_vals_sort[1:k_max + 1, :]
                diff_vec = evs_first - evs_second

                # test condition
                if (np.all(diff_vec < delta)): 
                    return 0,0
                
                else:
                    diff_vec_delta = np.where(diff_vec >= delta,diff_vec,0)
                    k_hat = np.argmax(diff_vec_delta) + 1

                    # step 4 - repeat steps 2 and 3 until convergence in r_hat
                    # --------------------------------------------------------
                    k_hat_iter = np.argmax(diff_vec_delta) + 1
                    
                    j = k_hat_iter + 1
                    iter_count += 1


        return k_hat_iter,iter_count





"""
i am not sure if i should add the 0 possibility also in the iterations
- for now, i will not add it
-- i think i should add it, for a high rho it goes to zero, it should be added just in case***
***
maybe also add a limit on how many iterations there can be,
On set it to max 4 iterations
"""

"""
new interesting fact:
the smallest dimension of the X matrix has to be at
least as big as r_max + 5
"""