
"""
Code for Ahn and Horenstein functions
"""

# importing packages
# ------------------
import numpy as np



"""
both functions take sorted eigenvalues of (1/(N*T)) * X@X.T or X.T@X matrix
- the multiplication is done outside of the functions

"""

# common part for both estimators
# eigen_val, eigen_vec = aux.eigen_fun(X)
# eigen_val_sort = aux.sorter_fun(eigen_val, eigen_vec,-1)[0]

# Ahn and Horenstein 2013
# -----------------------------------------------------------------------


"""
a common function V(k)

V(k) = 
k+1 + k+2 + k+3 + ... m

V(1) = 
2nd + 3rd + 4th + m-th
"""

def V_k_fun(eigen_val_sort,k,m):

    if (k == -1):
        mock_eigenvalue = np.sum(eigen_val_sort[0:m]) / np.log(m)  # mock_eigenvalue = V_k_fun(eigen_val_sort, 0, m) / np.log(m)
        V_0 = np.sum(eigen_val_sort[0:m])
        return V_0 + mock_eigenvalue

    else: # k >= 0
        V_k = np.sum(eigen_val_sort[k:m])
        return V_k

#V_k = Sum_{j=k+1}^{m} lambda_{j} , m = min{N,T}
# (k+1)th eigenvalue is indexed by k

# can be done in a vectorized form

# old code below:
# ---------------

# if k == 0:
#     V_k = np.sum(eigen_val_sort[k:m])
#     return V_k
#
# else: # k > 0
#     V_k = np.sum(eigen_val_sort[k+1:m])
#     return V_k


# the two above should be identical


# V_k = eigen_val_sort[k+1].copy()
# # use .copy() instead of =,
# # = is assignment operator and thus
# # eigen_val_sort[k] would be overwritten when using this loop:
#
# for i in range(k+2,m):
#     V_k += eigen_val_sort[i]
#
# return V_k






# ER
# -----------------------------------------------------------------------

def ER_k(eigenval_sort,k,m):

    if k == 0:
        mock_eigenvalue = V_k_fun(eigenval_sort,0,m) / np.log(m)
        return mock_eigenvalue / eigenval_sort[k] # eigenval_sort[k], k = 0 is the first eigenvalue
    else: # k > 0
        return eigenval_sort[k-1] / eigenval_sort[k] # if k = 1: eigenval_sort[k-1] is the first eigenvalue, eigenval_sort[k] is the second

# due to python indexing:
# first eigenvalue of eigenval_sort is indexed by k-1 as it is on the 0th position


def ER_fun(X, eigen_val_sort, k_max):

    N,T = X.shape
    m = np.minimum(N,T)

    eigen_val_sort = (1 / (N * T)) * eigen_val_sort

    output_vec = np.zeros((k_max + 1, 1))

    for k in range(0, k_max + 1):
        output_vec[k] = ER_k(eigen_val_sort, k, m)

    return np.argmax(output_vec), output_vec

# also range has to start with 0!!!!
# here we return np.argmax(output_vec) as we include 0th eigenvalue ****


# GR
# -----------------------------------------------------------------------




def GR_k(eigen_val_sort,k,m):
    numerator = np.log(V_k_fun(eigen_val_sort,k-1,m) / V_k_fun(eigen_val_sort,k,m))
    denominator = np.log(V_k_fun(eigen_val_sort,k,m) / V_k_fun(eigen_val_sort,k+1,m))

    return numerator/denominator


def GR_fun(X, eigen_vals_sort, k_max):
    N,T = X.shape
    m = np.minimum(N,T)

    eigen_vals_sort = (1 / (N * T)) * eigen_vals_sort

    output_vec = np.zeros((k_max + 1, 1))
    for i in range(0, k_max + 1):
        output_vec[i] = GR_k(eigen_vals_sort, i, m)

    return np.argmax(output_vec), output_vec

# here we return np.argmax(output_vec) as we include 0th eigenvalue ****



