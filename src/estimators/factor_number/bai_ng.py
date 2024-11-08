
"""
python script for Bai, J., & Ng, S. (2002) estimators
using specification from Onatski, A. (2010)
"""

"""
prior to the computation of eigenvalues for these estimators:
EACH SERIES OF THE PANEL DATA HAS TO BE TIME DE-MEANED, AND 
STANDARDIZED TO HAVE UNIT VARIANCE
- otherwise the estimators are under-performing
"""


"""
I include code for IC2, IC3 and PC2, PC3 estimators 
even though I do not use them in the analysis
- included at the bottom of the file
"""

# ************************************************************

# importing packages
# ------------------
import numpy as np



# ************************************************************

# common functions used by all estimators
# --------------------------------
# --------------------------------

"""
these use the specification from Onatski, A. (2010)
- V_k_fun and sigma2_fun are 
sums of eigenvalues of covariance or gram matrix of X
-> this significantly speeds up the computation time
"""

def V_k_fun(eigen_vals_sort, k):
    return np.sum(eigen_vals_sort[k:])


def sigma2_fun(eigen_vals_sort, k_max):
    return np.sum(eigen_vals_sort[k_max:])


# Estimators used in the analysis
# ************************************************************
# ************************************************************

"""
eigen_val_sort are sorted eigenvalues of X@X.T or X.T@X matrix
from eigenvals_fun

I divide them by (N*T) here, not in eigenvals_fun
also the division is not done when passing into the function,
just raw sorted eigenvalues are passed in
"""

# IC1
# --------------------------------

def IC1_k_fun(X, eigen_vals_sort, k):
    N,T = X.shape
    V_k = (1/(N*T)) * V_k_fun(eigen_vals_sort, k)

    IC1 = np.log(V_k) + \
          k * ( (N+T) / (N*T) ) * np.log((N*T)/(N+T))

    return IC1


def IC1_fun(X, eigen_vals_sort, k_max):
    output_vec = np.zeros((k_max, 1))
    for i in range(0, k_max):
        output_vec[i] = IC1_k_fun(X, eigen_vals_sort, i + 1)

    return np.argmin(output_vec) + 1, output_vec



# PC1
# --------------------------------

def PC1_k_fun(X, eigen_vals_sort, k, k_max):
    N,T = X.shape
    V_k = (1/(N*T)) * V_k_fun(eigen_vals_sort, k)
    sigma2 = (1/(N*T)) * sigma2_fun(eigen_vals_sort, k_max)

    PC1 = V_k +\
          k*sigma2*((N+T)/(N*T))*np.log((N*T)/(N+T))

    return PC1



def PC1_fun(X, eigen_vals_sort, k_max):
    output_vec = np.zeros((k_max,1))
    for i in range(0,k_max):
        output_vec[i] = PC1_k_fun(X, eigen_vals_sort, i + 1, k_max)

    return np.argmin(output_vec) + 1, output_vec



# BIC3
# --------------------------------

def BIC3_k_fun(X, eigen_vals_sort, k, k_max):
    N,T = X.shape
    V_k = (1/(N*T)) * V_k_fun(eigen_vals_sort, k)
    sigma2 = (1/(N*T)) * sigma2_fun(eigen_vals_sort, k_max)

    BIC3 = V_k +\
          k*sigma2*(((N+T-k)*np.log(N*T))/(N*T))

    return BIC3



def BIC3_fun(X, eigen_vals_sort, k_max):
    output_vec = np.zeros((k_max,1))
    for i in range(0,k_max):
        output_vec[i] = BIC3_k_fun(X, eigen_vals_sort, i + 1, k_max)

    return np.argmin(output_vec) +1 , output_vec



# Further Estimators - now no longer used
# ************************************************************
# ************************************************************

# IC2
# --------------------------------

def IC2_k_fun(X,eigen_val_sort,k):
    N,T = X.shape
    C2_NT = np.minimum(N,T)
    V_k = (1/(N*T)) * V_k_fun(eigen_val_sort,k)

    IC2 = np.log(V_k) + \
          k*((N+T)/(N*T))*np.log(C2_NT)

    return IC2


def IC2_fun(X,eigen_val_sort,kmax):
    output_vec = np.zeros((kmax,1))
    for i in range(0,kmax):
        output_vec[i] = IC2_k_fun(X,eigen_val_sort,i+1)

    return output_vec, np.argmin(output_vec) +1


# IC3
# --------------------------------

def IC3_k_fun(X,eigen_val_sort,k):
    N,T = X.shape
    C2_NT = np.minimum(N,T)
    V_k = (1/(N*T)) * V_k_fun(eigen_val_sort,k)

    IC3 = np.log(V_k) +\
          k*(np.log(C2_NT)/C2_NT)

    return IC3



def IC3_fun(X,eigen_val_sort,kmax):
    output_vec = np.zeros((kmax,1))
    for i in range(0,kmax):
        output_vec[i] = IC3_k_fun(X,eigen_val_sort,i+1)

    return output_vec, np.argmin(output_vec) +1



# PC2
# --------------------------------

def PC2_k_fun(X,eigen_val_sort,k,k_max):
    N,T = X.shape
    C2_NT = np.minimum(N,T)
    V_k = (1/(N*T)) * V_k_fun(eigen_val_sort,k)
    sigma2 = (1/(N*T)) * sigma2_fun(eigen_val_sort,k_max)

    PC2 = V_k +\
          k*sigma2*((N+T)/(N*T))*np.log(C2_NT)

    return PC2


def PC2_fun(X,eigen_val_sort,k_max):
    output_vec = np.zeros((k_max,1))
    for i in range(0,k_max):
        output_vec[i] = PC2_k_fun(X,eigen_val_sort,i+1,k_max)

    return output_vec, np.argmin(output_vec) +1


# PC3
# --------------------------------

def PC3_k_fun(X,eigen_val_sort,k,k_max):
    N,T = X.shape
    C2_NT = np.minimum(N,T)
    V_k = (1/(N*T)) * V_k_fun(eigen_val_sort,k)
    sigma2 = (1/(N*T)) * sigma2_fun(eigen_val_sort,k_max)

    PC3 = V_k +\
          k*sigma2*(np.log(C2_NT)/C2_NT)

    return PC3


def PC3_fun(X,eigen_val_sort,k_max):
    output_vec = np.zeros((k_max,1))
    for i in range(0,k_max):
        output_vec[i] = PC3_k_fun(X,eigen_val_sort,i+1,k_max)

    return output_vec, np.argmin(output_vec) +1