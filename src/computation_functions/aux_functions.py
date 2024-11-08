
"""
auxiliary functions used in the code files for estimators
"""

# importing packages
# ------------------
import numpy as np
import numba

# functions:
# ------------------------

@numba.jit(nopython = True)
def eigenvals_fun(X):

    N,T = X.shape # X is (NxT)    
    
    if N <= T:
        return np.linalg.eigvalsh(X @ X.T) # X @ X.T is (NxN)
    else: #(T < N)
        return np.linalg.eigvalsh(X.T @ X) # X.T @ X is (TxT)
    


"""
notes for eigenvals_fun(X):
--------------------------

for a given data matrix X (NxT),
this function returns eigenvalues 
of its covariance matrix X @ X.T (NxN)

as the first K = min{N,T} eigenvalues of
covariance matrix X @ X.T (NxN) and 
gram matrix X.T @ X (TxT) are the same

this function computes eigenvalues of the
matrix with smaller dimension    

---

also, eigenvalues of a matrix cA (matrix A multiplied by a scalar c)
are the same as c times eigenvalues of a matrix A

for this reason, I will be only computing eigenvalues of X @ X.T
(or X.T @ X) and then dividing them by NT or T or other scalars 
for specific estimators   
"""

