

import numpy as np
from src.computation_functions.data_generating_process import dgp_basic
import src.computation_functions.aux_functions as aux
import src.estimators.factor_number.ahn_horenstein as ah
from src.estimators.covariance_matrix.old.poet import poet_fun

# setting seed for the DGP
# - always to be done only once, and in the app file, not src file!!!
#np.random.seed(seed = 1327)

simulation_seed = 1372

# simulation parameters
# ----------------------

N = 150
T = 255

SNR = 1
rho = 0.5
beta = 0.5
J = 8

k_max = 8

r = 5


# DGP
# ----------------------

rng = np.random.default_rng(seed = simulation_seed)

Lambda = rng.normal(0,1,size = (N, r)) # matrix of factor loadings - generated once per setting

#X = DGP_V3_NT(Lambda = Lambda,N = N,T=T,r=r,theta = theta,rho = rho, beta = beta,J = J)
X = dgp_basic(Lambda = Lambda, N = N, T=T, r=r, SNR = SNR, rng =  rng)


# computing eigenvalues of cov mat
# --------------------------------

eigen_vals = aux.eigenvals_fun(X)
index = np.argsort(eigen_vals)[::-1]
eigen_val_sort = eigen_vals[index]


# estimating factor number
# ------------------------

N, T = X.shape

K = ah.GR_fun(X, (1 / (N * T)) * eigen_val_sort, k_max)[1]


# estimating population covariance matrix
# ---------------------------------------

covariance_matrix = X @ X.T

C = 0.5

POET_K = poet_fun(covariance_matrix,N,T,K,C)


# estimating portfolio weights
# ---------------------------------------

N = covariance_matrix.shape[0]

ones_vector = np.ones([N,1])

divisor = ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector

weights_vector = ( np.linalg.inv(POET_K) @ ones_vector ) / ( ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector )

#portfolio_weights = np.linalg.inv(POET_K) @ ones_vector / divisor

print("hello there xdd")
