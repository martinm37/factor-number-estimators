

import numpy as np

from src.computation_functions.eigenvalues_computation import eigenvals_fun
from src.computation_functions.data_generating_process import dgp_nt_correlations,dgp_t_correlations
from src.estimators.factor_number.ahn_horenstein import ER_fun, GR_fun

N = 100
T = 250
burning_period = 100


SNR = 1.2
rho = 0.25
beta = 0.25
J = 8

k_max = 16

r = 8

simulation_seed = 1372

rng = np.random.default_rng(seed=simulation_seed)

Lambda = rng.normal(0,1,size = (N, r)) # matrix of factor loadings - generated once per setting



X = dgp_nt_correlations(Lambda, N, T, burning_period, r, SNR, rho, beta, J, rng)

eigen_vals = eigenvals_fun(X)
index = np.argsort(eigen_vals)[::-1]
eigen_val_sort = eigen_vals[index]


ER_vec,ER_k_hat = ER_fun(X,eigen_val_sort,k_max = k_max)

GR_vec,GR_k_hat = GR_fun(X,eigen_val_sort,k_max = k_max)


print("hello xdd")

