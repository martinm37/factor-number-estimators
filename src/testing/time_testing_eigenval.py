


import sys
import timeit
import numpy as np



# eigenvalue_based_k_estimator_dictionary = {
#     "ER": ER_fun,
#     "GR": GR_fun,
#     "ED": ED_fun,
#     "IC1": IC1_fun,
#     "BIC3": BIC3_fun
# }

# mycode = """
#
# # Computing eigenvalues
# # ----------------------
# eigen_vals_tduv = eigenvals_fun(X_tduv)
# index_tduv = np.argsort(eigen_vals_tduv)[::-1]
# eigen_val_tduv_sort = eigen_vals_tduv[index_tduv]
#
# GR_fun(X_tduv, eigen_val_tduv_sort, k_max = 8)[0]"""

mycode = """TKCV_fun(X_tduv, fold_number=5, k_max=8)[0]"""


iter_num = 250


mysetup = f'''

N = 500
T = 500


import numpy as np
import timeit
import cProfile

# auxiliary functions
from src.computation_functions.aux_functions import eigenvals_fun

# data generating process
from src.computation_functions.data_generating_process import dgp_nt_correlations


from src.estimators.factor_number.ahn_horenstein import ER_fun, GR_fun
from src.estimators.factor_number.bai_ng import IC1_fun, BIC3_fun
from src.estimators.factor_number.onatski import ED_fun
from src.estimators.factor_number.wei_chen_numba import TKCV_fun

r = 3
J = 8
burning_period = 100
rng = np.random.default_rng(seed=1327)

Lambda = rng.normal(0, 1, size=(N, r))  # matrix of factor loadings - generated once per setting


# standardizing rows of lambda
Lambda_row_sum = np.sqrt(np.sum(Lambda * Lambda, axis=1).reshape(-1, 1))  # * is hadamard product , -> l2 norm for each row
Lambda = Lambda / Lambda_row_sum  # / is hadamard division? , -> standardizing

X_raw = dgp_nt_correlations(Lambda = Lambda, N = N, T = T,
                        burning_period = burning_period,
                        r = r, SNR = 0.5, rho = 0.5, beta = 0.5, J = J,
                        rng = rng)


X_tduv = X_raw

'''

# code snippet whose execution time is to be measured
# mycode = f"eigenvalue_based_k_estimator_dictionary[{estimator}](X_tduv, eigen_val_tduv_sort, k_max = 8)[0]"

time = timeit.timeit(stmt=mycode,
                     setup=mysetup,
                     number=iter_num)

# print(f"N: {N}, T: {T}")
print(np.round((time / iter_num),decimals=3))


