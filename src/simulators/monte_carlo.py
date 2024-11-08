
"""
code for the monte carlo simulations
"""

# import
# ******

# libraries
# ----------------
import os
import time
import numpy as np


# my own code
# -----------

# auxiliary functions
from src.computation_functions.aux_functions import eigenvals_fun

# data generating process
from src.computation_functions.data_generating_process import dgp_nt_correlations

# estimators code
from src.estimators.factor_number.ahn_horenstein import ER_fun, GR_fun
from src.estimators.factor_number.bai_ng import IC1_fun, PC1_fun, BIC3_fun
from src.estimators.factor_number.onatski import ED_fun
from src.estimators.factor_number.wei_chen_numba import TKCV_fun
from src.utils.computation_utils import time_demeaning_unit_variance, double_demeaning

# utils
from src.utils.paths import get_results_path

# code for different Monte Carlo simulation designs
# -------------------------------------------------


"""
general inputs and outputs to the funcitons:
********************************************

Monte Carlo function for * panel data
-----------------------------------------------------------------------------
inputs:
------
dims_tbl = table which has tuples of dimensions for panel data tests
iter_num = number of Monte Carlo iterations
r = true number of factors
theta = SNR
rho = strength of temporal correlations; 0 <= rho < 1 (assuming unit root)
beta = strength of correlation to other entries; 0 <= beta < 1
J = amount of entries the given entry is correlated to => hardcoded in the function
k_max = maximum assumed number of factors
outputs:
-------
output_cube = 3D array: (dimensional tuples) x (estimators) x (iterations)

"""


# setting estimator number
estimator_num = 7 # hardcoding, depends on the structure of the function


def monte_carlo_nt(dims_tbl, iter_num, burning_period, r, SNR, rho, beta, J_type, k_max, simulation_seed):

    # preallocating 3D array for results
    dims_length = dims_tbl.shape[0]
    output_cube = np.zeros((dims_length, estimator_num, iter_num), dtype=np.intc)

    for x in range(0, dims_length):

        """
        looping over panel data dimension tuples
        for each tuple Lambda matrix is geenrated only once
        """

        t0 = time.time()  # start time for a tuple

        N = dims_tbl[x, 0]

        # initializing random number generator class
        rng = np.random.default_rng(seed=simulation_seed)

        Lambda = rng.normal(0, 1, size=(N, r))  # matrix of factor loadings - generated once per setting


        # standardizing rows of lambda
        Lambda_row_sum = np.sqrt(np.sum(Lambda * Lambda, axis=1).reshape(-1, 1))  # * is hadamard product , -> l2 norm for each row
        Lambda = Lambda / Lambda_row_sum  # / is hadamard division? , -> standardizing

        # amount of cross sectional correlations:
        # ---------------------------------------
        J = 0

        if J_type == "static":
            J = 8
        elif J_type == "dynamic":
            J = int(np.maximum(np.floor(dims_tbl[x, 0] / 20), 10))

        assert J != 0, "J_type must be either 'static' or 'dynamic'"

        print(f"N: {dims_tbl[x, 0]}, J: {J}")

        with open(os.path.join(get_results_path(), "mc_run_log.txt"), "a") as text_file:
            print(f"N: {dims_tbl[x, 0]}, J: {J}", file=text_file)


        iter_counter = 0

        for e in range(0, iter_num):

            iter_counter += 1

            if iter_counter == 100:
                print(f"tuple: ({dims_tbl[x, 0]},{dims_tbl[x, 1]}) interation: {e + 1}")
                iter_counter = 0

            """
            Monte Carlo simulation for the given dimension tuple
            """

            # DGP
            # -------
            X_raw = dgp_nt_correlations(Lambda = Lambda, N = dims_tbl[x, 0], T = dims_tbl[x, 1],
                                    burning_period = burning_period,
                                    r = r, SNR = SNR, rho = rho, beta = beta, J = J,
                                    rng = rng)


            """
            specific transformations of panel data set X
            based on needs of estimators:            
            """

            # time demeaning and standardizing to unit variance
            # **********************************************************************************************************
            # **********************************************************************************************************

            X_tduv = time_demeaning_unit_variance(X_raw)

            # Computing eigenvalues
            # ----------------------
            eigen_vals_tduv = eigenvals_fun(X_tduv)
            index_tduv = np.argsort(eigen_vals_tduv)[::-1]
            eigen_val_tduv_sort = eigen_vals_tduv[index_tduv]

            # computing k_hat
            # ----------------
            # dividing ALWAYS done inside the function
            output_cube[x, 0, e] = IC1_fun(X_tduv, eigen_val_tduv_sort, k_max = k_max)[0]
            output_cube[x, 1, e] = PC1_fun(X_tduv, eigen_val_tduv_sort, k_max = k_max)[0]
            output_cube[x, 2, e] = BIC3_fun(X_tduv, eigen_val_tduv_sort, k_max = k_max)[0]

            output_cube[x, 6, e] = TKCV_fun(X_tduv, fold_number=5, k_max=k_max)[0]


            # no transformation
            # **********************************************************************************************************
            # **********************************************************************************************************

            # # Computing eigenvalues
            # # ----------------------
            # eigen_vals_raw = eigenvals_fun(X_raw)
            # index_raw = np.argsort(eigen_vals_raw)[::-1]
            # eigen_vals_raw_sort = eigen_vals_raw[index_raw]
            #
            # # computing k_hat
            # # ----------------
            # # dividing ALWAYS done inside the function
            # output_cube[x, 3, e] = ED_fun(X_raw, eigen_vals_raw_sort, k_max = k_max)[0]

            # double demeaning
            # **********************************************************************************************************
            # **********************************************************************************************************

            X_dd = double_demeaning(X_raw)

            # Computing eigenvalues
            # ----------------------
            eigen_vals_dd = eigenvals_fun(X_dd)
            index_dd = np.argsort(eigen_vals_dd)[::-1]
            eigen_vals_dd_sort = eigen_vals_dd[index_dd]

            # computing k_hat
            # ----------------
            # dividing ALWAYS done inside the function
            output_cube[x, 3, e] = ED_fun(X_dd, eigen_vals_dd_sort, k_max=k_max)[0]
            output_cube[x, 4, e] = ER_fun(X_dd, eigen_vals_dd_sort, k_max=k_max)[0]
            output_cube[x, 5, e] = GR_fun(X_dd, eigen_vals_dd_sort, k_max = k_max)[0]




        t1 = time.time()  # end time for a tuple

        print(f"tuple: ({dims_tbl[x, 0]},{dims_tbl[x, 1]}) - execution time:"
              f" {round(t1 - t0, 2)} sec,"
              f" {round((t1 - t0) / 60, 2)} min,"
              f" {round((t1 - t0) / (60 * 60), 2)} hour")

        with open(os.path.join(get_results_path(), "mc_run_log.txt"), "a") as text_file:

            print(f"tuple: ({dims_tbl[x, 0]},{dims_tbl[x, 1]}) - execution time:"
                  f" {round(t1 - t0, 2)} sec,"
                  f" {round((t1 - t0) / 60, 2)} min,"
                  f" {round((t1 - t0) / (60 * 60), 2)} hour", file=text_file)

    return output_cube




def monte_carlo_snr(dims_tbl, iter_num, burning_period,N , T, r, rho, beta, J_type, k_max, simulation_seed):

    # preallocating 3D array for results
    dims_length = dims_tbl.shape[0]
    output_cube = np.zeros((dims_length, estimator_num, iter_num), dtype=np.intc)

    for x in range(0, dims_length):

        """
        looping over panel data dimension tuples
        for each tuple Lambda matrix is geenrated only once
        """

        t0 = time.time()  # start time for a tuple

        N = N

        # initializing random number generator class
        rng = np.random.default_rng(seed=simulation_seed)

        Lambda = rng.normal(0, 1, size=(N, r))  # matrix of factor loadings - generated once per setting


        # standardizing rows of lambda
        Lambda_row_sum = np.sqrt(np.sum(Lambda * Lambda, axis=1).reshape(-1, 1))  # * is hadamard product , -> l2 norm for each row
        Lambda = Lambda / Lambda_row_sum  # / is hadamard division? , -> standardizing


        # amount of cross sectional correlations:
        # ---------------------------------------
        J = 0

        if J_type == "static":
            J = 8
        elif J_type == "dynamic":
            J = int(np.maximum(np.floor(dims_tbl[x, 0] / 20), 10))

        assert J != 0, "J_type must be either 'static' or 'dynamic'"

        print(f"N: {N}, J: {J}")

        with open(os.path.join(get_results_path(), "mc_run_log.txt"), "a") as text_file:
            print(f"N: {N}, J: {J}", file=text_file)

        iter_counter = 0

        for e in range(0, iter_num):

            iter_counter += 1

            if iter_counter == 100:
                print(f"SNR: ({dims_tbl[x, 0]}) interation: {e + 1}")
                iter_counter = 0

            """
            Monte Carlo simulation for the given dimension tuple
            """

            # DGP
            # -------
            X_raw = dgp_nt_correlations(Lambda = Lambda, N = N, T = T,
                                    burning_period = burning_period,
                                    r = r, SNR = dims_tbl[x,0], rho = rho, beta = beta, J = J,
                                    rng = rng)

            """
            specific transformations of panel data set X
            based on needs of estimators:            
            """

            # time demeaning and standardizing to unit variance
            # **********************************************************************************************************
            # **********************************************************************************************************

            X_time_dem_unit_var = time_demeaning_unit_variance(X_raw)

            # Computing eigenvalues
            # ----------------------
            eigen_vals_tduv = eigenvals_fun(X_time_dem_unit_var)
            index_tduv = np.argsort(eigen_vals_tduv)[::-1]
            eigen_val_tduv_sort = eigen_vals_tduv[index_tduv]

            # computing k_hat
            # ----------------
            # dividing ALWAYS done inside the function
            output_cube[x, 0, e] = IC1_fun(X_time_dem_unit_var, eigen_val_tduv_sort, k_max=k_max)[0]
            output_cube[x, 1, e] = PC1_fun(X_time_dem_unit_var, eigen_val_tduv_sort, k_max=k_max)[0]
            output_cube[x, 2, e] = BIC3_fun(X_time_dem_unit_var, eigen_val_tduv_sort, k_max=k_max)[0]

            output_cube[x, 6, e] = TKCV_fun(X_time_dem_unit_var, fold_number=5, k_max=k_max)[0]

            # no transformation
            # **********************************************************************************************************
            # **********************************************************************************************************

            # # Computing eigenvalues
            # # ----------------------
            # eigen_vals_raw = eigenvals_fun(X_raw)
            # index_raw = np.argsort(eigen_vals_raw)[::-1]
            # eigen_vals_raw_sort = eigen_vals_raw[index_raw]
            #
            # # computing k_hat
            # # ----------------
            # # dividing ALWAYS done inside the function
            # output_cube[x, 3, e] = ED_fun(X_raw, eigen_vals_raw_sort, k_max=k_max)[0]

            # double demeaning
            # **********************************************************************************************************
            # **********************************************************************************************************

            X_dbl_dem = double_demeaning(X_raw)

            # Computing eigenvalues
            # ----------------------
            eigen_vals_dd = eigenvals_fun(X_dbl_dem)
            index_dd = np.argsort(eigen_vals_dd)[::-1]
            eigen_vals_dd_sort = eigen_vals_dd[index_dd]

            # computing k_hat
            # ----------------
            # dividing ALWAYS done inside the function
            output_cube[x, 3, e] = ED_fun(X_dbl_dem, eigen_vals_dd_sort, k_max=k_max)[0]
            output_cube[x, 4, e] = ER_fun(X_dbl_dem, eigen_vals_dd_sort, k_max=k_max)[0]
            output_cube[x, 5, e] = GR_fun(X_dbl_dem, eigen_vals_dd_sort, k_max=k_max)[0]




        t1 = time.time()  # end time for a tuple

        print(f"SNR: ({dims_tbl[x, 0]}) - execution time:"
              f" {round(t1 - t0, 2)} sec,"
              f" {round((t1 - t0) / 60, 2)} min,"
              f" {round((t1 - t0) / (60 * 60), 2)} hour")

        with open(os.path.join(get_results_path(), "mc_run_log.txt"), "a") as text_file:

            print(f"SNR: ({dims_tbl[x, 0]}) - execution time:"
                  f" {round(t1 - t0, 2)} sec,"
                  f" {round((t1 - t0) / 60, 2)} min,"
                  f" {round((t1 - t0) / (60 * 60), 2)} hour", file=text_file)

    return output_cube



def monte_carlo_rho(dims_tbl, iter_num, burning_period, N, T, r, SNR, beta, J_type, k_max, simulation_seed):

    # preallocating 3D array for results
    dims_length = dims_tbl.shape[0]
    output_cube = np.zeros((dims_length, estimator_num, iter_num), dtype=np.intc)

    for x in range(0, dims_length):

        """
        looping over panel data dimension tuples
        for each tuple Lambda matrix is generated only once
        """

        t0 = time.time()  # start time for a tuple

        N = N

        # initializing random number generator class
        rng = np.random.default_rng(seed=simulation_seed)

        Lambda = rng.normal(0, 1, size=(N, r))  # matrix of factor loadings - generated once per setting


        # standardizing rows of lambda
        Lambda_row_sum = np.sqrt(np.sum(Lambda * Lambda, axis=1).reshape(-1, 1))  # * is hadamard product , -> l2 norm for each row
        Lambda = Lambda / Lambda_row_sum  # / is hadamard division? , -> standardizing

        # amount of cross sectional correlations:
        # ---------------------------------------
        J = 0

        if J_type == "static":
            J = 8
        elif J_type == "dynamic":
            J = int(np.maximum(np.floor(dims_tbl[x, 0] / 20), 10))

        assert J != 0, "J_type must be either 'static' or 'dynamic'"


        # printing out the current J
        print(f"N: {N}, J: {J}")

        with open(os.path.join(get_results_path(), "mc_run_log.txt"), "a") as text_file:
            print(f"N: {N}, J: {J}", file=text_file)

        iter_counter = 0

        for e in range(0, iter_num):

            iter_counter += 1

            if iter_counter == 100:
                print(f"rho: ({dims_tbl[x, 0]}) interation: {e + 1}")
                iter_counter = 0

            """
            Monte Carlo simulation for the given dimension tuple
            """

            # DGP
            # -------
            X_raw = dgp_nt_correlations(Lambda = Lambda, N = N, T = T,
                                    burning_period = burning_period,
                                    r = r, SNR = SNR, rho = dims_tbl[x,0], beta = beta, J = J,
                                    rng = rng)

            """
            specific transformations of panel data set X
            based on needs of estimators:            
            """

            # time demeaning and standardizing to unit variance
            # - for Bai and Ng and Wei and Chen estimators
            # **********************************************************************************************************
            # **********************************************************************************************************

            X_time_dem_unit_var = time_demeaning_unit_variance(X_raw)

            # Computing eigenvalues
            # ----------------------
            eigen_vals_tduv = eigenvals_fun(X_time_dem_unit_var)
            index_tduv = np.argsort(eigen_vals_tduv)[::-1]
            eigen_val_tduv_sort = eigen_vals_tduv[index_tduv]

            # computing k_hat
            # ----------------
            # dividing ALWAYS done inside the function
            output_cube[x, 0, e] = IC1_fun(X_time_dem_unit_var, eigen_val_tduv_sort, k_max=k_max)[0]
            output_cube[x, 1, e] = PC1_fun(X_time_dem_unit_var, eigen_val_tduv_sort, k_max=k_max)[0]
            output_cube[x, 2, e] = BIC3_fun(X_time_dem_unit_var, eigen_val_tduv_sort, k_max=k_max)[0]

            output_cube[x, 6, e] = TKCV_fun(X_time_dem_unit_var, fold_number=5, k_max=k_max)[0]

            # no transformation
            # - for Onatski estimator
            # **********************************************************************************************************
            # **********************************************************************************************************

            # # Computing eigenvalues
            # # ----------------------
            # eigen_vals_raw = eigenvals_fun(X_raw)
            # index_raw = np.argsort(eigen_vals_raw)[::-1]
            # eigen_vals_raw_sort = eigen_vals_raw[index_raw]
            #
            # # computing k_hat
            # # ----------------
            # # dividing ALWAYS done inside the function
            # output_cube[x, 3, e] = ED_fun(X_raw, eigen_vals_raw_sort, k_max=k_max)[0]

            # double demeaning
            # - for Ahn and Horenstein estimators
            # **********************************************************************************************************
            # **********************************************************************************************************

            X_dbl_dem = double_demeaning(X_raw)

            # Computing eigenvalues
            # ----------------------
            eigen_vals_dd = eigenvals_fun(X_dbl_dem)
            index_dd = np.argsort(eigen_vals_dd)[::-1]
            eigen_vals_dd_sort = eigen_vals_dd[index_dd]

            # computing k_hat
            # ----------------
            # dividing ALWAYS done inside the function
            output_cube[x, 3, e] = ED_fun(X_dbl_dem, eigen_vals_dd_sort, k_max=k_max)[0]
            output_cube[x, 4, e] = ER_fun(X_dbl_dem, eigen_vals_dd_sort, k_max=k_max)[0]
            output_cube[x, 5, e] = GR_fun(X_dbl_dem, eigen_vals_dd_sort, k_max=k_max)[0]


        t1 = time.time()  # end time for a tuple

        print(f"rho: ({dims_tbl[x, 0]}) - execution time:"
              f" {round(t1 - t0, 2)} sec,"
              f" {round((t1 - t0) / 60, 2)} min,"
              f" {round((t1 - t0) / (60 * 60), 2)} hour")

        with open(os.path.join(get_results_path(), "mc_run_log.txt"), "a") as text_file:

            print(f"rho: ({dims_tbl[x, 0]}) - execution time:"
                  f" {round(t1 - t0, 2)} sec,"
                  f" {round((t1 - t0) / 60, 2)} min,"
                  f" {round((t1 - t0) / (60 * 60), 2)} hour", file=text_file)

    return output_cube





def monte_carlo_beta(dims_tbl, iter_num, burning_period,N , T, r, SNR, rho, J_type, k_max, simulation_seed):

    # preallocating 3D array for results
    dims_length = dims_tbl.shape[0]
    output_cube = np.zeros((dims_length, estimator_num, iter_num), dtype=np.intc)

    for x in range(0, dims_length):

        """
        looping over panel data dimension tuples
        for each tuple Lambda matrix is geenrated only once
        """

        t0 = time.time()  # start time for a tuple

        N = N

        # initializing random number generator class
        rng = np.random.default_rng(seed=simulation_seed)

        Lambda = rng.normal(0, 1, size=(N, r))  # matrix of factor loadings - generated once per setting


        # standardizing rows of lambda
        Lambda_row_sum = np.sqrt(np.sum(Lambda * Lambda, axis=1).reshape(-1, 1))  # * is hadamard product , -> l2 norm for each row
        Lambda = Lambda / Lambda_row_sum  # / is hadamard division? , -> standardizing

        # amount of cross sectional correlations:
        # ---------------------------------------
        J = 0

        if J_type == "static":
            J = 8
        elif J_type == "dynamic":
            J = int(np.maximum(np.floor(dims_tbl[x, 0] / 20), 10))

        assert J != 0, "J_type must be either 'static' or 'dynamic'"

        print(f"N: {N}, J: {J}")

        with open(os.path.join(get_results_path(), "mc_run_log.txt"), "a") as text_file:
            print(f"N: {N}, J: {J}", file=text_file)

        iter_counter = 0

        for e in range(0, iter_num):

            iter_counter += 1

            if iter_counter == 100:
                print(f"beta: ({dims_tbl[x, 0]}) interation: {e + 1}")
                iter_counter = 0

            """
            Monte Carlo simulation for the given dimension tuple
            """

            # DGP
            # -------
            X_raw = dgp_nt_correlations(Lambda = Lambda, N = N, T = T,
                                    burning_period = burning_period,
                                    r = r, SNR = SNR, rho = rho, beta = dims_tbl[x,0], J = J,
                                    rng = rng)

            """
            specific transformations of panel data set X
            based on needs of estimators:            
            """

            # time demeaning and standardizing to unit variance
            # - for Bai and Ng and Wei and Chen estimators
            # **********************************************************************************************************
            # **********************************************************************************************************

            X_time_dem_unit_var = time_demeaning_unit_variance(X_raw)

            # Computing eigenvalues
            # ----------------------
            eigen_vals_tduv = eigenvals_fun(X_time_dem_unit_var)
            index_tduv = np.argsort(eigen_vals_tduv)[::-1]
            eigen_val_tduv_sort = eigen_vals_tduv[index_tduv]

            # computing k_hat
            # ----------------
            # dividing ALWAYS done inside the function
            output_cube[x, 0, e] = IC1_fun(X_time_dem_unit_var, eigen_val_tduv_sort, k_max=k_max)[0]
            output_cube[x, 1, e] = PC1_fun(X_time_dem_unit_var, eigen_val_tduv_sort, k_max=k_max)[0]
            output_cube[x, 2, e] = BIC3_fun(X_time_dem_unit_var, eigen_val_tduv_sort, k_max=k_max)[0]

            output_cube[x, 6, e] = TKCV_fun(X_time_dem_unit_var, fold_number=5, k_max=k_max)[0]

            # no transformation
            # - for Onatski estimator
            # **********************************************************************************************************
            # **********************************************************************************************************

            # # Computing eigenvalues
            # # ----------------------
            # eigen_vals_raw = eigenvals_fun(X_raw)
            # index_raw = np.argsort(eigen_vals_raw)[::-1]
            # eigen_vals_raw_sort = eigen_vals_raw[index_raw]
            #
            # # computing k_hat
            # # ----------------
            # # dividing ALWAYS done inside the function
            # output_cube[x, 3, e] = ED_fun(X_raw, eigen_vals_raw_sort, k_max=k_max)[0]

            # double demeaning
            # - for Ahn and Horenstein estimators
            # **********************************************************************************************************
            # **********************************************************************************************************

            X_dbl_dem = double_demeaning(X_raw)

            # Computing eigenvalues
            # ----------------------
            eigen_vals_dd = eigenvals_fun(X_dbl_dem)
            index_dd = np.argsort(eigen_vals_dd)[::-1]
            eigen_vals_dd_sort = eigen_vals_dd[index_dd]

            # computing k_hat
            # ----------------
            # dividing ALWAYS done inside the function
            output_cube[x, 3, e] = ED_fun(X_dbl_dem, eigen_vals_dd_sort, k_max=k_max)[0]
            output_cube[x, 4, e] = ER_fun(X_dbl_dem, eigen_vals_dd_sort, k_max=k_max)[0]
            output_cube[x, 5, e] = GR_fun(X_dbl_dem, eigen_vals_dd_sort, k_max=k_max)[0]


        t1 = time.time()  # end time for a tuple

        print(f"beta: ({dims_tbl[x, 0]}) - execution time:"
              f" {round(t1 - t0, 2)} sec,"
              f" {round((t1 - t0) / 60, 2)} min,"
              f" {round((t1 - t0) / (60 * 60), 2)} hour")

        with open(os.path.join(get_results_path(), "mc_run_log.txt"), "a") as text_file:

            print(f"beta: ({dims_tbl[x, 0]}) - execution time:"
                  f" {round(t1 - t0, 2)} sec,"
                  f" {round((t1 - t0) / 60, 2)} min,"
                  f" {round((t1 - t0) / (60 * 60), 2)} hour", file=text_file)

    return output_cube





