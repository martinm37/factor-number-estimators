
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
import src.computation_functions.aux_functions as aux


# data generating process
import src.computation_functions.data_generating_process as dgp


# estimators code
import src.estimators.factor_number.bai_ng as bn
import src.estimators.factor_number.ahn_horenstein as ah
import src.estimators.factor_number.onatski as on
import src.estimators.factor_number.wei_chen_numba as wc

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

# initializing random number generator class
rng = np.random.default_rng()

# setting estimator number
estimator_num = 7 # hardcoding, depends on the structure of the function


def monte_carlo_iid(dims_tbl,iter_num,r,theta,k_max):

    # preallocating 3D array for results

    dims_length = dims_tbl.shape[0]
    output_cube = np.zeros((dims_length,estimator_num,iter_num),dtype=np.intc)
    
    for x in range(0,dims_length):

        """
        looping over panel data dimension tuples
        for each tuple Lambda matrix is geenrated only once
        """

        t0 = time.time() # start time for a tuple
        
        N = dims_tbl[x,0]
        Lambda = rng.normal(0, 1, size=(N, r))  # matrix of factor loadings - generated once per setting

        iter_counter = 0

        for e in range(0,iter_num):

            iter_counter += 1

            if iter_counter == 100:
                print(f"tuple: ({dims_tbl[x, 0]},{dims_tbl[x, 1]}) interation: {e+1}")
                iter_counter = 0


            """
            Monte Carlo simulation for the given dimension tuple
            """

            # DGP
            # -------
            X = dgp.dgp_basic(Lambda, dims_tbl[x,0], dims_tbl[x,1], r, theta)

            # Common part - first 3 papers
            # -----------
            eigen_vals = aux.eigenvals_fun(X)
            index = np.argsort(eigen_vals)[::-1]
            eigen_val_sort = eigen_vals[index]

            N, T = X.shape

            # Specific part - first 3 papers
            # ------------
            output_cube[x,0,e] = bn.IC1_fun(X, eigen_val_sort, k_max)[1] # dividing done inside the function
            output_cube[x,1,e] = bn.PC1_fun(X, eigen_val_sort, k_max)[1] # dividing done inside the function
            output_cube[x,2,e] = bn.BIC3_fun(X, eigen_val_sort, k_max)[1] # dividing done inside the function
            output_cube[x,3,e] = ah.ER_fun((1 / (N * T)) * eigen_val_sort, k_max)[1]
            output_cube[x,4,e] = ah.GR_fun(X, (1 / (N * T)) * eigen_val_sort, k_max)[1]
            output_cube[x,5,e] = on.ED_fun((1 / T) * eigen_val_sort, k_max)[0]


            # Specific part
            # ------------
            j = int(N / 5)
            k = int(T / 5)

            output_cube[x,6,e] = wc.TKCV_fun(X, j=j, k=k, k_max=k_max)[1]



        t1 = time.time() # end time for a tuple


        print(f"tuple: ({dims_tbl[x, 0]},{dims_tbl[x, 1]}) - execution time:"
              f" {round(t1-t0,2)} sec,"
              f" {round((t1-t0)/60,2)} min,"
              f" {round((t1-t0)/(60*60),2)} hour")

        with open(os.path.join(get_results_path(),"mc_run_log.txt"),"a") as text_file:
            print(f"tuple: ({dims_tbl[x, 0]},{dims_tbl[x, 1]}) - execution time:"
                  f" {round(t1 - t0, 2)} sec,"
                  f" {round((t1 - t0) / 60, 2)} min,"
                  f" {round((t1 - t0) / (60 * 60), 2)} hour", file = text_file)

    return output_cube





def monte_carlo_Tcorr(dims_tbl,iter_num,burning_period,r,theta,rho,k_max):



    # preallocating 3D array for results

    dims_length = dims_tbl.shape[0]
    output_cube = np.zeros((dims_length,estimator_num,iter_num),dtype=np.intc)
    
    for x in range(0,dims_length):

        """
        looping over panel data dimension tuples
        for each tuple Lambda matrix is geenrated only once
        """

        t0 = time.time()  # start time for a tuple
        
        N = dims_tbl[x,0]
        Lambda = rng.normal(0, 1, size=(N, r))  # matrix of factor loadings - generated once per setting

        iter_counter = 0

        for e in range(0,iter_num):

            iter_counter += 1

            if iter_counter == 100:
                print(f"tuple: ({dims_tbl[x, 0]},{dims_tbl[x, 1]}) interation: {e+1}")
                iter_counter = 0
            
            """
            Monte Carlo simulation for the given dimension tuple
            """

            # DGP
            # -------
            X = dgp.dgp_t_correlations(Lambda = Lambda, N =  dims_tbl[x,0], T = dims_tbl[x,1],
                                       burning_period = burning_period, r = r, theta = theta, rho = rho)

            # Common part - first 3 papers
            # -----------
            eigen_vals = aux.eigenvals_fun(X)
            index = np.argsort(eigen_vals)[::-1]
            eigen_val_sort = eigen_vals[index]

            N, T = X.shape

            # Specific part - first 3 papers
            # ------------
            output_cube[x, 0, e] = bn.IC1_fun(X, eigen_val_sort, k_max)[1]  # dividing done inside the function
            output_cube[x, 1, e] = bn.PC1_fun(X, eigen_val_sort, k_max)[1]  # dividing done inside the function
            output_cube[x, 2, e] = bn.BIC3_fun(X, eigen_val_sort, k_max)[1]  # dividing done inside the function
            output_cube[x, 3, e] = ah.ER_fun((1 / (N * T)) * eigen_val_sort, k_max)[1]
            output_cube[x, 4, e] = ah.GR_fun(X, (1 / (N * T)) * eigen_val_sort, k_max)[1]
            output_cube[x, 5, e] = on.ED_fun((1 / T) * eigen_val_sort, k_max)[0]

            # Specific part
            # ------------
            j = int(N / 5)
            k = int(T / 5)

            output_cube[x, 6, e] = wc.TKCV_fun(X, j=j, k=k, k_max=k_max)[1]

        t1 = time.time() # end time for a tuple

        print(f"tuple: ({dims_tbl[x, 0]},{dims_tbl[x, 1]}) - execution time:"
              f" {round(t1-t0,2)} sec,"
              f" {round((t1-t0)/60,2)} min,"
              f" {round((t1-t0)/(60*60),2)} hour")

        with open(os.path.join(get_results_path(),"mc_run_log.txt"),"a") as text_file:
            print(f"tuple: ({dims_tbl[x, 0]},{dims_tbl[x, 1]}) - execution time:"
                  f" {round(t1 - t0, 2)} sec,"
                  f" {round((t1 - t0) / 60, 2)} min,"
                  f" {round((t1 - t0) / (60 * 60), 2)} hour", file = text_file)


    return output_cube

def monte_carlo_Ncorr(dims_tbl,iter_num,r,theta,beta,k_max):



    # preallocating 3D array for results

    dims_length = dims_tbl.shape[0]
    output_cube = np.zeros((dims_length,estimator_num,iter_num),dtype=np.intc)
    
    for x in range(0,dims_length):

        """
        looping over panel data dimension tuples
        for each tuple Lambda matrix is geenrated only once
        """

        t0 = time.time()  # start time for a tuple
        
        N = dims_tbl[x,0]
        Lambda = rng.normal(0, 1, size=(N, r))  # matrix of factor loadings - generated once per setting

        iter_counter = 0

        for e in range(0,iter_num):

            iter_counter += 1

            if iter_counter == 100:
                print(f"tuple: ({dims_tbl[x, 0]},{dims_tbl[x, 1]}) interation: {e+1}")
                iter_counter = 0
            
            """
            Monte Carlo simulation for the given dimension tuple
            """

            # DGP
            # -------
            J = 8
            #J=int(np.maximum(np.floor(dims_tbl[x,0]/20),10))
            X = dgp.dgp_n_correlations(Lambda = Lambda, N = dims_tbl[x,0], T = dims_tbl[x,1],
                                       r = r, theta = theta, beta = beta, J = J)

            # Common part - first 3 papers
            # -----------
            eigen_vals = aux.eigenvals_fun(X)
            index = np.argsort(eigen_vals)[::-1]
            eigen_val_sort = eigen_vals[index]

            N, T = X.shape

            # Specific part - first 3 papers
            # ------------
            output_cube[x, 0, e] = bn.IC1_fun(X, eigen_val_sort, k_max)[1]  # dividing done inside the function
            output_cube[x, 1, e] = bn.PC1_fun(X, eigen_val_sort, k_max)[1]  # dividing done inside the function
            output_cube[x, 2, e] = bn.BIC3_fun(X, eigen_val_sort, k_max)[1]  # dividing done inside the function
            output_cube[x, 3, e] = ah.ER_fun((1 / (N * T)) * eigen_val_sort, k_max)[1]
            output_cube[x, 4, e] = ah.GR_fun(X, (1 / (N * T)) * eigen_val_sort, k_max)[1]
            output_cube[x, 5, e] = on.ED_fun((1 / T) * eigen_val_sort, k_max)[0]

            # Specific part
            # ------------
            j = int(N / 5)
            k = int(T / 5)

            output_cube[x, 6, e] = wc.TKCV_fun(X, j=j, k=k, k_max=k_max)[1]


        t1 = time.time() # end time for a tuple

        print(f"tuple: ({dims_tbl[x, 0]},{dims_tbl[x, 1]}) - execution time:"
              f" {round(t1-t0,2)} sec,"
              f" {round((t1-t0)/60,2)} min,"
              f" {round((t1-t0)/(60*60),2)} hour")

        with open(os.path.join(get_results_path(),"mc_run_log.txt"),"a") as text_file:
            print(f"tuple: ({dims_tbl[x, 0]},{dims_tbl[x, 1]}) - execution time:"
                  f" {round(t1 - t0, 2)} sec,"
                  f" {round((t1 - t0) / 60, 2)} min,"
                  f" {round((t1 - t0) / (60 * 60), 2)} hour", file = text_file)


    return output_cube



def monte_carlo_NTcorr(dims_tbl,iter_num,burning_period,r,SNR,rho,beta,k_max,simulation_seed):



    # preallocating 3D array for results

    dims_length = dims_tbl.shape[0]
    output_cube = np.zeros((dims_length,estimator_num,iter_num),dtype=np.intc)
    
    for x in range(0,dims_length):

        """
        looping over panel data dimension tuples
        for each tuple Lambda matrix is geenrated only once
        """

        t0 = time.time()  # start time for a tuple

        N = dims_tbl[x,0]

        # initializing random number generator class
        rng = np.random.default_rng(seed = simulation_seed)

        # Lambda = np.ones([N, r]) * (1/np.sqrt(r))

        #Lambda = rng.normal(0, 1, size=(N, r))  # matrix of factor loadings - generated once per setting

        setting = np.array([1.0, 1.1, 1.05]).reshape(1, -1)
        #setting = np.array([1.0,2.0,3.0]).reshape(1,-1)
        Lambda = np.ones([N, r]) * setting

        # standardizing rows of lambda
        Lambda_row_sum = np.sqrt(np.sum(Lambda * Lambda, axis=1).reshape(-1,1)) # * is hadamard product , -> l2 norm for each row
        Lambda = Lambda / Lambda_row_sum # / is hadamard division? , -> standardizing


        iter_counter = 0

        for e in range(0,iter_num):

            iter_counter += 1

            if iter_counter == 100:
                print(f"tuple: ({dims_tbl[x, 0]},{dims_tbl[x, 1]}) interation: {e+1}")
                iter_counter = 0
            
            """
            Monte Carlo simulation for the given dimension tuple
            """

            # DGP
            # -------
            J = 8
            #J=int(np.maximum(np.floor(dims_tbl[x,0]/20),10))

            X = dgp.dgp_nt_correlations(Lambda = Lambda, N = dims_tbl[x,0], T = dims_tbl[x,1], burning_period = burning_period,
                                        r = r, SNR = SNR, rho = rho, beta = beta, J = J, rng = rng)

            # Common part - first 3 papers
            # -----------
            eigen_vals = aux.eigenvals_fun(X)
            index = np.argsort(eigen_vals)[::-1]
            eigen_val_sort = eigen_vals[index]

            N, T = X.shape

            # Specific part - first 3 papers
            # ------------
            output_cube[x, 0, e] = bn.IC1_fun(X, eigen_val_sort, k_max)[1]  # dividing done inside the function
            output_cube[x, 1, e] = bn.PC1_fun(X, eigen_val_sort, k_max)[1]  # dividing done inside the function
            output_cube[x, 2, e] = bn.BIC3_fun(X, eigen_val_sort, k_max)[1]  # dividing done inside the function
            output_cube[x, 3, e] = ah.ER_fun((1 / (N * T)) * eigen_val_sort, k_max)[1]
            output_cube[x, 4, e] = ah.GR_fun(X, (1 / (N * T)) * eigen_val_sort, k_max)[1]
            output_cube[x, 5, e] = on.ED_fun((1 / T) * eigen_val_sort, k_max)[0]

            # Specific part
            # ------------
            j = int(N / 5)
            k = int(T / 5)

            output_cube[x, 6, e] = 11111.1
            #output_cube[x, 6, e] = wc.TKCV_fun(X, j=j, k=k, k_max=k_max)[1]



        t1 = time.time() # end time for a tuple

        print(f"tuple: ({dims_tbl[x, 0]},{dims_tbl[x, 1]}) - execution time:"
              f" {round(t1-t0,2)} sec,"
              f" {round((t1-t0)/60,2)} min,"
              f" {round((t1-t0)/(60*60),2)} hour")



        with open(os.path.join(get_results_path(),"mc_run_log.txt"),"a") as text_file:

            print(f"tuple: ({dims_tbl[x, 0]},{dims_tbl[x, 1]}) - execution time:"
                  f" {round(t1 - t0, 2)} sec,"
                  f" {round((t1 - t0) / 60, 2)} min,"
                  f" {round((t1 - t0) / (60 * 60), 2)} hour", file = text_file)


    return output_cube



"""
code for analysis varying parameters

"""


def monte_carlo_Tcorr_rho_var(rho_tbl,T_tbl,N,iter_num,r,theta,k_max):



    # preallocating 3D array for results

    dims_length = T_tbl.shape[0]
    output_cube = np.zeros((dims_length,estimator_num,iter_num),dtype=np.intc)
    
    for x in range(0,dims_length):

        """
        looping over panel data dimension tuples
        for each tuple Lambda matrix is geenrated only once
        """

        t0 = time.time()  # start time for a tuple

        N = N
        Lambda = rng.normal(0, 1, size=(N, r))  # matrix of factor loadings - generated once per setting

        iter_counter = 0

        for e in range(0,iter_num):

            iter_counter += 1

            if iter_counter == 100:
                print(f"tuple: ({rho_tbl[x,0]},{T_tbl[x,0]}) interation: {e+1}")
                iter_counter = 0
            
            """
            Monte Carlo simulation for the given dimension tuple
            """

            # DGP
            # -------
            X = dgp.dgp_t_correlations(Lambda, N, T_tbl[x,0], r, theta, rho_tbl[x,0])

            N, T = X.shape
            j = int(N / 5)
            k = int(T / 5)

            # Specific part
            # ------------
            output_cube[x,0,e] = wc.TKCV_fun(X, j=j, k=k, k_max=k_max)[1]

        t1 = time.time() # end time for a tuple

        print(f"tuple: ({rho_tbl[x,0]},{T_tbl[x,0]}) - execution time: {round(t1-t0,2)} sec, {round((t1-t0)/60,2)} min, {round((t1-t0)/(60*60),2)} hour")

        with open(os.path.join(get_results_path(),"mc_run_log.txt"),"a") as text_file:
            print(f"tuple: ({rho_tbl[x,0]},{T_tbl[x,0]}) - execution time: {round(t1 - t0, 2)} sec, {round((t1 - t0) / 60, 2)} min, {round((t1 - t0) / (60 * 60), 2)} hour", file = text_file)


    return output_cube





def monte_carlo_Ncorr_beta_var(beta_tbl,T_tbl,N,iter_num,r,theta,k_max):



    # preallocating 3D array for results

    dims_length = T_tbl.shape[0]
    output_cube = np.zeros((dims_length,estimator_num,iter_num),dtype=np.intc)
    
    for x in range(0,dims_length):

        """
        looping over panel data dimension tuples
        for each tuple Lambda matrix is geenrated only once
        """

        t0 = time.time()  # start time for a tuple
        
        N = N
        Lambda = rng.normal(0, 1, size=(N, r))  # matrix of factor loadings - generated once per setting

        iter_counter = 0

        for e in range(0,iter_num):

            iter_counter += 1

            if iter_counter == 100:
                print(f"tuple: ({beta_tbl[x,0]},{T_tbl[x,0]}) interation: {e+1}")
                iter_counter = 0
            
            """
            Monte Carlo simulation for the given dimension tuple
            """

            # DGP
            # -------
            J = 8
            #J=int(np.maximum(np.floor(N/20),10))
            X = dgp.dgp_n_correlations(Lambda, N, T_tbl[x,0], r, theta, beta_tbl[x,0], J)

            N, T = X.shape
            j = int(N / 5)
            k = int(T / 5)

            # Specific part
            # ------------
            output_cube[x,0,e] = wc.TKCV_fun(X, j=j, k=k, k_max=k_max)[1]

        t1 = time.time() # end time for a tuple

        print(f"tuple: ({beta_tbl[x,0]},{T_tbl[x,0]}) - execution time: {round(t1-t0,2)} sec, {round((t1-t0)/60,2)} min, {round((t1-t0)/(60*60),2)} hour")

        with open(os.path.join(get_results_path(),"mc_run_log.txt"),"a") as text_file:
            print(f"tuple: ({beta_tbl[x,0]},{T_tbl[x,0]}) - execution time: {round(t1 - t0, 2)} sec, {round((t1 - t0) / 60, 2)} min, {round((t1 - t0) / (60 * 60), 2)} hour", file = text_file)


    return output_cube
