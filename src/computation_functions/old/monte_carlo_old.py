
"""
code for the monte carlo simulations
"""

# import
# ******

# libraries
# ----------------
import numpy as np
from scipy.stats import norm # normal distribution

# my own code
# -----------

# auxiliary functions
import src.computation_functions.aux_functions as aux

# data generating process
import src.computation_functions.data_generating_process as dgp

# estimators code
import src.estimators.factor_number.bai_ng as BN
import src.estimators.factor_number.ahn_horenstein as AH
import src.estimators.factor_number.onatski as On


# code for different Monte Carlo simulation designs
# -------------------------------------------------


def monte_carlo_iid(dims_tbl,iter_num,r,theta,k_max):

    """
    Monte Carlo function for the iid data
    -------------------------------------
    inputs:
    ------
    dims_tbl = table which has tuples of dimensions for panel data tests
    iter_num = number of Monte Carlo iterations
    r = true number of factors
    theta = SNR
    k_max = maximum assumed number of factors
    outputs:
    -------
    output_cube = 3D array: (dimensional tuples) x (estimators) x (iterations)
    """

    # preallocating 3D array for results
    estimator_num = 6 # hardcoding, depends on the structure of the function
    dims_length = dims_tbl.shape[0]
    output_cube = np.zeros((dims_length,estimator_num,iter_num),dtype=np.intc)
    
    for x in range(0,dims_length):

        """
        looping over panel data dimension tuples
        for each tuple Lambda matrix is geenrated only once
        """
        
        N = dims_tbl[x,0]
        Lambda = norm.rvs(0,1,(N,r))  # matrix of factor loadings - generated once per setting

        for e in range(0,iter_num):
            
            """
            Monte Carlo simulation for the given dimension tuple
            """

            # DGP
            # -------
            X = dgp.dgp_basic(Lambda, dims_tbl[x,0], dims_tbl[x,1], r, theta)
            N,T = X.shape

            # Common part
            # -----------
            eigen_vals = aux.eigenvals_fun(X)
            index = np.argsort(eigen_vals)[::-1]
            eigen_val_sort = eigen_vals[index]

            # Specific part
            # ------------
            output_cube[x,0,e] = BN.IC1_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,1,e] = BN.PC1_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,2,e] = BN.BIC3_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,3,e] = AH.ER_fun((1/(N*T))*eigen_val_sort,k_max)[1]
            output_cube[x,4,e] = AH.GR_fun(X,(1/(N*T))*eigen_val_sort,k_max)[1]
            output_cube[x,5,e] = On.ED_fun((1/T)*eigen_val_sort,k_max)[0]

    return output_cube





def monte_carlo_Tcorr(dims_tbl,iter_num,r,theta,rho,k_max):

    """
    Monte Carlo function for temporaly correlated panel data
    -------------------------------------
    inputs:
    ------
    dims_tbl = table which has tuples of dimensions for panel data tests
    iter_num = number of Monte Carlo iterations
    r = true number of factors
    theta = SNR
    rho = strength of temporal correlations; 0 <= rho < 1 (assuming unit root)
    k_max = maximum assumed number of factors
    outputs:
    -------
    output_cube = 3D array: (dimensional tuples) x (estimators) x (iterations)
    """

    # preallocating 3D array for results
    estimator_num = 6 # hardcoding, depends on the structure of the function
    dims_length = dims_tbl.shape[0]
    output_cube = np.zeros((dims_length,estimator_num,iter_num),dtype=np.intc)
    
    for x in range(0,dims_length):

        """
        looping over panel data dimension tuples
        for each tuple Lambda matrix is geenrated only once
        """
        
        N = dims_tbl[x,0]
        Lambda = norm.rvs(0,1,(N,r))  # matrix of factor loadings - generated once per setting

        for e in range(0,iter_num):
            
            """
            Monte Carlo simulation for the given dimension tuple
            """

            # DGP
            # -------
            X = dgp.dgp_t_correlations(Lambda, dims_tbl[x,0], dims_tbl[x,1], r, theta, rho)
            N,T = X.shape

            # Common part
            # -----------
            eigen_vals = aux.eigenvals_fun(X)
            index = np.argsort(eigen_vals)[::-1]
            eigen_val_sort = eigen_vals[index]

            # Specific part
            # ------------
            output_cube[x,0,e] = BN.IC1_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,1,e] = BN.PC1_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,2,e] = BN.BIC3_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,3,e] = AH.ER_fun((1/(N*T))*eigen_val_sort,k_max)[1]
            output_cube[x,4,e] = AH.GR_fun(X,(1/(N*T))*eigen_val_sort,k_max)[1]
            output_cube[x,5,e] = On.ED_fun((1/T)*eigen_val_sort,k_max)[0]

    return output_cube

def monte_carlo_Ncorr(dims_tbl,iter_num,r,theta,beta,k_max):

    """
    Monte Carlo function for cross sectionaly correlated panel data
    -------------------------------------
    inputs:
    ------
    dims_tbl = table which has tuples of dimensions for panel data tests
    iter_num = number of Monte Carlo iterations
    r = true number of factors
    theta = SNR
    beta = strength of correlation to other entries; 0 <= beta < 1
    J = amount of entries the given entry is correlated to => hardcoded in the function
    k_max = maximum assumed number of factors
    outputs:
    -------
    output_cube = 3D array: (dimensional tuples) x (estimators) x (iterations)
    """

    # preallocating 3D array for results
    estimator_num = 6 # hardcoding, depends on the structure of the function
    dims_length = dims_tbl.shape[0]
    output_cube = np.zeros((dims_length,estimator_num,iter_num),dtype=np.intc)
    
    for x in range(0,dims_length):

        """
        looping over panel data dimension tuples
        for each tuple Lambda matrix is geenrated only once
        """
        
        N = dims_tbl[x,0]
        Lambda = norm.rvs(0,1,(N,r))  # matrix of factor loadings - generated once per setting

        for e in range(0,iter_num):
            
            """
            Monte Carlo simulation for the given dimension tuple
            """

            # DGP
            # -------
            J = 8
            #J=int(np.maximum(np.floor(dims_tbl[x,0]/20),10))
            X = dgp.dgp_n_correlations(Lambda, dims_tbl[x,0], dims_tbl[x,1], r, theta, beta, J)
            N,T = X.shape

            # Common part
            # -----------
            eigen_vals = aux.eigenvals_fun(X)
            index = np.argsort(eigen_vals)[::-1]
            eigen_val_sort = eigen_vals[index]

            # Specific part
            # ------------
            output_cube[x,0,e] = BN.IC1_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,1,e] = BN.PC1_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,2,e] = BN.BIC3_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,3,e] = AH.ER_fun((1/(N*T))*eigen_val_sort,k_max)[1]
            output_cube[x,4,e] = AH.GR_fun(X,(1/(N*T))*eigen_val_sort,k_max)[1]
            output_cube[x,5,e] = On.ED_fun((1/T)*eigen_val_sort,k_max)[0]

    return output_cube



def monte_carlo_NTcorr(dims_tbl,iter_num,r,theta,rho,beta,k_max):

    """
    Monte Carlo function for cross sectionaly and temporaly correlated panel data
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

    # preallocating 3D array for results
    estimator_num = 6 # hardcoding, depends on the structure of the function
    dims_length = dims_tbl.shape[0]
    output_cube = np.zeros((dims_length,estimator_num,iter_num),dtype=np.intc)
    
    for x in range(0,dims_length):

        """
        looping over panel data dimension tuples
        for each tuple Lambda matrix is geenrated only once
        """
        
        N = dims_tbl[x,0]
        Lambda = norm.rvs(0,1,(N,r))  # matrix of factor loadings - generated once per setting

        for e in range(0,iter_num):
            
            """
            Monte Carlo simulation for the given dimension tuple
            """

            # DGP
            # -------
            J = 8
            #J=int(np.maximum(np.floor(dims_tbl[x,0]/20),10))
            X = dgp.dgp_nt_correlations(Lambda, dims_tbl[x,0], dims_tbl[x,1], r, theta, rho, beta, J)
            N,T = X.shape

            # Common part
            # -----------
            eigen_vals = aux.eigenvals_fun(X)
            index = np.argsort(eigen_vals)[::-1]
            eigen_val_sort = eigen_vals[index]

            # Specific part
            # ------------
            output_cube[x,0,e] = BN.IC1_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,1,e] = BN.PC1_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,2,e] = BN.BIC3_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,3,e] = AH.ER_fun((1/(N*T))*eigen_val_sort,k_max)[1]
            output_cube[x,4,e] = AH.GR_fun(X,(1/(N*T))*eigen_val_sort,k_max)[1]
            output_cube[x,5,e] = On.ED_fun((1/T)*eigen_val_sort,k_max)[0]

    return output_cube



"""
code for analysis varying parameters


"""





def monte_carlo_Tcorr_rho_var(rho_tbl,T_tbl,N,iter_num,r,theta,k_max):

    """
    Monte Carlo function for temporaly correlated panel data
    -------------------------------------
    inputs:
    ------
    dims_tbl = table which has tuples of dimensions for panel data tests
    iter_num = number of Monte Carlo iterations
    r = true number of factors
    theta = SNR
    rho = strength of temporal correlations; 0 <= rho < 1 (assuming unit root)
    k_max = maximum assumed number of factors
    outputs:
    -------
    output_cube = 3D array: (dimensional tuples) x (estimators) x (iterations)
    """

    # preallocating 3D array for results
    estimator_num = 6 # hardcoding, depends on the structure of the function
    dims_length = T_tbl.shape[0]
    output_cube = np.zeros((dims_length,estimator_num,iter_num),dtype=np.intc)
    
    for x in range(0,dims_length):

        """
        looping over panel data dimension tuples
        for each tuple Lambda matrix is geenrated only once
        """
        
        N = N
        Lambda = norm.rvs(0,1,(N,r))  # matrix of factor loadings - generated once per setting

        for e in range(0,iter_num):
            
            """
            Monte Carlo simulation for the given dimension tuple
            """

            # DGP
            # -------
            X = dgp.dgp_t_correlations(Lambda, N, T_tbl[x,0], r, theta, rho_tbl[x,0])
            N,T = X.shape

            # Common part
            # -----------
            eigen_vals = aux.eigenvals_fun(X)
            index = np.argsort(eigen_vals)[::-1]
            eigen_val_sort = eigen_vals[index]

            # Specific part
            # ------------
            output_cube[x,0,e] = BN.IC1_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,1,e] = BN.PC1_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,2,e] = BN.BIC3_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,3,e] = AH.ER_fun((1/(N*T))*eigen_val_sort,k_max)[1]
            output_cube[x,4,e] = AH.GR_fun(X,(1/(N*T))*eigen_val_sort,k_max)[1]
            output_cube[x,5,e] = On.ED_fun((1/T)*eigen_val_sort,k_max)[0]

    return output_cube





def monte_carlo_Ncorr_beta_var(beta_tbl,T_tbl,N,iter_num,r,theta,k_max):

    """
    Monte Carlo function for cross sectionaly correlated panel data
    -------------------------------------
    inputs:
    ------
    dims_tbl = table which has tuples of dimensions for panel data tests
    iter_num = number of Monte Carlo iterations
    r = true number of factors
    theta = SNR
    beta = strength of correlation to other entries; 0 <= beta < 1
    J = amount of entries the given entry is correlated to => hardcoded in the function
    k_max = maximum assumed number of factors
    outputs:
    -------
    output_cube = 3D array: (dimensional tuples) x (estimators) x (iterations)
    """

    # preallocating 3D array for results
    estimator_num = 6 # hardcoding, depends on the structure of the function
    dims_length = T_tbl.shape[0]
    output_cube = np.zeros((dims_length,estimator_num,iter_num),dtype=np.intc)
    
    for x in range(0,dims_length):

        """
        looping over panel data dimension tuples
        for each tuple Lambda matrix is geenrated only once
        """
        
        N = N
        Lambda = norm.rvs(0,1,(N,r))  # matrix of factor loadings - generated once per setting

        for e in range(0,iter_num):
            
            """
            Monte Carlo simulation for the given dimension tuple
            """

            # DGP
            # -------
            J = 8
            #J=int(np.maximum(np.floor(N/20),10))
            X = dgp.dgp_n_correlations(Lambda, N, T_tbl[x,0], r, theta, beta_tbl[x,0], J)
            N,T = X.shape

            # Common part
            # -----------
            eigen_vals = aux.eigenvals_fun(X)
            index = np.argsort(eigen_vals)[::-1]
            eigen_val_sort = eigen_vals[index]

            # Specific part
            # ------------
            output_cube[x,0,e] = BN.IC1_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,1,e] = BN.PC1_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,2,e] = BN.BIC3_fun(X,eigen_val_sort,k_max)[1] # dividing done inside the function
            output_cube[x,3,e] = AH.ER_fun((1/(N*T))*eigen_val_sort,k_max)[1]
            output_cube[x,4,e] = AH.GR_fun(X,(1/(N*T))*eigen_val_sort,k_max)[1]
            output_cube[x,5,e] = On.ED_fun((1/T)*eigen_val_sort,k_max)[0]

    return output_cube
