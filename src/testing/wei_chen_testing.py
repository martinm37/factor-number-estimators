

"""
file for adapting wei chen estimator
to non square datasets
"""
import time
import os
import numpy as np
import pandas as pd
import src.estimators.factor_number.wei_chen as wc
from scipy.stats import norm # normal distribution
import src.computation_functions.data_generating_process as dgp
from src.utils.paths import get_results_path

start = time.time()

iter_num = 1000


N_list = np.array([100,100,100,100,
                   200,200,200,200,
                   300,300,300,300,
                   400,400,400,400,
                   500,500,500,500],dtype=np.intc)
N_list.shape = (len(N_list),1)
T_list = np.array([100,250,500,1000,
                   100,250,500,1000,
                   100,250,500,1000,
                   100,250,500,1000,
                   100,250,500,1000],dtype=np.intc)
T_list.shape = (len(T_list),1)


input_tbl = np.concatenate((N_list,T_list),axis=1)



###

dims_tbl = input_tbl


# preallocating 3D array for results
estimator_num = 1  # hardcoding, depends on the structure of the function
dims_length = dims_tbl.shape[0]
output_cube = np.zeros((dims_length, estimator_num, iter_num), dtype=np.intc)


r = 3
theta = r


for x in range(0, dims_length):

    """
    looping over panel data dimension tuples
    for each tuple Lambda matrix is geenrated only once
    """

    N = dims_tbl[x, 0]
    Lambda = norm.rvs(0, 1, (N, r))  # matrix of factor loadings - generated once per setting

    for e in range(0, iter_num):
        """
        Monte Carlo simulation for the given dimension tuple
        """

        # DGP
        # -------
        X = dgp.dgp_nt_correlations(Lambda, dims_tbl[x,0], dims_tbl[x,1], r, theta, rho=0.5, beta=0.3, J=8)


        N,T = X.shape
        j = int(N/5)
        k = int(T/5)


        # Specific part
        # ------------
        output_cube[x, 0, e] = wc.TKCV_fun(X, j=j, k=k, R_max=8)[1]


# function for making dataframes
def df_maker(input_tbl,result_tbl):
    intermediate_tbl = np.concatenate((input_tbl,result_tbl),axis=1,dtype=object)
    col_names = ["$N$","$T$","$TKCV$"]
    intermediate_df = pd.DataFrame(data = intermediate_tbl,
                                   columns = col_names )
    return intermediate_df


np.save(os.path.join(get_results_path(),"test_res_NT"),output_cube)


avg_result_Tcorr = np.mean(output_cube,axis=2,keepdims=False)

results_Tcorr_df = df_maker(input_tbl,avg_result_Tcorr)

results_Tcorr_df.to_csv(os.path.join(get_results_path(),"test_res_NT.csv"), sep=",", index=False)

end = time.time()

print(end - start)

print("hello there xdd")