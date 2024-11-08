
"""
code for the monte carlo simulations
"""



# importing packages
# ------------------
import numpy as np
import pandas as pd

# importing code from other files
# ------------------------------- 
import src.computation_functions.old.monte_carlo as mc

# setting seed
# ----
np.random.seed(seed=1372)




# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

""""
common part for all simulations:
* create the output_cube just once, then in
  each setup jost make a new copy, saves space
"""


iter_num = 1000


T_list = np.array([100,100,100,100,100,
                   200,200,200,200,200,
                   300,300,300,300,300,
                   400,400,400,400,400,
                   500,500,500,500,500],dtype=np.intc)
T_list.shape = (len(T_list),1)
N_list = np.array([100,250,500,1000,2000,
                   100,250,500,1000,2000,
                   100,250,500,1000,2000,
                   100,250,500,1000,2000,
                   100,250,500,1000,2000],dtype=np.intc)
N_list.shape = (len(N_list),1)

input_tbl = np.concatenate((N_list,T_list),axis=1)

# function for making dataframes
def df_maker(input_tbl,result_tbl):
    intermediate_tbl = np.concatenate((input_tbl,result_tbl),axis=1,dtype=object)
    col_names = ["$N$","$T$","$IC_1$","$PC_1$","$BIC_3$","$ER$","$GR$","$ED$"]
    intermediate_df = pd.DataFrame(data = intermediate_tbl,
                                   columns = col_names )
    return intermediate_df




# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
""" N and T correlations case:
       """

output_cube_NTcorr = mc.monte_carlo_NTcorr(input_tbl,iter_num,r=3,theta=3,rho=0.5,beta=0.25,k_max=8)

np.save("output_cube_NTcorr_N_large",output_cube_NTcorr)

avg_result_NTcorr = np.mean(output_cube_NTcorr,axis=2,keepdims=False)

results_NTcorr_df = df_maker(input_tbl,avg_result_NTcorr)

results_NTcorr_df.to_csv("results_NTcorr_N_large.csv", sep=",", index=False)








