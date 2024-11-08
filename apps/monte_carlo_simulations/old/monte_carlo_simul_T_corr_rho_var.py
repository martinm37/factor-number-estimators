
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

rho_list = np.array([0.2,0.2,0.2,0.2,0.2,
                     0.4,0.4,0.4,0.4,0.4,
                     0.6,0.6,0.6,0.6,0.6,
                     0.8,0.8,0.8,0.8,0.8]) 
rho_list.shape = (len(rho_list),1)

T_list = np.array([100,250,500,1000,2000,
                   100,250,500,1000,2000,
                   100,250,500,1000,2000,
                   100,250,500,1000,2000],dtype=np.intc)
T_list.shape = (len(T_list),1)




# function for making dataframes
def df_maker(rho_list,T_list,result_tbl):
    intermediate_tbl = np.concatenate((rho_list,T_list,result_tbl),axis=1,dtype=object)
    col_names = ["$N$","$T$","$IC_1$","$PC_1$","$BIC_3$","$ER$","$GR$","$ED$"]
    intermediate_df = pd.DataFrame(data = intermediate_tbl,
                                   columns = col_names )
    return intermediate_df






# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
""" T correlations case:
       """
output_cube_Tcorr = mc.monte_carlo_Tcorr_rho_var(rho_list,T_list,N=500,iter_num=iter_num,r=3,theta=3,k_max=8)

np.save("output_cube_Tcorr_rho_var",output_cube_Tcorr)

avg_result_Tcorr = np.mean(output_cube_Tcorr,axis=2,keepdims=False)

results_Tcorr_df = df_maker(rho_list,T_list,avg_result_Tcorr)


results_Tcorr_df.to_csv("results_Tcorr_rho_var.csv", sep=",", index=False)








