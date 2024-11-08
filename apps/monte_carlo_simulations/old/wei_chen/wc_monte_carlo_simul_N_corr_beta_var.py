
"""
code for the monte carlo simulations
"""



# importing packages
# ------------------
import os
import time
import numpy as np
import pandas as pd

# importing code from other files
# -------------------------------
import src.computation_functions.old.monte_carlo_wei_chen as mc
from src.utils.paths import get_results_path

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

with open(os.path.join(get_results_path(),"mc_run_log.txt"), "a") as text_file:
    print("log file for _Ncorr_beta_var", file = text_file)

t0 = time.time()



iter_num = 1000

beta_list = np.array([0.1,0.1,0.1,0.1,
                     0.2,0.2,0.2,0.2,
                     0.3,0.3,0.3,0.3,
                     0.4,0.4,0.4,0.4])
beta_list.shape = (len(beta_list),1)

T_list = np.array([100,250,500,1000,
                   100,250,500,1000,
                   100,250,500,1000,
                   100,250,500,1000],dtype=np.intc)
T_list.shape = (len(T_list),1)




# function for making dataframes
def df_maker(beta_list,T_list,result_tbl):
    intermediate_tbl = np.concatenate((beta_list,T_list,result_tbl),axis=1,dtype=object)
    col_names = ["$N$","$T$","$TKCV$"]
    intermediate_df = pd.DataFrame(data = intermediate_tbl,
                                   columns = col_names )
    return intermediate_df






# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
""" N correlations case:
       """

output_cube_Ncorr = mc.monte_carlo_Ncorr_beta_var(beta_list,T_list,N=500,iter_num=iter_num,r=3,theta=3,k_max=8)



np.save(os.path.join(get_results_path(),"output_cube_Ncorr_beta_var"),output_cube_Ncorr)


avg_result_Ncorr = np.mean(output_cube_Ncorr,axis=2,keepdims=False)

results_Ncorr_df = df_maker(beta_list,T_list,avg_result_Ncorr)


results_Ncorr_df.to_csv(os.path.join(get_results_path(),"results_Ncorr_beta_var.csv"), sep=",", index=False)

t1 = time.time()

print(f"total execution time: {round(t1 - t0, 2)} sec, {round((t1 - t0) / 60, 2)} min, {round((t1 - t0) / (60 * 60), 2)} hour")

with open(os.path.join(get_results_path(),"mc_run_log.txt"), "a") as text_file:
    print(f"total execution time: {round(t1 - t0, 2)} sec, {round((t1 - t0) / 60, 2)} min, {round((t1 - t0) / (60 * 60), 2)} hour", file = text_file)

# renaming the file - as righ now my code is a mess and this is the easiest way
# how to quickly fix it

os.rename(os.path.join(get_results_path(),"mc_run_log.txt"),
          os.path.join(get_results_path(),"mc_run_log_Ncorr_beta_var.txt"))







