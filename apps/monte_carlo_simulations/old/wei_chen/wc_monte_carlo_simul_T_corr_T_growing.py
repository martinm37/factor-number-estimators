
"""
code for the monte carlo simulations
"""



# importing packages
# ------------------
import os
import time
import numpy as np

# importing code from other files
# -------------------------------
import src.computation_functions.old.monte_carlo_wei_chen as mc
from src.utils.paths import get_results_path
from src.utils.utilities import df_maker, mc_dimensions_setting

# setting seed
# ----
np.random.seed(seed=1372)




# -----------------------------------------------------------------------
# -----------------------------------------------------------------------


with open(os.path.join(get_results_path(),"mc_run_log.txt"), "a") as text_file:
    print("log file for _Tcorr_T_large", file = text_file)

t0 = time.time()



iter_num = 1000




N_list = [100,200,300,400,500]
T_list = [100,250,500,1000]
input_tbl = mc_dimensions_setting(N_list,T_list)




# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
""" T correlations case:
       """

output_cube_Tcorr = mc.monte_carlo_Tcorr(input_tbl,iter_num,r=3,theta=3,rho=0.75,k_max=8)


np.save(os.path.join(get_results_path(),"output_cube_Tcorr_T_large"),output_cube_Tcorr)

avg_result_Tcorr = np.mean(output_cube_Tcorr,axis=2,keepdims=False)

results_Tcorr_df = df_maker(input_tbl,avg_result_Tcorr,["$N$","$T$","$TKCV$"])


results_Tcorr_df.to_csv(os.path.join(get_results_path(),"results_Tcorr_T_large.csv"), sep=",", index=False)


t1 = time.time()

print(f"total execution time: {round(t1 - t0, 2)} sec, {round((t1 - t0) / 60, 2)} min, {round((t1 - t0) / (60 * 60), 2)} hour")

with open(os.path.join(get_results_path(),"mc_run_log.txt"), "a") as text_file:
    print(f"total execution time: {round(t1 - t0, 2)} sec, {round((t1 - t0) / 60, 2)} min, {round((t1 - t0) / (60 * 60), 2)} hour", file = text_file)

# renaming the file - as righ now my code is a mess and this is the easiest way
# how to quickly fix it

os.rename(os.path.join(get_results_path(),"mc_run_log.txt"),
          os.path.join(get_results_path(),"mc_run_log_Tcorr_T_large.txt"))






