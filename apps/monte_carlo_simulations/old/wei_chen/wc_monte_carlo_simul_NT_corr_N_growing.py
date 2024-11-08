
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
import src.computation_functions.old.monte_carlo as mc
from src.utils.paths import get_results_path
from src.utils.utilities import df_maker, mc_dimensions_setting

# # setting seed
# # ----
# np.random.seed(seed=1372)

# setting seed for the DGP
# - always to be done only once, and in the app file, not src file!!!
#np.random.seed(seed = 1372)
simulation_seed = 1372


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------



with open(os.path.join(get_results_path(),"mc_run_log.txt"), "a") as text_file:
    print("log file for _NTcorr_N_large", file = text_file)

t0 = time.time()


iter_num = 50



N_list = [100,250,500,1000]
T_list = [100,200,300,400,500]
input_tbl = mc_dimensions_setting(N_list,T_list)




# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
""" N and T correlations case:
       """

output_cube_NTcorr = mc.monte_carlo_NTcorr(input_tbl,iter_num,burning_period = 100,
                                           r=3,SNR=1,rho=0.5,beta=0.25,k_max=8,
                                           simulation_seed = simulation_seed)


np.save(os.path.join(get_results_path(),"output_cube_NTcorr_N_large"),output_cube_NTcorr)

avg_result_NTcorr = np.mean(output_cube_NTcorr,axis=2,keepdims=False)


results_NTcorr_df = df_maker(input_tbl,avg_result_NTcorr,
                             ["$N$","$T$","$IC_1$","$PC_1$","$BIC_3$","$ER$","$GR$","$ED$","$TKCV$"])



results_NTcorr_df.to_csv(os.path.join(get_results_path(),"results_NTcorr_N_large.csv"), sep=",", index=False)


t1 = time.time()


print(f"total execution time:"
      f" {round(t1 - t0, 2)} sec,"
      f" {round((t1 - t0) / 60, 2)} min,"
      f" {round((t1 - t0) / (60 * 60), 2)} hour")

with open(os.path.join(get_results_path(),"mc_run_log.txt"), "a") as text_file:
    print(f"total execution time:"
          f" {round(t1 - t0, 2)} sec,"
          f" {round((t1 - t0) / 60, 2)} min,"
          f" {round((t1 - t0) / (60 * 60), 2)} hour", file = text_file)


# renaming the file - as righ now my code is a mess and this is the easiest way
# how to quickly fix it

os.rename(os.path.join(get_results_path(),"mc_run_log.txt"),
          os.path.join(get_results_path(),"mc_run_log_NTcorr_N_large.txt"))






