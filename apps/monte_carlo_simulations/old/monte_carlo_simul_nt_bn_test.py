

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
from src.simulators.old.monte_carlo_bn_test import monte_carlo_nt
from src.utils.paths import get_results_path
from src.utils.statistical_utils_latex_bold import joined_mc_stats_latex_fun
from src.utils.statistical_utils_normal import joined_mc_stats_normal_fun
from src.utils.utilities import df_maker, mc_dimensions_setting


"""
accepting the input from the terminal:
--------------------------------------
"""

# SNR = float(sys.argv[1])
# rho = float(sys.argv[2])
# beta = float(sys.argv[3])
# iter_num = int(sys.argv[4])


SNR = 1
rho = 0.5
beta = 0.2
iter_num = 50

suffix = "_new_bn_test3"

"""
running the script
--------------------------------------
"""

# setting seed for the DGP
# - always to be done only once, and in the app file, not src file!!!
simulation_seed = 1372


"""
simulation settings
"""


r = 3
k_max = 8

burning_period = 100




# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

print(f"log file for monte carlo nt_snr_{SNR}_rho_{rho}_beta_{beta}_iter_num_{iter_num}")


with open(os.path.join(get_results_path(),"mc_run_log.txt"), "a") as text_file:
    print(f"log file for monte carlo nt_snr_{SNR}_rho_{rho}_beta_{beta}_iter_num_{iter_num}", file = text_file)

t0 = time.time()


N_list = [25,50,100,150,200,250,300,350,400,500]
T_list = [250]

input_tbl_1 = mc_dimensions_setting(N_list,T_list)

N_list = [250]
T_list = [25,50,100,150,200,250,300,350,400,500]

input_tbl_2 = mc_dimensions_setting(N_list,T_list)

N_list = [25,50,100,150,200,250,300,350,400,500]
T_list = [25,50,100,150,200,250,300,350,400,500]

input_tbl_3 = np.concatenate([np.array(N_list).reshape(-1,1),
                              np.array(N_list).reshape(-1,1)],axis = 1)


input_tbl = np.concatenate([input_tbl_1,input_tbl_2,input_tbl_3],axis = 0)

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

"""
exploring the effect of growing NT on a given SNR, rho and beta
"""

output_cube_nt = monte_carlo_nt(input_tbl, iter_num, burning_period = burning_period,
                                   r = r, SNR = SNR, rho = rho, beta = beta, k_max = k_max,
                                   simulation_seed = simulation_seed)


np.save(os.path.join(get_results_path(),f"output_cube_nt_snr_{SNR}_rho_{rho}_beta_{beta}_iter_num_{iter_num}{suffix}"), output_cube_nt)




mean_stats_nt = np.mean(output_cube_nt, axis=2, keepdims=False)

joined_stats_normal_nt = joined_mc_stats_normal_fun(output_cube_nt,
                                                    true_r = r, precision_ueo_estimation = 0, precision_rmse = 2)

joined_stats_latex_nt = joined_mc_stats_latex_fun(output_cube_nt,
                                                    true_r = r, precision_ueo_estimation = 0, precision_rmse = 2)



colnames = ["$N$","$T$","$IC_1$","$PC_1$","$BIC_3$"]

# colnames = ["$N$","$T$","$IC_1$","$PC_1$","$BIC_3$","$ED$","$ER$","$GR$","$TKCV$"]


mean_stats_nt_df = df_maker(input_tbl, mean_stats_nt, colnames)

joined_stats_normal_nt_df = df_maker(input_tbl, joined_stats_normal_nt, colnames)

joined_stats_latex_nt_df = df_maker(input_tbl, joined_stats_latex_nt, colnames)




mean_stats_nt_df.to_csv(os.path.join(get_results_path(), f"mean_stats_nt_snr_{SNR}_rho_{rho}_beta_{beta}_iter_num_{iter_num}{suffix}.csv"), sep=",", index=False)

joined_stats_normal_nt_df.to_csv(os.path.join(get_results_path(), f"joined_stats_normal_nt_snr_{SNR}_rho_{rho}_beta_{beta}_iter_num_{iter_num}{suffix}.csv"), sep=",", index=False)

joined_stats_latex_nt_df.to_csv(os.path.join(get_results_path(), f"joined_stats_latex_nt_snr_{SNR}_rho_{rho}_beta_{beta}_iter_num_{iter_num}{suffix}.csv"), sep=",", index=False)


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
          os.path.join(get_results_path(),f"mc_run_log_nt_snr_{SNR}_rho_{rho}_beta_{beta}_iter_num_{iter_num}.txt"))
