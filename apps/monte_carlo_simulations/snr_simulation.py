

"""
code for the monte carlo simulations
"""


# importing packages
# ------------------
import os
import sys
import time
import numpy as np

# importing code from other files
# -------------------------------
import src.simulators.monte_carlo_designs as mc
from src.utils.paths import get_results_path
from src.utils.statistical_utils_latex_bold import joined_mc_stats_latex_fun
from src.utils.statistical_utils_normal import joined_mc_stats_normal_fun
from src.utils.utilities import df_maker

"""
accepting the input from the terminal:
--------------------------------------
"""

N = int(sys.argv[1])
T = int(sys.argv[2])
rho = float(sys.argv[3])
beta = float(sys.argv[4])
J_type = sys.argv[5]        # a string
iter_num = int(sys.argv[6])

# N = 100
# T = 100
# rho = 0.75
# beta = 0.50
# iter_num = 10

suffix = ""

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


print(f"log file for SNR_n_{N}_t_{T}_rho_{rho}_beta_{beta}_Jtype_{J_type}_iter_num_{iter_num}")


with open(os.path.join(get_results_path(),"mc_run_log.txt"), "a") as text_file:
    print(f"log file for SNR_n_{N}_t_{T}_rho_{rho}_beta_{beta}_Jtype_{J_type}_iter_num_{iter_num}", file = text_file)



t0 = time.time()


SNR_list = [0.1, 0.25, 0.50 , 0.75, 1.00, 4/3, 2, 4, 10]

input_tbl = np.array(SNR_list).reshape(-1,1)



# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

"""
exploring the effect of changing SNR for given N, T and rho, beta
"""

# Monte Carlo simulation
# ----------------------

output_cube_snr = mc.monte_carlo_snr(input_tbl, iter_num, burning_period = burning_period,
                                     N = N, T = T, r = r,
                                     rho = rho, beta = beta, J_type = J_type, k_max = k_max,
                                     simulation_seed = simulation_seed)


# saving raw results
# ------------------
np.save(os.path.join(get_results_path(),f"output_cube_snr_n_{N}_t_{T}_rho_{rho}_beta_{beta}_Jtype_{J_type}_iter_num_{iter_num}{suffix}"), output_cube_snr)



# computing statistics
# --------------------
mean_stats_snr = np.mean(output_cube_snr, axis=2, keepdims=False)

joined_stats_normal_snr = joined_mc_stats_normal_fun(output_cube_snr,
                                                     true_r = r, precision_ueo_estimation = 0, precision_rmse = 2)

joined_stats_latex_snr = joined_mc_stats_latex_fun(output_cube_snr,
                                                   true_r = r, precision_ueo_estimation = 0, precision_rmse = 2)


# creating dataframes for export
# ------------------------------

colnames = ["$SNR$","$BN_IC1$","$BN_PC1$","$BN_BIC3$","$ON_ED$","$AH_ER$","$AH_GR$","$WC_TKCV$"]


mean_stats_snr_df = df_maker(input_tbl, mean_stats_snr, colnames)

joined_stats_normal_snr_df = df_maker(input_tbl, joined_stats_normal_snr, colnames)

joined_stats_latex_snr_df = df_maker(input_tbl, joined_stats_latex_snr, colnames)


# exporting
# ---------

mean_stats_snr_df.to_csv(os.path.join(get_results_path(),
                  f"mean_stats_snr_n_{N}_t_{T}_rho_{rho}_beta_{beta}_Jtype_{J_type}_iter_num_{iter_num}{suffix}.csv"), sep=",", index=False)

joined_stats_normal_snr_df.to_csv(os.path.join(get_results_path(),
                   f"joined_stats_normal_snr_n_{N}_t_{T}_rho_{rho}_beta_{beta}_Jtype_{J_type}_iter_num_{iter_num}{suffix}.csv"), sep=",", index=False)

joined_stats_latex_snr_df.to_csv(os.path.join(get_results_path(),
                  f"joined_stats_latex_snr_n_{N}_t_{T}_rho_{rho}_beta_{beta}_Jtype_{J_type}_iter_num_{iter_num}{suffix}.csv"), sep=",", index=False)



t1 = time.time()


# time logs

print(f"total execution time:"
      f" {round(t1 - t0, 2)} sec,"
      f" {round((t1 - t0) / 60, 2)} min,"
      f" {round((t1 - t0) / (60 * 60), 2)} hour")

with open(os.path.join(get_results_path(),"mc_run_log.txt"), "a") as text_file:
    print(f"total execution time:"
          f" {round(t1 - t0, 2)} sec,"
          f" {round((t1 - t0) / 60, 2)} min,"
          f" {round((t1 - t0) / (60 * 60), 2)} hour", file = text_file)


# not sure if this is the correct/best way, but it works

os.rename(os.path.join(get_results_path(),"mc_run_log.txt"),
          os.path.join(get_results_path(),f"mc_run_log_snr_n_{N}_t_{T}_rho_{rho}_beta_{beta}_Jtype_{J_type}_iter_num_{iter_num}.txt"))