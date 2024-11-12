

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
from src.simulators.monte_carlo_designs import monte_carlo_beta
from src.utils.paths import get_results_path
from src.utils.statistical_utils_latex_bold import joined_mc_stats_latex_fun
from src.utils.statistical_utils_normal import joined_mc_stats_normal_fun
from src.utils.utilities import df_maker

"""
accepting the input from the terminal:
--------------------------------------
"""

r = int(sys.argv[1])
k_max = int(sys.argv[2])
N = int(sys.argv[3])
T = int(sys.argv[4])
SNR = float(sys.argv[5])
rho = float(sys.argv[6])
J_type = sys.argv[7]        # a string
iter_num = int(sys.argv[8])

# N = 100
# T = 100
# SNR = 0.75
# rho = 0.75
#iter_num = 1

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

# r = 3
# k_max = 8

burning_period = 100


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------


print(f"log file for beta_r_{r}_kmax_{k_max}_n_{N}_t_{T}_snr_{SNR}_rho_{rho}_Jtype_{J_type}_iter_num_{iter_num}")


with open(os.path.join(get_results_path(),"mc_run_log.txt"), "a") as text_file:
    print(f"log file for beta_r_{r}_kmax_{k_max}_n_{N}_t_{T}_snr_{SNR}_rho_{rho}_Jtype_{J_type}_iter_num_{iter_num}", file = text_file)



t0 = time.time()


#beta_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
beta_list = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

input_tbl = np.array(beta_list).reshape(-1, 1)



# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

"""
exploring the effect of changing beta for given N, T and SNR, rho
"""

# Monte Carlo simulation
# ----------------------

output_cube_beta = monte_carlo_beta(input_tbl, iter_num, burning_period = burning_period,
                                       N = N, T = T, r = r,
                                       SNR = SNR, rho = rho, J_type = J_type, k_max = k_max,
                                       simulation_seed = simulation_seed)


# saving raw results
# ------------------
np.save(os.path.join(get_results_path(),f"output_cube_beta_r_{r}_kmax_{k_max}_n_{N}_t_{T}_snr_{SNR}_rho_{rho}_Jtype_{J_type}_iter_num_{iter_num}"), output_cube_beta)



# computing statistics
# --------------------
mean_stats_beta = np.mean(output_cube_beta, axis=2, keepdims=False)

joined_stats_normal_beta = joined_mc_stats_normal_fun(output_cube_beta,
                                                      true_r = r, precision_ueo_estimation = 0, precision_rmse = 2)

joined_stats_latex_beta = joined_mc_stats_latex_fun(output_cube_beta,
                                                    true_r = r, precision_ueo_estimation = 0, precision_rmse = 2)



# creating dataframes for export
# ------------------------------
colnames = ["$beta$","$BN_IC1$","$BN_PC1$","$BN_BIC3$","$ON_ED$","$AH_ER$","$AH_GR$","$WC_TKCV$"]


mean_stats_beta_df = df_maker(input_tbl, mean_stats_beta, colnames)

joined_stats_normal_beta_df = df_maker(input_tbl, joined_stats_normal_beta, colnames)

joined_stats_latex_beta_df = df_maker(input_tbl, joined_stats_latex_beta, colnames)



# exporting
# ---------

mean_stats_beta_df.to_csv(os.path.join(get_results_path(),
                           f"mean_stats_beta_r_{r}_kmax_{k_max}_n_{N}_t_{T}_snr_{SNR}_rho_{rho}_Jtype_{J_type}_iter_num_{iter_num}{suffix}.csv"), sep=",", index=False)

joined_stats_normal_beta_df.to_csv(os.path.join(get_results_path(),
                            f"joined_stats_normal_beta_r_{r}_kmax_{k_max}_n_{N}_t_{T}_snr_{SNR}_rho_{rho}_Jtype_{J_type}_iter_num_{iter_num}{suffix}.csv"), sep=",", index=False)

joined_stats_latex_beta_df.to_csv(os.path.join(get_results_path(),
                           f"joined_stats_latex_beta_r_{r}_kmax_{k_max}_n_{N}_t_{T}_snr_{SNR}_rho_{rho}_Jtype_{J_type}_iter_num_{iter_num}{suffix}.csv"), sep=",", index=False)





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
          os.path.join(get_results_path(),f"mc_run_log_beta_r_{r}_kmax_{k_max}_n_{N}_t_{T}_snr_{SNR}_rho_{rho}_Jtype_{J_type}_iter_num_{iter_num}.txt"))
