
import os
import numpy as np
import pandas as pd

from src.utils.paths import get_results_path

simul = "beta"

r = 5
k_max = 8

N = 300
T = 250

SNR = 0.5
rho = 0.5
beta = 0.2

J_type = "static"

iter_num = 1000

suffix = ""

if simul == "nt":
    results_df = pd.read_csv(os.path.join(get_results_path(),
                  f"joined_stats_latex_{simul}_r_{r}_kmax_{k_max}_snr_{SNR}_rho_{rho}_beta_{beta}_Jtype_{J_type}_iter_num_{iter_num}{suffix}.csv"),index_col=False)
elif simul == "beta":
    results_df = pd.read_csv(os.path.join(get_results_path(),
                  f"joined_stats_latex_{simul}_r_{r}_kmax_{k_max}_n_{N}_t_{T}_snr_{SNR}_rho_{rho}_Jtype_{J_type}_iter_num_{iter_num}.csv"),index_col=False)
                  # f"joined_stats_latex_{simul}_n_{N}_t_{T}_snr_{SNR}_rho_{rho}_Jtype_{J_type}_iter_num_{iter_num}.csv"),index_col=False)
elif simul == "rho":
    results_df = pd.read_csv(os.path.join(get_results_path(),
                  f"joined_stats_latex_{simul}_n_{N}_t_{T}_snr_{SNR}_beta_{beta}_Jtype_{J_type}_iter_num_{iter_num}.csv"),index_col=False)
elif simul == "snr":
    results_df = pd.read_csv(os.path.join(get_results_path(),
                  f"joined_stats_latex_{simul}_n_{N}_t_{T}_rho_{rho}_beta_{beta}_Jtype_{J_type}_iter_num_{iter_num}.csv"),index_col=False)



# converting the results to latex table format
print(results_df.style.format(precision=2).hide(axis="index").to_latex())
