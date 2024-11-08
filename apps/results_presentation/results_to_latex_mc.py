
import os
import numpy as np
import pandas as pd

from src.utils.paths import get_results_path
#from src.utils.statistical_utils_latex_bold import joined_mc_stats_normal_fun
from src.utils.statistical_utils_normal import joined_mc_stats_normal_fun
from src.utils.utilities import df_maker, df_joiner, mc_dimensions_setting



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






#results_df.to_latex()

#print(results_df.style.format(precision=2).hide(axis="index").to_latex())
#print(results_df.style.format(precision=2).to_latex())





# old code below where it was not fully integrated:
# ------------------------------------------------


#results_df = pd.read_csv(os.path.join(get_results_path(),f"portfolio_optimization_{N}_results.csv"),index_col=0)


# reading an output cube
#
# simul = "nt"
#
# r = 3
# # N = 250
# # T = 250
#
# SNR = 1
# rho = 0.5
# beta = 0.2
# iter_num = 35
#
# #addition = "_try1"
# addition = "_bntest"
# # nt
# # -----------------------------------------------
#
# results = np.load(os.path.join(get_results_path(),
#                                       f"output_cube_{simul}_snr_{SNR}_rho_{rho}_beta_{beta}_iter_num_{iter_num}{addition}.npy"))
#
#
#
# N_list = [25,50,100,150,200,250,300,350,400,500]
# T_list = [250]
#
# input_tbl_1 = mc_dimensions_setting(N_list,T_list)
#
# N_list = [250]
# T_list = [25,50,100,150,200,250,300,350,400,500]
#
# input_tbl_2 = mc_dimensions_setting(N_list,T_list)
#
# N_list = [25,50,100,150,200,250,300,350,400,500]
# T_list = [25,50,100,150,200,250,300,350,400,500]
#
# input_tbl_3 = np.concatenate([np.array(N_list).reshape(-1,1),
#                               np.array(N_list).reshape(-1,1)],axis = 1)
#
#
# input_tbl = np.concatenate([input_tbl_1,input_tbl_2,input_tbl_3],axis = 0)
#
#
# #colnames = ["$N$","$T$","$IC_1$","$PC_1$","$BIC_3$","$ED$","$ER$","$GR$","$TKCV$"]
# colnames = ["$N$","$T$","$IC_1$","$PC_1$","$BIC_3$"]
# # -----------------------------------------------
#
#
# # beta
# # -----------------------------------------------
#
# # results = np.load(os.path.join(get_results_path(),
# #                                       f"output_cube_{simul}_n_{N}_t_{T}_snr_{SNR}_rho_{rho}_iter_num_{iter_num}.npy"))
# #
# # beta_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# #
# # input_tbl = np.array(beta_list).reshape(-1, 1)
# # colnames = ["$beta$","$IC_1$","$PC_1$","$BIC_3$","$ED$","$ER$","$GR$","$TKCV$"]
# # -----------------------------------------------
#
# # rho
# # -----------------------------------------------
#
# # results = np.load(os.path.join(get_results_path(),
# #                                       f"output_cube_{simul}_n_{N}_t_{T}_snr_{SNR}_beta_{beta}_iter_num_{iter_num}.npy"))
# #
# # rho_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# # input_tbl = np.array(rho_list).reshape(-1, 1)
# # colnames = ["$rho$","$IC_1$","$PC_1$","$BIC_3$","$ED$","$ER$","$GR$","$TKCV$"]
#
# # SNR
# # -----------------------------------------------
# # results = np.load(os.path.join(get_results_path(),
# #                                       f"output_cube_{simul}_n_{N}_t_{T}_rho_{rho}_beta_{beta}_iter_num_{iter_num}.npy"))
# #
# # SNR_list = [0.1, 0.25, 0.50 , 0.75, 1.00, 4/3, 2, 4, 10]
# #
# # input_tbl = np.array(SNR_list).reshape(-1,1)
# # colnames = ["$SNR$","$IC_1$","$PC_1$","$BIC_3$","$ED$","$ER$","$GR$","$TKCV$"]
#
# # -----------------------------------------------
#
#
#
#
# joined_stats_for_latex = joined_mc_stats_normal_fun(results, true_r=r, precision_ueo_estimation=0, precision_rmse=2)
#
# #
# # joined_stats_for_latex_df = df_joiner(input_tbl, joined_stats_for_latex, colnames)
# joined_stats_for_latex_df = df_maker(input_tbl, joined_stats_for_latex, colnames)
#
#
# print("hello")
#
# #results_df.to_latex()
#
# #print(results_df.style.format(precision=2).hide(axis="index").to_latex())
# print(joined_stats_for_latex_df.style.format(precision=2).hide(axis="index").to_latex())