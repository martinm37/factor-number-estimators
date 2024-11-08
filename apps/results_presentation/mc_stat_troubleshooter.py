

import os
import numpy as np

from src.utils.paths import get_results_path
from src.utils.statistical_utils import joined_mc_statistics_fun
from src.utils.utilities import df_maker, mc_dimensions_setting

SNR = 0.5
rho = 0.5
beta = 0.5
iter_num = 500
r = 3

output_cube_nt = np.load(os.path.join(get_results_path(),f"output_cube_nt_snr_{SNR}_rho_{rho}_beta_{beta}_iter_num_{iter_num}.npy"))



joined_stats_nt = joined_mc_statistics_fun(output_cube_nt,
                                           true_r = r, precision_ueo_estimation = 0, precision_rmse = 2)



N_list = [25,50,100,150,200,250,300,350,400,500]
T_list = [250]

input_tbl = mc_dimensions_setting(N_list,T_list)



joined_stats_nt_df = df_maker(input_tbl, joined_stats_nt,
                              ["$N$","$T$","$IC_1$","$PC_1$","$BIC_3$","$ED$","$ER$","$GR$","$TKCV$"])


joined_stats_nt_df.to_csv(os.path.join(get_results_path(), f"joined_stats_nt_snr_{SNR}_rho_{rho}_beta_{beta}_iter_num_{iter_num}.csv"), sep=",", index=False)



print("ooopps xdd")
