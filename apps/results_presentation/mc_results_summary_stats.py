

import os
import numpy as np
import pandas as pd

from src.utils.paths import get_results_path
from src.utils.statistical_utils_latex_bold import rmse_stat_fun, under_exact_over_estimation_stat_fun, \
    joined_mc_stats_latex_fun
from src.utils.utilities import   df_maker

results = np.load(os.path.join(get_results_path(),f"output_cube_nt_snr_0.5_rho_0.5_beta_0.5_iter_num_100.npy"))

aa = pd.read_csv(os.path.join(get_results_path(),f"results_nt_snr_0.5_rho_0.5_beta_0.5_iter_num_100.csv"),index_col=None)

input_tbl = aa.iloc[:,:2].to_numpy()



hmmm = joined_mc_stats_latex_fun(results, 3, precision_ueo_estimation = 0, precision_rmse = 2)


stats_rmse = rmse_stat_fun(results, true_r = 3, precision = 2)




joined_stats_beta = under_exact_over_estimation_stat_fun(results,
                                           true_r = 3, precision = 0)


joined_stats_beta.to_csv(os.path.join(get_results_path(), f"test11.csv"), sep=",", index=False)



#stats_mean = np.mean(results, axis=2, keepdims=False)

stats_rmse = rmse_fun(results, true_r = 3)

#stats_under_over = under_over_estimation_stat_fun(results, true_r = 3)

the_triple = under_exact_over_estimation_stat_fun(results, true_r = 3)

dims_length, estimator_num, iter_num = results.shape
joined  = np.zeros((dims_length, estimator_num), dtype=object)

for i in range(dims_length):
    for j in range(estimator_num):
        joined[i,j] = f"{the_triple[i,j]} ({stats_rmse[i,j]})"




colnames = ["$N$","$T$","$IC_1$","$PC_1$","$BIC_3$","$ER$","$GR$","$ED$","$TKCV$"]


stats_rmse_df = df_maker(input_tbl, stats_rmse, colnames)

#stats_under_over_df = df_maker(input_tbl, stats_under_over, colnames)

the_triple_df = df_maker(input_tbl, the_triple, colnames)

joined_df = df_maker(input_tbl, joined, colnames)


stats_rmse_df.to_csv(os.path.join(get_results_path(), f"stats_rmse_df.csv"), sep=",", index=False)

joined_df.to_csv(os.path.join(get_results_path(), f"joined_df.csv"), sep=",", index=False)




#stats_under_over_df.to_csv(os.path.join(get_results_path(), f"stats_under_over_df.csv"), sep=",", index=False)

the_triple_df.to_csv(os.path.join(get_results_path(), f"the_triple_df.csv"), sep=",", index=False)




print("hello there ")


