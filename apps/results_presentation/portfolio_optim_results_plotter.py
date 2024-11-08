
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.graphs.portfolio_optim_graphs import optim_subplots, optim_subplots_triple
from src.graphs.portfolio_optim_graphs_v2 import optim_subplots_v2, optim_subplots_triple_v2
from src.utils.paths import get_data_path, get_results_path
from src.utils.utilities import poet_k_hat_stats_fun, poet_stats_full_rounded


# choose the N dimension
# N = [150,200,240,300,400]
# years = [10,14]

N = 400
years = 14
k_max = 20
k_min = 0
dataset = "orig"

figure_extension = "eps"

suffix = ""

stepsize = 4


if years == 14:

    # importing data
    results_df = pd.read_csv(os.path.join(get_results_path(),f"portfolio_optimization_{N}_years_{years}_kmax_{k_max}_kmin_{k_min}_{dataset}_results{suffix}.csv"),index_col=0)
    results_df["UTCTIME"] = pd.to_datetime(results_df["UTCTIME"], format="%Y-%m-%d")

elif years == 10:

    # importing data
    results_df = pd.read_csv(os.path.join(get_results_path(), f"portfolio_optimization_{N}_years_{14}_kmax_{k_max}_kmin_{k_min}_{dataset}_results{suffix}.csv"),index_col=0)
    results_df["UTCTIME"] = pd.to_datetime(results_df["UTCTIME"], format="%Y-%m-%d")
    cutoff_date = datetime.datetime(2020, 1, 3)
    results_df = results_df[(results_df["UTCTIME"] <= cutoff_date)]

else:
    print("incorrect number of years selected")




#desired_estimators = ["BN_IC1", "BN_BIC3","ON_ED","AH_ER", "AH_GR","WC_TKCV","sample_covar_mat","one_over_n"]
desired_estimators = ["BN_IC1", "BN_BIC3","ON_ED","AH_ER", "AH_GR","WC_TKCV","dummy","sample_covar_mat","one_over_n"]


# done for both year lengths
poet_stats = poet_stats_full_rounded(results_df,desired_estimators)

poet_stats.to_csv(os.path.join(get_results_path(), f"summary_stats_{N}_years_{years}_kmax_{k_max}_kmin_{k_min}_{dataset}.csv"),index=True)




if years == 14:
    # done just for the 14 year long time series

    k_hat_df = pd.read_csv(os.path.join(get_results_path(),f"k_hat_{N}_years_{years}_kmax_{k_max}_kmin_{k_min}_{dataset}{suffix}.csv"),index_col=0)
    c_min_df = pd.read_csv(os.path.join(get_results_path(),f"c_min_{N}_years_{years}_kmax_{k_max}_kmin_{k_min}_{dataset}{suffix}.csv"),index_col=0)
    m_star_df = pd.read_csv(os.path.join(get_results_path(),f"m_star_{N}_years_{years}_kmax_{k_max}_kmin_{k_min}_{dataset}{suffix}.csv"),index_col=0)
    c_star_df = pd.read_csv(os.path.join(get_results_path(),f"c_star_{N}_years_{years}_kmax_{k_max}_kmin_{k_min}_{dataset}{suffix}.csv"),index_col=0)

    k_hat_df["UTCTIME"] = pd.to_datetime(k_hat_df["UTCTIME"], format="%Y-%m-%d")
    c_min_df["UTCTIME"] = pd.to_datetime(c_min_df["UTCTIME"], format="%Y-%m-%d")
    m_star_df["UTCTIME"] = pd.to_datetime(m_star_df["UTCTIME"], format="%Y-%m-%d")
    c_star_df["UTCTIME"] = pd.to_datetime(c_star_df["UTCTIME"], format="%Y-%m-%d")


    t_elap_df = pd.read_csv(os.path.join(get_results_path(),f"t_elap_{N}_years_{years}_kmax_{k_max}_kmin_{k_min}_{dataset}{suffix}.csv"),index_col=0)

    t_elap_df = t_elap_df.iloc[1:,1:] # excluding the compilation time, and date column, in preparation for mean statistics

    t_elap_mean_stat = t_elap_df.mean()
    t_elap_mean_df = t_elap_mean_stat.to_frame().transpose()
    t_elap_mean_df = t_elap_mean_df.rename(index={0: f"{N}"})

    print(t_elap_mean_df.style.format(precision=3).to_latex())
    # df_index = [f"{N}"]
    # df_columns = t_elap_mean_stat.index
    # stats_df = pd.DataFrame(t_elap_mean_stat, index=df_index, columns=df_columns)



    #fig = optim_subplots(k_hat_df, f"estimated factor number, N = {N}", [0, 9])
    fig = optim_subplots_v2(k_hat_df, f"estimated factor number, N = {N}", [0, k_max + 1],stepsize = stepsize)
    fig.savefig(os.path.join(get_results_path(), f"estimated_factor_numbers_N_{N}_years{years}_kmax_{k_max}_kmin_{k_min}_{dataset}.{figure_extension}"))
    # plt.show()

    #fig = optim_subplots_triple(c_min_df, c_star_df, m_star_df, f"estimated C_star and its interval, N = {N}")
    fig = optim_subplots_triple_v2(c_min_df, c_star_df, m_star_df, f"estimated C_star and its interval, N = {N}")
    fig.savefig(os.path.join(get_results_path(), f"estimated_c_star_N_{N}_years{years}_kmax_{k_max}_kmin_{k_min}_{dataset}.{figure_extension}"))
    # plt.show()






    # creating series of monthly returns

    results_numpy = results_df.iloc[:,1:].to_numpy()

    T = c_star_df.shape[0]
    N_loop = len(desired_estimators)

    out_of_sample_width = 21


    aggregator = np.zeros([T,N_loop])

    for n in range(N_loop):
        for t in range(len(c_star_df["UTCTIME"])):
            aggregator[t,n] = np.mean(results_numpy[(t * out_of_sample_width) : out_of_sample_width + (t * out_of_sample_width),n])


    results_monthly_df = pd.DataFrame(data = aggregator,
                                      columns = desired_estimators)


    results_monthly_df.insert(0, "UTCTIME", c_star_df["UTCTIME"])

    # plt.figure(figsize=(6.4 * 1.5, 4.8))
    # plt.plot(results_monthly_df["UTCTIME"],results_monthly_df["ED"], color = "tab:green",label = "ED")
    # plt.plot(results_monthly_df["UTCTIME"],results_monthly_df["ER"], color = "tab:red",label = "ER")
    # plt.plot(results_monthly_df["UTCTIME"],results_monthly_df["GR"], color = "tab:blue",label = "GR")
    # plt.plot(results_monthly_df["UTCTIME"],results_monthly_df["one_over_n"], color = "tab:orange",label = "1/n")
    # plt.plot(results_monthly_df["UTCTIME"],np.zeros(len(results_monthly_df["ER"])),"k--")
    # plt.title(f"comparison of monthly returns, N = {N}")
    # plt.legend()
    # plt.savefig(os.path.join(get_results_path(),f"returns_comparison_1_N_{N}_years{years}.{figure_extension}"))
    # #plt.show()

    # plt.plot(results_monthly_df["UTCTIME"],results_monthly_df["IC1"],label = "IC1")
    # plt.plot(results_monthly_df["UTCTIME"],results_monthly_df["BIC3"],label = "BIC3")
    # plt.plot(results_monthly_df["UTCTIME"],np.zeros(len(results_monthly_df["ER"])),"k--")
    # plt.title(f"monthly returns, N = {N}")
    # plt.legend()
    # plt.show()


    # plt.plot(results_monthly_df["UTCTIME"],results_monthly_df["IC1"],label = "IC1")
    # plt.plot(results_monthly_df["UTCTIME"],results_monthly_df["TKCV"],label = "TKCV")
    # #plt.plot(results_monthly_df["UTCTIME"],results_monthly_df["one_over_n"],label = "1/n")
    # plt.plot(results_monthly_df["UTCTIME"],np.zeros(len(results_monthly_df["ER"])),"k--")
    # plt.title(f"monthly returns, N = {N}")
    # plt.legend()
    # plt.show()


    # plt.figure(figsize=(6.4 * 1.5, 4.8))
    # plt.plot(results_monthly_df["UTCTIME"],results_monthly_df["GR"],label = "GR")
    # plt.plot(results_monthly_df["UTCTIME"],results_monthly_df["TKCV"],label = "TKCV")
    # plt.plot(results_monthly_df["UTCTIME"],np.zeros(len(results_monthly_df["ER"])),"k--")
    # plt.title(f"comparison of monthly returns, N = {N}")
    # plt.legend()
    # plt.savefig(os.path.join(get_results_path(),f"returns_comparison_2_N_{N}_years{years}.{figure_extension}"))
    # #plt.show()



#
# fig = optim_subplots_double(m_star_df,c_star_df,f"estimated M and c_star, N = {N}")
# fig.savefig(os.path.join(get_results_path(),f"estimated_M_and_c_star_N_{N}.png"))
# plt.show()



# plt.plot(k_hat_df["ER_k_hat"],label="ER")
# plt.plot(k_hat_df["GR_k_hat"],label="GR")
# plt.plot(k_hat_df["BIC3_k_hat"],label="BIC3")
# plt.plot(k_hat_df["IC1_k_hat"],label="IC1")
# plt.plot(k_hat_df["ED_k_hat"],label="ED")
# plt.title(f"estimated factor numbers, N = {N}")
# plt.legend()
# plt.savefig(os.path.join(get_results_path(),f"estimated_factor_numbers_N_{N}.png"))
# plt.show()
# plt.close()
#
# plt.plot(c_star_df["ER"],label="ER")
# plt.plot(c_star_df["GR"],label="GR")
# plt.plot(c_star_df["BIC3"],label="BIC3")
# plt.plot(c_star_df["IC1"],label="IC1")
# plt.plot(c_star_df["ED"],label="ED")
# plt.title(f"c star, N = {N}")
# plt.legend()
# plt.savefig(os.path.join(get_results_path(),f"c_star_N_{N}.png"))
# plt.show()
# plt.close()
#
# plt.subplot(2,2,1)
# plt.plot(c_star_df["ER"])
# plt.ylabel("ER")
# plt.subplot(2,2,2)
# plt.plot(c_star_df["GR"])
# plt.ylabel("GR")
#
#
# plt.subplot(2,2,3)
# plt.plot(c_star_df["ED"])
# plt.ylabel("ED")
# plt.subplot(2,2,4)
# plt.plot(c_star_df["BIC3"])
# plt.ylabel("BIC3")
# plt.show()


#print("hello there xd")
