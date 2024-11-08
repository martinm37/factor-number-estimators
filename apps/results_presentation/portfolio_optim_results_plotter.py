
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


print("hello there xd")

# old code below

# poet_er_stats = poet_k_hat_stats_fun(results_df,"ER_returns")
# poet_gr_stats = poet_k_hat_stats_fun(results_df,"GR_returns")
# poet_ed_stats = poet_k_hat_stats_fun(results_df,"ED_returns")
# poet_ic1_stats = poet_k_hat_stats_fun(results_df,"IC1_returns")
# poet_bic3_stats = poet_k_hat_stats_fun(results_df,"BIC3_returns")
# poet_sample_covar_mat_stats = poet_k_hat_stats_fun(results_df,"sample_covar_mat_returns")
# poet_one_over_n_stats = poet_k_hat_stats_fun(results_df,"one_over_n_returns")
#
#
#
#
#
#
# poet_er_return_std = np.std(results_df["ER_returns"])
# poet_gr_return_std = np.std(results_df["GR_returns"])
# poet_ed_return_std = np.std(results_df["ED_returns"])
# poet_ic1_return_std = np.std(results_df["IC1_returns"])
# poet_bic3_return_std = np.std(results_df["BIC3_returns"])
# sample_covar_mat_return_std = np.std(results_df["sample_covar_mat_returns"])
# one_over_n_return_std = np.std(results_df["one_over_n_returns"])
#
# poet_er_return_std_yearly =  poet_er_return_std * np.sqrt(250)
# poet_gr_return_std_yearly =  poet_gr_return_std * np.sqrt(250)
# poet_ed_return_std_yearly =  poet_ed_return_std * np.sqrt(250)
# poet_ic1_return_std_yearly =  poet_ic1_return_std * np.sqrt(250)
# poet_bic3_return_std_yearly =  poet_bic3_return_std * np.sqrt(250)
# sample_covar_mat_return_std_yearly =  sample_covar_mat_return_std * np.sqrt(250)
# one_over_n_return_std_yearly =  one_over_n_return_std * np.sqrt(250)
#
#
# print(f"poet_er_return_std: {np.round(poet_er_return_std,2)}")
# print(f"poet_gr_return_std: {np.round(poet_gr_return_std,2)}")
# print(f"poet_ed_return_std: {np.round(poet_ed_return_std,2)}")
# print(f"poet_ic1_return_std: {np.round(poet_ic1_return_std,2)}")
# print(f"poet_bic3_return_std: {np.round(poet_bic3_return_std,2)}")
# print(f"sample_covar_mat_return_std: {np.round(sample_covar_mat_return_std,2)}")
# print(f"one_over_n_return_std: {np.round(one_over_n_return_std,2)}")
#
# print(f"")
#
# print(f"poet_er_return_std_yearly: {np.round(poet_er_return_std_yearly,2)}")
# print(f"poet_gr_return_std_yearly: {np.round(poet_gr_return_std_yearly,2)}")
# print(f"poet_ed_return_std_yearly: {np.round(poet_ed_return_std_yearly,2)}")
# print(f"poet_ic1_return_std_yearly: {np.round(poet_ic1_return_std_yearly,2)}")
# print(f"poet_bic3_return_std_yearly: {np.round(poet_bic3_return_std_yearly,2)}")
# print(f"sample_covar_mat_return_std_yearly: {np.round(sample_covar_mat_return_std_yearly,2)}")
# print(f"one_over_n_return_std_yearly: {np.round(one_over_n_return_std_yearly,2)}")
#
#
# plt.plot(results_df["UTCTIME"],results_df["ER_returns"],label="ER_returns")
# plt.plot(results_df["UTCTIME"],results_df["GR_returns"],label="GR_returns")
# plt.plot(results_df["UTCTIME"],results_df["ED_returns"],label="ED_returns")
# plt.plot(results_df["UTCTIME"],results_df["IC1_returns"],label="IC1_returns")
# plt.plot(results_df["UTCTIME"],results_df["BIC3_returns"],label="BIC3_returns")
# plt.plot(results_df["UTCTIME"],results_df["sample_covar_mat_returns"],label="sample_covar_mat_returns")
# plt.plot(results_df["UTCTIME"],results_df["one_over_n_returns"],label="one_over_n_returns")
# plt.legend()
# plt.show()
# plt.close()
#
#











# ********************************************************
# old old old code below
# ********************************************************

#
# BIC3_var = returns_df["BIC3"].var()
# ER_var = returns_df["ER"].var()
# GR_var = returns_df["GR"].var()
# ED_var = returns_df["ED"].var()
#
#
# print(f"BIC3_var: {np.sqrt(BIC3_var * 12)}")
# print(f"ER_var: {np.sqrt(ER_var * 12)}")
# print(f"GR_var: {np.sqrt(GR_var * 12)}")
# print(f"ED_var: {np.sqrt(ED_var * 12)}")
#
# # print(f"BIC3_var: {np.round(BIC3_var,2) * np.sqrt(12)}")
# # print(f"ER_var: {np.round(ER_var,2)* np.sqrt(12)}")
# # print(f"GR_var: {np.round(GR_var,2)* np.sqrt(12)}")
# # print(f"ED_var: {np.round(ED_var,2)* np.sqrt(12)}")
#
#
# fig, axs = plt.subplots(2, 2, figsize=(6.4 * 1.5, 4.8), layout="constrained")
#
# # fig, axs = plt.subplots(3,2,layout="constrained")
#
# #fig.suptitle(date + " " + policy_type + " vs. baseline" , fontsize=15)
#
# axs[0, 0].plot(returns_df["BIC3"])
# axs[0, 0].set_title("BIC3")
#
# axs[0, 1].plot(returns_df["ER"])
# axs[0, 1].set_title("ER")
#
# axs[1, 0].plot(returns_df["GR"])
# axs[1, 0].set_title("GR")
#
# axs[1, 1].plot(returns_df["ED"])
# axs[1, 1].set_title("ED")





# axs[1, 0].plot(time_vector, charge_policy)
# axs[1, 0].plot(time_vector, charge_nocontrol)
# axs[1, 0].plot(time_vector, zeros_vector, "k--")
# axs[1, 0].set_ylabel("charge [kW]")
# axs[1, 0].set_xlabel("time [hour]")
# axs[1, 0].set_xticks(np.arange(1, hrz, step=1))
#
# axs[0, 1].plot(time_vector, energy_policy, label="optimal control")
# axs[0, 1].plot(time_vector, energy_nocontrol, label="baseline")
# axs[0, 1].plot(time_vector, zeros_vector, "k--")
# axs[0, 1].set_ylabel("energy in battery [kWh]")
# axs[0, 1].set_xlabel("time [hour]")
# axs[0, 1].set_xticks(np.arange(1, hrz, step=1))
# axs[0, 1].legend(bbox_to_anchor=(1.00,0.4))
#
# axs[1, 1].plot(time_vector, cost_policy.cumsum())
# axs[1, 1].plot(time_vector, cost_nocontrol.cumsum())
# axs[1, 1].plot(time_vector, zeros_vector, "k--")
# axs[1, 1].set_ylabel("cumulative cost [€]")
# axs[1, 1].set_xlabel("time [hour]")
# axs[1, 1].set_xticks(np.arange(1, hrz, step=1))


# plt.show()
