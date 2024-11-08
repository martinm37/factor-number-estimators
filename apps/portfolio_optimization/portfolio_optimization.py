

import os
import sys
import time
import pandas as pd
import numpy as np

from src.models.portfolio_optim_configs import PortfolioOptimConfig, POETConfig
from src.models.portfolio_optim_solutions import BasicSolverSolution
from src.simulators.portfolio_optimizers import POETPortfolioOptimizer, BasicPortfolioOptimizer

from src.utils.paths import get_data_path, get_results_path

# choose N dimension

# N = int(sys.argv[1])
# years = int(sys.argv[2])
# k_max = int(sys.argv[3])
# k_min = int(sys.argv[4])
# dataset = sys.argv[5] # orig or edit

N = 150
years = 14
k_max = 8
k_min = 0
dataset = "orig"

t0_optim = time.time()


# importing data
daily_returns_df = pd.read_csv(os.path.join(get_data_path(),f"SP500_{N}_daily_returns_years_{years}_{dataset}.csv"))


# starting the log file
print(f"log file for SP500_{N}_daily_returns_years_{years}_kmax_{k_max}_kmin_{k_min}_{dataset} data")


with open(os.path.join(get_results_path(),"portfolio_optim_log.txt"), "a") as text_file:
    print(f"log file for SP500_{N}_daily_returns_years_{years}_kmax_{k_max}_kmin_{k_min}_{dataset} data", file = text_file)



# converting to numpy
daily_returns = daily_returns_df.iloc[:,1:].to_numpy()

"""
rows: dates
columns : returns
"""

# slicing
training_sample_width = 250
out_of_sample_width = 21

portfolio_optimization_config = PortfolioOptimConfig(training_sample_width = training_sample_width,
                                                     out_of_sample_width = out_of_sample_width)


poet_config = POETConfig(cross_validation_fold_number = 5,
                         C_grid_precision= 0.1,
                         cross_validation_precision= 0.1)


poet_portfolio_optimizer = POETPortfolioOptimizer(portfolio_optim_config = portfolio_optimization_config,
                                                  poet_config = poet_config)

basic_portfolio_optimizer = BasicPortfolioOptimizer(portfolio_optim_config = portfolio_optimization_config)




# POET solver
# ------------

poet_ic1_solution = poet_portfolio_optimizer.eigenvalue_based_k_estimator(daily_returns, estimator ="IC1", k_max = k_max,k_min=k_min)
poet_bic3_solution = poet_portfolio_optimizer.eigenvalue_based_k_estimator(daily_returns, estimator ="BIC3", k_max = k_max,k_min=k_min)
poet_ed_solution = poet_portfolio_optimizer.eigenvalue_based_k_estimator(daily_returns, estimator ="ED", k_max = k_max,k_min=k_min)
poet_er_solution = poet_portfolio_optimizer.eigenvalue_based_k_estimator(daily_returns, estimator ="ER", k_max = k_max,k_min=k_min)
poet_gr_solution = poet_portfolio_optimizer.eigenvalue_based_k_estimator(daily_returns, estimator ="GR", k_max = k_max,k_min=k_min)

poet_tkcv_solution = poet_portfolio_optimizer.cross_validation_based_k_estimator(daily_returns,estimator = "TKCV", fold_number = 5, k_max = k_max,k_min=k_min)

poet_dummy_solution = poet_portfolio_optimizer.dummy_k_estimator(daily_returns, k_hat = k_max)


# basic solvers
# -------------

if N < training_sample_width:
    sample_covar_mat_solution = basic_portfolio_optimizer.portfolio_optim_sample_covariance_matrix(daily_returns)
else:
    sample_covar_mat_solution = BasicSolverSolution(daily_returns_vec = np.zeros(len(daily_returns) - training_sample_width))

one_over_n_solution = basic_portfolio_optimizer.portfolio_optim_one_over_n(daily_returns)



# unpacking the results
# ---------------------------------------------

dict_returns = {
    "BN_IC1": poet_ic1_solution.daily_returns_vec,
    "BN_BIC3": poet_bic3_solution.daily_returns_vec,
    "ON_ED": poet_ed_solution.daily_returns_vec,
    "AH_ER": poet_er_solution.daily_returns_vec,
    "AH_GR": poet_gr_solution.daily_returns_vec,
    "WC_TKCV": poet_tkcv_solution.daily_returns_vec,
    "dummy": poet_dummy_solution.daily_returns_vec,
    "sample_covar_mat": sample_covar_mat_solution.daily_returns_vec,
    "one_over_n": one_over_n_solution.daily_returns_vec
}
returns_df = pd.DataFrame(dict_returns)

dict_k_hat = {
    "BN_IC1": poet_ic1_solution.k_estimate_vec,
    "BN_BIC3": poet_bic3_solution.k_estimate_vec,
    "ON_ED": poet_ed_solution.k_estimate_vec,
    "AH_ER": poet_er_solution.k_estimate_vec,
    "AH_GR": poet_gr_solution.k_estimate_vec,
    "WC_TKCV": poet_tkcv_solution.k_estimate_vec,
    "dummy": poet_dummy_solution.k_estimate_vec
}

k_hat_df = pd.DataFrame(dict_k_hat)


dict_c_min_estimate = {
    "BN_IC1": poet_ic1_solution.C_min_estimate_vec,
    "BN_BIC3": poet_bic3_solution.C_min_estimate_vec,
    "ON_ED": poet_ed_solution.C_min_estimate_vec,
    "AH_ER": poet_er_solution.C_min_estimate_vec,
    "AH_GR": poet_gr_solution.C_min_estimate_vec,
    "WC_TKCV": poet_tkcv_solution.C_min_estimate_vec,
    "dummy": poet_dummy_solution.C_min_estimate_vec
}
c_min_est_df = pd.DataFrame(dict_c_min_estimate)


dict_m_estimate = {
    "BN_IC1": poet_ic1_solution.M_estimate_vec,
    "BN_BIC3": poet_bic3_solution.M_estimate_vec,
    "ON_ED": poet_ed_solution.M_estimate_vec,
    "AH_ER": poet_er_solution.M_estimate_vec,
    "AH_GR": poet_gr_solution.M_estimate_vec,
    "WC_TKCV": poet_tkcv_solution.M_estimate_vec,
    "dummy": poet_dummy_solution.M_estimate_vec
}

m_est_df = pd.DataFrame(dict_m_estimate)


dict_c_star = {
    "BN_IC1": poet_ic1_solution.C_star_vec,
    "BN_BIC3": poet_bic3_solution.C_star_vec,
    "ON_ED": poet_ed_solution.C_star_vec,
    "AH_ER": poet_er_solution.C_star_vec,
    "AH_GR": poet_gr_solution.C_star_vec,
    "WC_TKCV": poet_tkcv_solution.C_star_vec,
    "dummy": poet_dummy_solution.C_star_vec
}

c_star_df = pd.DataFrame(dict_c_star)


dict_t_elap = {
    "BN_IC1": poet_ic1_solution.t_elap_vec,
    "BN_BIC3": poet_bic3_solution.t_elap_vec,
    "ON_ED": poet_ed_solution.t_elap_vec,
    "AH_ER": poet_er_solution.t_elap_vec,
    "AH_GR": poet_gr_solution.t_elap_vec,
    "WC_TKCV": poet_tkcv_solution.t_elap_vec,
    "dummy": poet_dummy_solution.t_elap_vec
}

t_elap_df = pd.DataFrame(dict_t_elap)




# re-inserting the time column

# long

time_cut = daily_returns_df.iloc[training_sample_width:,:]

time_col_long = time_cut["UTCTIME"]
time_col_long = time_col_long.reset_index()
time_col_long = time_col_long.drop("index", axis=1)

returns_df.insert(0, "UTCTIME", time_col_long)


# short

time_col_short = []

for i in range(len(c_min_est_df["BN_BIC3"])):
    time_col_short.append(time_col_long.iloc[i * out_of_sample_width,0])

c_min_est_df.insert(0, "UTCTIME", time_col_short)
m_est_df.insert(0, "UTCTIME", time_col_short)
c_star_df.insert(0, "UTCTIME", time_col_short)
k_hat_df.insert(0, "UTCTIME", time_col_short)
t_elap_df.insert(0, "UTCTIME", time_col_short)




# exporting
returns_df.to_csv(os.path.join(get_results_path(), f"portfolio_optimization_{N}_years_{years}_kmax_{k_max}_kmin_{k_min}_{dataset}_results.csv"))

c_min_est_df.to_csv(os.path.join(get_results_path(), f"c_min_{N}_years_{years}_kmax_{k_max}_kmin_{k_min}_{dataset}.csv"))
m_est_df.to_csv(os.path.join(get_results_path(), f"m_star_{N}_years_{years}_kmax_{k_max}_kmin_{k_min}_{dataset}.csv"))
c_star_df.to_csv(os.path.join(get_results_path(), f"c_star_{N}_years_{years}_kmax_{k_max}_kmin_{k_min}_{dataset}.csv"))
k_hat_df.to_csv(os.path.join(get_results_path(), f"k_hat_{N}_years_{years}_kmax_{k_max}_kmin_{k_min}_{dataset}.csv"))
t_elap_df.to_csv(os.path.join(get_results_path(), f"t_elap_{N}_years_{years}_kmax_{k_max}_kmin_{k_min}_{dataset}.csv"))


t1_optim = time.time()

print(f"total execution time:"
      f" {round(t1_optim - t0_optim, 2)} sec,"
      f" {round((t1_optim - t0_optim) / 60, 2)} min,"
      f" {round((t1_optim - t0_optim) / (60 * 60), 2)} hour")



with open(os.path.join(get_results_path(),"portfolio_optim_log.txt"), "a") as text_file:
    print(f"total execution time:"
          f" {round(t1_optim - t0_optim, 2)} sec,"
          f" {round((t1_optim - t0_optim) / 60, 2)} min,"
          f" {round((t1_optim - t0_optim) / (60 * 60), 2)} hour",file = text_file)



os.rename(os.path.join(get_results_path(),"portfolio_optim_log.txt"),
          os.path.join(get_results_path(),f"portfolio_optim_log_N_{N}_years_{years}_kmax_{k_max}_kmin_{k_min}_{dataset}.txt"))



#print("hello there xdddd")


# ****************************************************
# unpacking the results - test of bic3
# ---------------------------------------------
#
# dict_returns = {
#     "BIC3": poet_bic3_solution.daily_returns_vec,
#     "sample_covar_mat": sample_covar_mat_solution.daily_returns_vec,
#     "one_over_n": one_over_n_solution.daily_returns_vec
# }
# returns_df = pd.DataFrame(dict_returns)
#
# dict_k_hat = {
#     "BIC3": poet_bic3_solution.k_estimate_vec,
# }
#
# k_hat_df = pd.DataFrame(dict_k_hat)
#
#
# dict_c_min_estimate = {
#     "BIC3": poet_bic3_solution.C_min_estimate_vec,
# }
# c_min_est_df = pd.DataFrame(dict_c_min_estimate)
#
#
# dict_m_estimate = {
#     "BIC3": poet_bic3_solution.M_estimate_vec,
# }
#
# m_est_df = pd.DataFrame(dict_m_estimate)
#
#
# dict_c_star = {
#     "BIC3": poet_bic3_solution.C_star_vec,
# }
#
# c_star_df = pd.DataFrame(dict_c_star)
#
#






# ****************************************************




# # cross validation settings for c
# C_min_plus_eps = 0.5
# # M = 3.0
#
# precision = 0.1 # no, its both for M and c_star
# # this is for the M calculation!!! -> can shift it to 0.2 and i save half the time on this
#

#
# poet_er_returns, poet_er_k_estimates, er_c_estimates, er_m_estimates = portfolio_optimizer_poet_er(daily_returns,training_sample_width,out_of_sample_width, C_min_plus_eps = C_min_plus_eps,  precision = precision)
# poet_gr_returns, poet_gr_k_estimates, gr_c_estimates, gr_m_estimates = portfolio_optimizer_poet_gr(daily_returns,training_sample_width,out_of_sample_width, C_min_plus_eps = C_min_plus_eps, precision = precision)
# poet_ed_returns, poet_ed_k_estimates, ed_c_estimates, ed_m_estimates = portfolio_optimizer_poet_ed(daily_returns,training_sample_width,out_of_sample_width, C_min_plus_eps = C_min_plus_eps,  precision = precision)
# poet_ic1_returns, poet_ic1_k_estimates, ic1_c_estimates, ic1_m_estimates = portfolio_optimizer_poet_ic1(daily_returns,training_sample_width,out_of_sample_width, C_min_plus_eps = C_min_plus_eps,  precision = precision)
# poet_bic3_returns, poet_bic3_k_estimates, bic3_c_estimates, bic3_m_estimates = portfolio_optimizer_poet_bic3(daily_returns,training_sample_width,out_of_sample_width, C_min_plus_eps = C_min_plus_eps,  precision = precision)
# #sample_covar_mat_returns = portfolio_optimizer_sample_covariance(daily_returns,training_sample_width,out_of_sample_width)
# one_over_n_returns = portfolio_optimizer_1_over_n(daily_returns,training_sample_width,out_of_sample_width)
#
#
#
# dict_returns = {
#     "ER": poet_er_returns,
#     "GR": poet_gr_returns,
#     "ED": poet_ed_returns,
#     "IC1": poet_ic1_returns,
#     "BIC3": poet_bic3_returns,
#     "one_over_n": one_over_n_returns,
# }
#
#
#
# # dict_returns = {
# #     "ER": poet_er_returns,
# #     "GR": poet_gr_returns,
# #     "ED": poet_ed_returns,
# #     "IC1": poet_ic1_returns,
# #     "BIC3": poet_bic3_returns,
# #     "sample_covar_mat": sample_covar_mat_returns,
# #     "one_over_n": one_over_n_returns,
# # }
#
# returns_df = pd.DataFrame(dict_returns)
#
#
# dict_k_hat = {
#     "ER": poet_er_k_estimates,
#     "GR": poet_gr_k_estimates,
#     "ED": poet_ed_k_estimates,
#     "IC1": poet_ic1_k_estimates,
#     "BIC3": poet_bic3_k_estimates,
# }
#
# k_hat_df = pd.DataFrame(dict_k_hat)
#
# dict_c_star = {
#     "ER": er_c_estimates,
#     "GR": gr_c_estimates,
#     "ED": ed_c_estimates,
#     "IC1": ic1_c_estimates,
#     "BIC3": bic3_c_estimates,
#
# }
#
# c_star_df = pd.DataFrame(dict_c_star)
#
#
# dict_m_estimate = {
#     "ER": er_m_estimates,
#     "GR": gr_m_estimates,
#     "ED": ed_m_estimates,
#     "IC1": ic1_m_estimates,
#     "BIC3": bic3_m_estimates,
#
# }
#
# m_est_df = pd.DataFrame(dict_m_estimate)
#
#
# # inserting the time column again
#
# time_cut = daily_returns_df.iloc[training_sample_width:,:]
#
# time_col = time_cut["UTCTIME"]
# time_col = time_col.reset_index()
# time_col = time_col.drop("index",axis=1)
#
# #hmm.shift(-training_sample_width)
#
# returns_df.insert(0, "UTCTIME", time_col)
#
#
#
# returns_df.to_csv(os.path.join(get_results_path(), f"portfolio_optimization_{N}_results.csv"))
#
# k_hat_df.to_csv(os.path.join(get_results_path(), f"k_hat_{N}.csv"))
#
# c_star_df.to_csv(os.path.join(get_results_path(), f"c_star_{N}.csv"))
#
# m_est_df.to_csv(os.path.join(get_results_path(), f"m_star_{N}.csv"))
#
#
# t1_optim = time.time()
#
#
# print(f"total elapsed time: {t1_optim - t0_optim} sec")
#
#


#calculating the statistics - now done elsewhere

#
# poet_er_sats = {
#     "MEAN" : np.mean(returns_df["ER_returns"],axis=0),
#     "STD": np.std(returns_df["ER_returns"]),
#     "SHARPE_RATIO": np.mean(returns_df["ER_returns"],axis=0) / np.std(returns_df["ER_returns"]) # shouldnt this also include the risk free rate?
#
# }
#
#
#
# poet_er_return_mean_yearly = np.mean(returns_df["ER_returns"],axis=0)
# poet_gr_return_std = np.std(returns_df["GR_returns"])
# poet_ed_return_std = np.std(returns_df["ED_returns"])
# poet_ic1_return_std = np.std(returns_df["IC1_returns"])
# poet_bic3_return_std = np.std(returns_df["BIC3_returns"])
# sample_covar_mat_return_std = np.std(returns_df["sample_covar_mat_returns"])
# one_over_n_return_std = np.std(returns_df["one_over_n_returns"])
#
#
# poet_er_return_std_yearly =  np.std(returns_df["ER_returns"]) * np.sqrt(250)
# poet_gr_return_std_yearly =  np.std(returns_df["GR_returns"]) * np.sqrt(250)
# poet_ed_return_std_yearly =  np.std(returns_df["ED_returns"]) * np.sqrt(250)
# poet_ic1_return_std_yearly =  np.std(returns_df["IC1_returns"]) * np.sqrt(250)
# poet_bic3_return_std_yearly =  np.std(returns_df["BIC3_returns"]) * np.sqrt(250)
# sample_covar_mat_return_std_yearly =  np.std(returns_df["sample_covar_mat_returns"]) * np.sqrt(250)
# one_over_n_return_std_yearly =  np.std(returns_df["one_over_n_returns"]) * np.sqrt(250)
#
#
#
#
#






#
# returnr_std_poet_er = np.std(poet_er_returns)
#
# retunr_std_yearly_poet_er =  returnr_std_poet_er * np.sqrt(250)
# #
# #
# # plt.plot(poet_er_returns)
# # plt.show()
# #
# # plt.plot(poet_er_k_estimates)
# # plt.show()
#
#
# sample_covar_mat_returns = portfolio_optimizer_sample_covariance(daily_returns,training_sample_width,out_of_sample_width)
#
#
# returnr_std = np.std(sample_covar_mat_returns)
#
# retunr_std_yearly =  returnr_std * np.sqrt(250)
#
#
#
# plt.plot(sample_covar_mat_returns)
# plt.plot(poet_er_returns)
# plt.show()






# returns_bic3 = portfolio_optimizer_poet_bic3(daily_returns,training_sample_width,out_of_sample_width)
# returns_er = portfolio_optimizer_poet_er(daily_returns,training_sample_width,out_of_sample_width)
# returns_gr = portfolio_optimizer_poet_gr(daily_returns,training_sample_width,out_of_sample_width)
# returns_ed = portfolio_optimizer_poet_ed(daily_returns,training_sample_width,out_of_sample_width)


# dict = {
#     "BIC3": returns_bic3,
#     "ER": returns_er,
#     "GR": returns_gr,
#     "ED": returns_ed,
# }
#
# returns_df = pd.DataFrame(dict)
#
# returns_df.to_csv(os.path.join(get_data_path(),"portfolio_optimization_results.csv"))



# plt.plot(returns_er)
# plt.show()
















# old code below

#
# iter_num = int(np.floor((len(daily_returns) - training_sample_width) / out_of_sample_width))
#
# storage_vector = np.zeros(iter_num)
#
# iter_counter = 0
#
# for t in range(iter_num):
#
#     iter_counter += 1
#
#     if iter_counter == 10:
#         print(f"iteration number: {t+1}")
#         iter_counter = 0
#
#
#     whole_sample = daily_returns[ (t * out_of_sample_width): (training_sample_width + out_of_sample_width) + (t * out_of_sample_width), :]
#
#     training_sample = whole_sample[ 0 : training_sample_width, :]
#
#     out_of_sample = whole_sample[ training_sample_width: (training_sample_width + out_of_sample_width), :]
#
#     # transposing
#     training_sample = training_sample.T
#     out_of_sample = out_of_sample.T
#
#     """
#     rows: returns
#     columns : dates
#     """
#
#     # flipping, so that the oldest values are on the left
#     training_sample = np.flip(training_sample, axis=1)
#     out_of_sample = np.flip(out_of_sample, axis=1)
#
#     # time de - meaning
#     #training_sample = training_sample - np.mean(training_sample,axis=1).reshape(-1,1)
#
#     # training sample
#     # ***************
#
#     # computing eigenvalues of the training sample
#     # --------------------------------
#
#     eigen_vals = eigenvals_fun(training_sample)
#     index = np.argsort(eigen_vals)[::-1]
#     eigen_val_sort = eigen_vals[index]
#
#     # should the data be de-meaned prior to this or not?
#
#     # estimating factor number
#     # ------------------------
#
#     N, T = training_sample.shape
#
#     K = ER_fun((1 / (N * T)) * eigen_val_sort, k_max = 8)[1]
#     #K = GR_fun(training_sample, (1 / (N * T)) * eigen_val_sort, k_max=8)[1]
#
#     # computing sample covariance matrix
#     # ---------------------------------------
#
#     covariance_matrix = training_sample @ training_sample.T
#
#     # estimating population covariance matrix
#     # ---------------------------------------
#
#     C = 0.5
#
#     POET_K = poet_fun(covariance_matrix, N, T, K, C)
#
#     # estimating portfolio weights
#     # ---------------------------------------
#
#     N = covariance_matrix.shape[0]
#
#     ones_vector = np.ones([N, 1])
#
#     # divisor = ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector
#
#     portfolio_weights = (np.linalg.inv(POET_K) @ ones_vector) / (ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector)
#
#     sum_check = np.sum(portfolio_weights, 0)
#
#     # out of sample
#     # ***************
#
#     sum_of_individual_returns = np.sum(out_of_sample, axis=1)
#
#     portfolio_return = sum_of_individual_returns.T @ portfolio_weights
#
#     storage_vector[t] = portfolio_return
#
#
#
# plt.plot(storage_vector)
# plt.show()



# old old code below

############

#
# a = 2
#
# t = 0 # runs within a for loop
#
# whole_sample = daily_returns[(t * out_of_sample_width): (training_sample_width + out_of_sample_width) + (t * out_of_sample_width), :]
#
# training_sample = whole_sample[0:training_sample_width, :]
#
# out_of_sample = whole_sample[training_sample_width: (training_sample_width + out_of_sample_width), :]
#
# # transposing
# training_sample = training_sample.T
# out_of_sample = out_of_sample.T
#
# """
# rows: returns
# columns : dates
# """
#
# # flipping, so that the oldest values are on the left
# training_sample = np.flip(training_sample,axis = 1)
# out_of_sample = np.flip(out_of_sample,axis = 1)
#
#
#
# # training sample
# # ***************
#
#
#
# # computing eigenvalues of the training sample
# # --------------------------------
#
# eigen_vals = eigenvals_fun(training_sample)
# index = np.argsort(eigen_vals)[::-1]
# eigen_val_sort = eigen_vals[index]
#
# # should the data be de-meaned prior to this or not?
#
# # estimating factor number
# # ------------------------
#
# N, T = training_sample.shape
#
# K = GR_fun(training_sample, (1 / (N * T)) * eigen_val_sort, k_max = 8)[1]
#
#
#
# # computing sample covariance matrix
# # ---------------------------------------
#
# covariance_matrix = training_sample @ training_sample.T
#
# # estimating population covariance matrix
# # ---------------------------------------
#
#
# C = 0.5
#
# POET_K = poet_fun(covariance_matrix,N,T,K,C)
#
#
#
# # estimating portfolio weights
# # ---------------------------------------
#
# N = covariance_matrix.shape[0]
#
# ones_vector = np.ones([N,1])
#
# #divisor = ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector
#
# portfolio_weights = (np.linalg.inv(POET_K) @ ones_vector) / (ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector)
#
# sum_check = np.sum(portfolio_weights, 0)
#
#
#
#
# # out of sample
# # ***************
#
# sum_of_individual_returns = np.sum(out_of_sample,axis=1)
#
# portfolio_return = sum_of_individual_returns.T @ portfolio_weights















