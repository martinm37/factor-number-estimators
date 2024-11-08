

import os
import sys
import time
import pandas as pd

from src.computation_functions.old_2.portfolio_optimizers import portfolio_optimizer_poet_er, \
    portfolio_optimizer_poet_bic3, \
    portfolio_optimizer_poet_gr, portfolio_optimizer_poet_ed, \
    portfolio_optimizer_poet_ic1, portfolio_optimizer_1_over_n

from src.utils.paths import get_data_path, get_results_path

# choose N dimension

N = int(sys.argv[1])
# N = 150

t0_optim = time.time()


# importing data
daily_returns_df = pd.read_csv(os.path.join(get_data_path(),f"SP500_{N}_daily_returns.csv"))

# converting to numpy
daily_returns = daily_returns_df.iloc[:,1:].to_numpy()

"""
rows: dates
columns : returns
"""

# slicing
training_sample_width = 250
out_of_sample_width = 21

# cross validation settings for c
C_min = 0.5
precision = 0.1



poet_er_returns, poet_er_k_estimates, er_c_estimates, er_m_estimates = portfolio_optimizer_poet_er(daily_returns,training_sample_width,out_of_sample_width, C_min = C_min,  precision = precision)
poet_gr_returns, poet_gr_k_estimates, gr_c_estimates, gr_m_estimates = portfolio_optimizer_poet_gr(daily_returns,training_sample_width,out_of_sample_width, C_min = C_min, precision = precision)
poet_ed_returns, poet_ed_k_estimates, ed_c_estimates, ed_m_estimates = portfolio_optimizer_poet_ed(daily_returns,training_sample_width,out_of_sample_width, C_min = C_min,  precision = precision)
poet_ic1_returns, poet_ic1_k_estimates, ic1_c_estimates, ic1_m_estimates = portfolio_optimizer_poet_ic1(daily_returns,training_sample_width,out_of_sample_width, C_min = C_min,  precision = precision)
poet_bic3_returns, poet_bic3_k_estimates, bic3_c_estimates, bic3_m_estimates = portfolio_optimizer_poet_bic3(daily_returns,training_sample_width,out_of_sample_width, C_min = C_min,  precision = precision)
one_over_n_returns = portfolio_optimizer_1_over_n(daily_returns,training_sample_width,out_of_sample_width)


dict_returns = {
    "ER": poet_er_returns,
    "GR": poet_gr_returns,
    "ED": poet_ed_returns,
    "IC1": poet_ic1_returns,
    "BIC3": poet_bic3_returns,
    "one_over_n": one_over_n_returns,
}

returns_df = pd.DataFrame(dict_returns)


dict_k_hat = {
    "ER": poet_er_k_estimates,
    "GR": poet_gr_k_estimates,
    "ED": poet_ed_k_estimates,
    "IC1": poet_ic1_k_estimates,
    "BIC3": poet_bic3_k_estimates,
}

k_hat_df = pd.DataFrame(dict_k_hat)

dict_c_star = {
    "ER": er_c_estimates,
    "GR": gr_c_estimates,
    "ED": ed_c_estimates,
    "IC1": ic1_c_estimates,
    "BIC3": bic3_c_estimates,

}

c_star_df = pd.DataFrame(dict_c_star)


dict_m_star = {
    "ER": er_m_estimates,
    "GR": gr_m_estimates,
    "ED": ed_m_estimates,
    "IC1": ic1_m_estimates,
    "BIC3": bic3_m_estimates,

}

m_star_df = pd.DataFrame(dict_m_star)


# inserting the time column again

time_cut = daily_returns_df.iloc[training_sample_width:,:]

time_col = time_cut["UTCTIME"]
time_col = time_col.reset_index()
time_col = time_col.drop("index",axis=1)

#hmm.shift(-training_sample_width)

returns_df.insert(0, "UTCTIME", time_col)



returns_df.to_csv(os.path.join(get_results_path(), f"portfolio_optimization_{N}_results.csv"))

k_hat_df.to_csv(os.path.join(get_results_path(), f"k_hat_{N}.csv"))

c_star_df.to_csv(os.path.join(get_results_path(), f"c_star_{N}.csv"))

m_star_df.to_csv(os.path.join(get_results_path(), f"m_star_{N}.csv"))


t1_optim = time.time()


print(f"total elapsed time: {t1_optim - t0_optim} sec")




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





print("hello there xdddd")












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















