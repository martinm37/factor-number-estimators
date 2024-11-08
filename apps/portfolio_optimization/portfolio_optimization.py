
"""
code for the portfolio optimization
"""

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

N = int(sys.argv[1])
years = int(sys.argv[2])
k_max = int(sys.argv[3])
k_min = int(sys.argv[4])
dataset = sys.argv[5] # orig or edit

# N = 150
# years = 14
# k_max = 8
# k_min = 0
# dataset = "orig"

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

