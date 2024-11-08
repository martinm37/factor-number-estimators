

import time

import numpy as np

from src.computation_functions.aux_functions import eigenvals_fun
from src.computation_functions.old_3.m_calculation_poet import m_calculation
from src.computation_functions.old_3.multifold_cross_validation_poet import c_mcv_poet_fun
from src.estimators.covariance_matrix.old.poet_numba import poet_fun
from src.estimators.factor_number.ahn_horenstein import ER_fun, GR_fun
from src.estimators.factor_number.bai_ng import BIC3_fun, IC1_fun
from src.estimators.factor_number.onatski import ED_fun



# C_global = 2
M_global = 3

training_share_global = 0.5


def portfolio_optimizer_poet_er(daily_returns,training_sample_width,out_of_sample_width, C_min, precision):

    t0_optimizer = time.time()

    """
    daily_returns
    ------------
    rows: dates
    columns : returns
    """

    iter_num = int(np.floor((len(daily_returns) - training_sample_width) / out_of_sample_width))

    #storage_vector_return = np.zeros(iter_num)

    daily_returns_vec = np.zeros(len(daily_returns) - training_sample_width)

    k_estimate_vec = np.zeros(iter_num)

    M_estimate_vec = np.zeros(iter_num)

    c_estimate_vec = np.zeros(iter_num)

    iter_counter = 0

    for t in range(iter_num):

        iter_counter += 1

        if iter_counter == 10:
            print(f"iteration number: {t + 1}")
            iter_counter = 0

        t_start = time.time()

        whole_sample = daily_returns[(t * out_of_sample_width): (training_sample_width + out_of_sample_width) + (
                    t * out_of_sample_width), :]

        training_sample = whole_sample[0: training_sample_width, :]

        out_of_sample = whole_sample[training_sample_width: (training_sample_width + out_of_sample_width), :]

        # transposing
        training_sample = training_sample.T
        out_of_sample = out_of_sample.T

        """
        rows: returns
        columns : dates
        """

        # flipping, so that the oldest values are on the left
        training_sample = np.flip(training_sample, axis=1)
        out_of_sample = np.flip(out_of_sample, axis=1)

        """
        applying double demeaning
        """

        N_ts, T_ts = training_sample.shape

        T_mean = np.mean(training_sample,axis=0).reshape(1,-1) # time_mean -> row vector
        N_mean = np.mean(training_sample,axis=1).reshape(-1,1) # cross_sectional_mean -> column vector
        NT_mean = 1/(N_ts*T_ts) * np.sum(training_sample,axis = None) # mean across both dimensions

        T_mean_matrix = np.ones([N_ts, T_ts]) * T_mean
        N_mean_matrix = np.ones([N_ts, T_ts]) * N_mean
        NT_mean_matrix = np.ones([N_ts, T_ts]) * NT_mean


        training_sample = training_sample - T_mean_matrix - N_mean_matrix + NT_mean_matrix



        # # time de - meaning
        # training_sample = training_sample - np.mean(training_sample,axis=1).reshape(-1,1)

        # training sample
        # ***************

        # computing eigenvalues of the training sample
        # --------------------------------

        eigen_vals = eigenvals_fun(training_sample)
        index = np.argsort(eigen_vals)[::-1]
        eigen_val_sort = eigen_vals[index]

        # should the data be de-meaned prior to this or not?

        # estimating factor number
        # ------------------------

        N, T = training_sample.shape

        K = ER_fun(training_sample,eigen_val_sort, k_max=8)[0]

        k_estimate_vec[t] = K
        # K = GR_fun(training_sample, (1 / (N * T)) * eigen_val_sort, k_max=8)[1]

        # computing sample covariance matrix
        # ---------------------------------------

        covariance_matrix = training_sample @ training_sample.T

        # estimating population covariance matrix
        # ---------------------------------------

        # determining m

        M = m_calculation(covariance_matrix, N, T, K, C_min, precision)

        M_estimate_vec[t] = M

        # determining C by multifold cross validation

        C_star = c_mcv_poet_fun(training_sample, cross_val_fold_number= 5, k_hat = K, C_min_plus_eps= C_min, M = M, cross_val_precision= precision)

        c_estimate_vec[t] = C_star




        #C = C_global

        POET_K = poet_fun(covariance_matrix, N, T, K, C_star)

        # estimating portfolio weights
        # ---------------------------------------

        N = covariance_matrix.shape[0]

        ones_vector = np.ones([N, 1])

        # divisor = ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector

        portfolio_weights = (np.linalg.inv(POET_K) @ ones_vector) / (
                    ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector)

        tolerance = 5
        sum_check = np.round(np.sum(portfolio_weights, 0)[0], decimals=tolerance)

        assert sum_check == 1.0, f"The sum of weights is not one, it is {sum_check}!"

        # out of sample
        # ***************

        for i in range(out_of_sample_width):
            portfolio_return_day_i = out_of_sample[:,i].T @ portfolio_weights

            daily_returns_vec[(t * out_of_sample_width) + i] = portfolio_return_day_i



        # sum_of_individual_returns = np.sum(out_of_sample, axis=1)
        #
        # portfolio_return = sum_of_individual_returns.T @ portfolio_weights
        #
        # storage_vector_return[t] = portfolio_return

        t_end = time.time()

        print(f"iteration number: {t + 1},M: {np.round(M,2)},C_star: {C_star},"
              f"elapsed time: {np.round(t_end-t_start,2)} seconds")
        #print(f"elapsed time: {np.round(t_end-t_start,2)} seconds")



    t1_optimizer = time.time()

    print(f"poet_er total epalsed time: {t1_optimizer - t0_optimizer} seconds")

    return daily_returns_vec, k_estimate_vec, c_estimate_vec, M_estimate_vec



def portfolio_optimizer_poet_gr(daily_returns,training_sample_width,out_of_sample_width, C_min, precision):

    t0_optimizer = time.time()

    """
    daily_returns
    ------------
    rows: dates
    columns : returns
    """

    iter_num = int(np.floor((len(daily_returns) - training_sample_width) / out_of_sample_width))

    #storage_vector_return = np.zeros(iter_num)

    daily_returns_vec = np.zeros(len(daily_returns) - training_sample_width)

    k_estimate_vec = np.zeros(iter_num)

    M_estimate_vec = np.zeros(iter_num)

    c_estimate_vec = np.zeros(iter_num)

    iter_counter = 0

    for t in range(iter_num):

        iter_counter += 1

        if iter_counter == 10:
            print(f"iteration number: {t + 1}")
            iter_counter = 0

        whole_sample = daily_returns[(t * out_of_sample_width): (training_sample_width + out_of_sample_width) + (
                    t * out_of_sample_width), :]

        training_sample = whole_sample[0: training_sample_width, :]

        out_of_sample = whole_sample[training_sample_width: (training_sample_width + out_of_sample_width), :]

        # transposing
        training_sample = training_sample.T
        out_of_sample = out_of_sample.T

        """
        rows: returns
        columns : dates
        """

        # flipping, so that the oldest values are on the left
        training_sample = np.flip(training_sample, axis=1)
        out_of_sample = np.flip(out_of_sample, axis=1)

        """
        applying double demeaning
        """

        N_ts, T_ts = training_sample.shape

        T_mean = np.mean(training_sample,axis=0).reshape(1,-1) # time_mean -> row vector
        N_mean = np.mean(training_sample,axis=1).reshape(-1,1) # cross_sectional_mean -> column vector
        NT_mean = 1/(N_ts*T_ts) * np.sum(training_sample,axis = None) # mean across both dimensions

        T_mean_matrix = np.ones([N_ts, T_ts]) * T_mean
        N_mean_matrix = np.ones([N_ts, T_ts]) * N_mean
        NT_mean_matrix = np.ones([N_ts, T_ts]) * NT_mean


        training_sample = training_sample - T_mean_matrix - N_mean_matrix + NT_mean_matrix


        # # time de - meaning
        # training_sample = training_sample - np.mean(training_sample,axis=1).reshape(-1,1)

        # training sample
        # ***************

        # computing eigenvalues of the training sample
        # --------------------------------

        eigen_vals = eigenvals_fun(training_sample)
        index = np.argsort(eigen_vals)[::-1]
        eigen_val_sort = eigen_vals[index]

        # should the data be de-meaned prior to this or not?

        # estimating factor number
        # ------------------------

        N, T = training_sample.shape

        K = GR_fun(training_sample, (1 / (N * T)) * eigen_val_sort, k_max=8)[1]

        k_estimate_vec[t] = K
        # K = GR_fun(training_sample, (1 / (N * T)) * eigen_val_sort, k_max=8)[1]

        # computing sample covariance matrix
        # ---------------------------------------

        covariance_matrix = training_sample @ training_sample.T

        # estimating population covariance matrix
        # ---------------------------------------

        # determining m

        M = m_calculation(covariance_matrix, N, T, K, C_min, precision)

        M_estimate_vec[t] = M

        # determining C by multifold cross validation

        C_star = c_mcv_poet_fun(training_sample, training_share=training_share_global, k_hat=K, C_min_plus_eps= C_min, M = M, precision = precision)

        c_estimate_vec[t] = C_star

        print(f"iteration number: {t + 1},M: {M},C_star: {C_star}")

        # C = C_global

        POET_K = poet_fun(covariance_matrix, N, T, K, C_star)

        # estimating portfolio weights
        # ---------------------------------------

        N = covariance_matrix.shape[0]

        ones_vector = np.ones([N, 1])

        # divisor = ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector

        portfolio_weights = (np.linalg.inv(POET_K) @ ones_vector) / (
                    ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector)

        tolerance = 5
        sum_check = np.round(np.sum(portfolio_weights, 0)[0], decimals=tolerance)

        assert sum_check == 1.0, f"The sum of weights is not one, it is {sum_check}!"

        # out of sample
        # ***************

        for i in range(out_of_sample_width):
            portfolio_return_day_i = out_of_sample[:,i].T @ portfolio_weights

            daily_returns_vec[(t * out_of_sample_width) + i] = portfolio_return_day_i



        # sum_of_individual_returns = np.sum(out_of_sample, axis=1)
        #
        # portfolio_return = sum_of_individual_returns.T @ portfolio_weights
        #
        # storage_vector_return[t] = portfolio_return



    t1_optimizer = time.time()

    print(f"poet_gr total epalsed time: {t1_optimizer - t0_optimizer} seconds")

    return daily_returns_vec, k_estimate_vec, c_estimate_vec, M_estimate_vec


def portfolio_optimizer_poet_ed(daily_returns,training_sample_width,out_of_sample_width, C_min, precision):

    t0_optimizer = time.time()

    """
    daily_returns
    ------------
    rows: dates
    columns : returns
    """

    iter_num = int(np.floor((len(daily_returns) - training_sample_width) / out_of_sample_width))

    #storage_vector_return = np.zeros(iter_num)

    daily_returns_vec = np.zeros(len(daily_returns) - training_sample_width)

    k_estimate_vec = np.zeros(iter_num)

    M_estimate_vec = np.zeros(iter_num)

    c_estimate_vec = np.zeros(iter_num)

    iter_counter = 0

    for t in range(iter_num):

        iter_counter += 1

        if iter_counter == 10:
            print(f"iteration number: {t + 1}")
            iter_counter = 0

        whole_sample = daily_returns[(t * out_of_sample_width): (training_sample_width + out_of_sample_width) + (
                    t * out_of_sample_width), :]

        training_sample = whole_sample[0: training_sample_width, :]

        out_of_sample = whole_sample[training_sample_width: (training_sample_width + out_of_sample_width), :]

        # transposing
        training_sample = training_sample.T
        out_of_sample = out_of_sample.T

        """
        rows: returns
        columns : dates
        """

        # flipping, so that the oldest values are on the left
        training_sample = np.flip(training_sample, axis=1)
        out_of_sample = np.flip(out_of_sample, axis=1)

        """
        applying double demeaning - but he did not mention it in his paper !!!!
        """

        N_ts, T_ts = training_sample.shape

        T_mean = np.mean(training_sample, axis=0).reshape(1, -1)  # time_mean -> row vector
        N_mean = np.mean(training_sample, axis=1).reshape(-1, 1)  # cross_sectional_mean -> column vector
        NT_mean = 1 / (N_ts * T_ts) * np.sum(training_sample, axis=None)  # mean across both dimensions

        T_mean_matrix = np.ones([N_ts, T_ts]) * T_mean
        N_mean_matrix = np.ones([N_ts, T_ts]) * N_mean
        NT_mean_matrix = np.ones([N_ts, T_ts]) * NT_mean

        training_sample = training_sample - T_mean_matrix - N_mean_matrix + NT_mean_matrix

        # # time de - meaning
        # training_sample = training_sample - np.mean(training_sample,axis=1).reshape(-1,1)

        # training sample
        # ***************

        # computing eigenvalues of the training sample
        # --------------------------------

        eigen_vals = eigenvals_fun(training_sample)
        index = np.argsort(eigen_vals)[::-1]
        eigen_val_sort = eigen_vals[index]

        # should the data be de-meaned prior to this or not?

        # estimating factor number
        # ------------------------

        N, T = training_sample.shape

        K = ED_fun((1 / T) * eigen_val_sort, k_max= 8)[0]

        k_estimate_vec[t] = K
        # K = GR_fun(training_sample, (1 / (N * T)) * eigen_val_sort, k_max=8)[1]

        # computing sample covariance matrix
        # ---------------------------------------

        covariance_matrix = training_sample @ training_sample.T

        # estimating population covariance matrix
        # ---------------------------------------

        # determining m

        M = m_calculation(covariance_matrix, N, T, K, C_min, precision)

        M_estimate_vec[t] = M

        # determining C by multifold cross validation

        C_star = c_mcv_poet_fun(training_sample, training_share=training_share_global, k_hat=K, C_min_plus_eps= C_min, M = M, precision = precision)

        c_estimate_vec[t] = C_star

        print(f"iteration number: {t + 1},M: {M},C_star: {C_star}")

        # C = C_global

        POET_K = poet_fun(covariance_matrix, N, T, K, C_star)

        # estimating portfolio weights
        # ---------------------------------------

        N = covariance_matrix.shape[0]

        ones_vector = np.ones([N, 1])

        # divisor = ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector

        portfolio_weights = (np.linalg.inv(POET_K) @ ones_vector) / (
                    ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector)

        tolerance = 5
        sum_check = np.round(np.sum(portfolio_weights, 0)[0], decimals=tolerance)

        assert sum_check == 1.0, f"The sum of weights is not one, it is {sum_check}!"

        # out of sample
        # ***************

        for i in range(out_of_sample_width):
            portfolio_return_day_i = out_of_sample[:,i].T @ portfolio_weights

            daily_returns_vec[(t * out_of_sample_width) + i] = portfolio_return_day_i



        # sum_of_individual_returns = np.sum(out_of_sample, axis=1)
        #
        # portfolio_return = sum_of_individual_returns.T @ portfolio_weights
        #
        # storage_vector_return[t] = portfolio_return



    t1_optimizer = time.time()

    print(f"poet_ed total epalsed time: {t1_optimizer - t0_optimizer} seconds")

    return daily_returns_vec, k_estimate_vec, c_estimate_vec, M_estimate_vec



def portfolio_optimizer_poet_bic3(daily_returns,training_sample_width,out_of_sample_width, C_min,  precision):

    t0_optimizer = time.time()

    """
    daily_returns
    ------------
    rows: dates
    columns : returns
    """

    iter_num = int(np.floor((len(daily_returns) - training_sample_width) / out_of_sample_width))

    #storage_vector_return = np.zeros(iter_num)

    daily_returns_vec = np.zeros(len(daily_returns) - training_sample_width)

    k_estimate_vec = np.zeros(iter_num)

    M_estimate_vec = np.zeros(iter_num)

    c_estimate_vec = np.zeros(iter_num)

    iter_counter = 0

    for t in range(iter_num):

        iter_counter += 1

        if iter_counter == 10:
            print(f"iteration number: {t + 1}")
            iter_counter = 0

        whole_sample = daily_returns[(t * out_of_sample_width): (training_sample_width + out_of_sample_width) + (
                    t * out_of_sample_width), :]

        training_sample = whole_sample[0: training_sample_width, :]

        out_of_sample = whole_sample[training_sample_width: (training_sample_width + out_of_sample_width), :]

        # transposing
        training_sample = training_sample.T
        out_of_sample = out_of_sample.T

        """
        rows: returns
        columns : dates
        """

        # flipping, so that the oldest values are on the left
        training_sample = np.flip(training_sample, axis=1)
        out_of_sample = np.flip(out_of_sample, axis=1)

        # time de - meaning
        training_sample = training_sample - np.mean(training_sample,axis=1).reshape(-1,1)

        # training sample
        # ***************

        # computing eigenvalues of the training sample
        # --------------------------------

        eigen_vals = eigenvals_fun(training_sample)
        index = np.argsort(eigen_vals)[::-1]
        eigen_val_sort = eigen_vals[index]

        # should the data be de-meaned prior to this or not?

        # estimating factor number
        # ------------------------

        N, T = training_sample.shape

        K = BIC3_fun(training_sample, eigen_val_sort, k_max = 8)[1]

        k_estimate_vec[t] = K
        # K = GR_fun(training_sample, (1 / (N * T)) * eigen_val_sort, k_max=8)[1]

        # computing sample covariance matrix
        # ---------------------------------------

        covariance_matrix = training_sample @ training_sample.T

        # estimating population covariance matrix
        # ---------------------------------------

        # determining m

        M = m_calculation(covariance_matrix, N, T, K, C_min, precision)

        M_estimate_vec[t] = M

        # determining C by multifold cross validation

        C_star = c_mcv_poet_fun(training_sample, training_share=training_share_global, k_hat=K, C_min_plus_eps= C_min, M = M, precision = precision)

        c_estimate_vec[t] = C_star

        print(f"iteration number: {t + 1},M: {M},C_star: {C_star}")

        # C = C_global

        POET_K = poet_fun(covariance_matrix, N, T, K, C_star)

        # estimating portfolio weights
        # ---------------------------------------

        N = covariance_matrix.shape[0]

        ones_vector = np.ones([N, 1])

        # divisor = ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector

        portfolio_weights = (np.linalg.inv(POET_K) @ ones_vector) / (
                    ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector)

        tolerance = 5
        sum_check = np.round(np.sum(portfolio_weights, 0)[0], decimals=tolerance)

        assert sum_check == 1.0, f"The sum of weights is not one, it is {sum_check}!"

        # out of sample
        # ***************

        for i in range(out_of_sample_width):
            portfolio_return_day_i = out_of_sample[:,i].T @ portfolio_weights

            daily_returns_vec[(t * out_of_sample_width) + i] = portfolio_return_day_i



        # sum_of_individual_returns = np.sum(out_of_sample, axis=1)
        #
        # portfolio_return = sum_of_individual_returns.T @ portfolio_weights
        #
        # storage_vector_return[t] = portfolio_return




    t1_optimizer = time.time()

    print(f"poet_bic3 total epalsed time: {t1_optimizer - t0_optimizer} seconds")

    return daily_returns_vec, k_estimate_vec, c_estimate_vec, M_estimate_vec



def portfolio_optimizer_poet_ic1(daily_returns,training_sample_width,out_of_sample_width, C_min,  precision):

    t0_optimizer = time.time()

    """
    daily_returns
    ------------
    rows: dates
    columns : returns
    """

    iter_num = int(np.floor((len(daily_returns) - training_sample_width) / out_of_sample_width))

    #storage_vector_return = np.zeros(iter_num)

    daily_returns_vec = np.zeros(len(daily_returns) - training_sample_width)

    k_estimate_vec = np.zeros(iter_num)

    M_estimate_vec = np.zeros(iter_num)

    c_estimate_vec = np.zeros(iter_num)

    iter_counter = 0

    for t in range(iter_num):

        iter_counter += 1

        if iter_counter == 10:
            print(f"iteration number: {t + 1}")
            iter_counter = 0

        whole_sample = daily_returns[(t * out_of_sample_width): (training_sample_width + out_of_sample_width) + (
                    t * out_of_sample_width), :]

        training_sample = whole_sample[0: training_sample_width, :]

        out_of_sample = whole_sample[training_sample_width: (training_sample_width + out_of_sample_width), :]

        # transposing
        training_sample = training_sample.T
        out_of_sample = out_of_sample.T

        """
        rows: returns
        columns : dates
        """

        # flipping, so that the oldest values are on the left
        training_sample = np.flip(training_sample, axis=1)
        out_of_sample = np.flip(out_of_sample, axis=1)

        # time de - meaning
        training_sample = training_sample - np.mean(training_sample,axis=1).reshape(-1,1)

        # training sample
        # ***************

        # computing eigenvalues of the training sample
        # --------------------------------

        eigen_vals = eigenvals_fun(training_sample)
        index = np.argsort(eigen_vals)[::-1]
        eigen_val_sort = eigen_vals[index]

        # should the data be de-meaned prior to this or not?

        # estimating factor number
        # ------------------------

        N, T = training_sample.shape

        K = IC1_fun(training_sample, eigen_val_sort, k_max=8)[1]

        k_estimate_vec[t] = K
        # K = GR_fun(training_sample, (1 / (N * T)) * eigen_val_sort, k_max=8)[1]

        # computing sample covariance matrix
        # ---------------------------------------

        covariance_matrix = training_sample @ training_sample.T

        # estimating population covariance matrix
        # ---------------------------------------

        # determining m

        M = m_calculation(covariance_matrix, N, T, K, C_min, precision)

        M_estimate_vec[t] = M

        # determing c star

        C_star = c_mcv_poet_fun(training_sample, training_share=training_share_global, k_hat=K, C_min_plus_eps= C_min, M = M, precision = precision)

        c_estimate_vec[t] = C_star

        print(f"iteration number: {t + 1},M: {M},C_star: {C_star}")

        # C = C_global

        POET_K = poet_fun(covariance_matrix, N, T, K, C_star)

        # estimating portfolio weights
        # ---------------------------------------

        N = covariance_matrix.shape[0]

        ones_vector = np.ones([N, 1])

        # divisor = ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector

        portfolio_weights = (np.linalg.inv(POET_K) @ ones_vector) / (
                    ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector)

        tolerance = 5
        sum_check = np.round(np.sum(portfolio_weights, 0)[0], decimals=tolerance)

        assert sum_check == 1.0, f"The sum of weights is not one, it is {sum_check}!"

        # out of sample
        # ***************

        for i in range(out_of_sample_width):
            portfolio_return_day_i = out_of_sample[:,i].T @ portfolio_weights

            daily_returns_vec[(t * out_of_sample_width) + i] = portfolio_return_day_i



        # sum_of_individual_returns = np.sum(out_of_sample, axis=1)
        #
        # portfolio_return = sum_of_individual_returns.T @ portfolio_weights
        #
        # storage_vector_return[t] = portfolio_return




    t1_optimizer = time.time()

    print(f"poet_ic1 total epalsed time: {t1_optimizer - t0_optimizer} seconds")

    return daily_returns_vec, k_estimate_vec, c_estimate_vec, M_estimate_vec




def portfolio_optimizer_sample_covariance(daily_returns,training_sample_width,out_of_sample_width):

    t0_optimizer = time.time()

    """
    daily_returns
    ------------
    rows: dates
    columns : returns
    """

    iter_num = int(np.floor((len(daily_returns) - training_sample_width) / out_of_sample_width))

    #storage_vector_return = np.zeros(iter_num)

    daily_returns_vec = np.zeros(len(daily_returns) - training_sample_width)

    iter_counter = 0

    for t in range(iter_num):

        iter_counter += 1

        if iter_counter == 10:
            print(f"iteration number: {t + 1}")
            iter_counter = 0

        whole_sample = daily_returns[(t * out_of_sample_width): (training_sample_width + out_of_sample_width) + (
                    t * out_of_sample_width), :]

        training_sample = whole_sample[0: training_sample_width, :]

        out_of_sample = whole_sample[training_sample_width: (training_sample_width + out_of_sample_width), :]

        # transposing
        training_sample = training_sample.T
        out_of_sample = out_of_sample.T

        """
        rows: returns
        columns : dates
        """

        # flipping, so that the oldest values are on the left
        training_sample = np.flip(training_sample, axis=1)
        out_of_sample = np.flip(out_of_sample, axis=1)

        # time de - meaning
        training_sample = training_sample - np.mean(training_sample,axis=1).reshape(-1,1)

        # training sample
        # ***************

        # computing sample covariance matrix
        # ---------------------------------------

        covariance_matrix = training_sample @ training_sample.T


        # estimating portfolio weights
        # ---------------------------------------

        N = covariance_matrix.shape[0]

        ones_vector = np.ones([N, 1])

        # divisor = ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector

        portfolio_weights = (np.linalg.inv(covariance_matrix) @ ones_vector) / (
                    ones_vector.T @ np.linalg.inv(covariance_matrix) @ ones_vector)

        tolerance = 5
        sum_check = np.round(np.sum(portfolio_weights, 0)[0], decimals=tolerance)

        assert sum_check == 1.0, f"The sum of weights is not one, it is {sum_check}!"

        # out of sample
        # ***************

        for i in range(out_of_sample_width):
            portfolio_return_day_i = out_of_sample[:,i].T @ portfolio_weights

            daily_returns_vec[(t * out_of_sample_width) + i] = portfolio_return_day_i




    t1_optimizer = time.time()

    print(f"sample_covariance total epalsed time: {t1_optimizer - t0_optimizer} seconds")

    return daily_returns_vec






def portfolio_optimizer_1_over_n(daily_returns,training_sample_width,out_of_sample_width):

    t0_optimizer = time.time()

    """
    daily_returns
    ------------
    rows: dates
    columns : returns
    """

    iter_num = int(np.floor((len(daily_returns) - training_sample_width) / out_of_sample_width))

    #storage_vector_return = np.zeros(iter_num)

    daily_returns_vec = np.zeros(len(daily_returns) - training_sample_width)

    iter_counter = 0

    for t in range(iter_num):

        iter_counter += 1

        if iter_counter == 10:
            print(f"iteration number: {t + 1}")
            iter_counter = 0

        whole_sample = daily_returns[(t * out_of_sample_width): (training_sample_width + out_of_sample_width) + (
                    t * out_of_sample_width), :]

        training_sample = whole_sample[0: training_sample_width, :]

        out_of_sample = whole_sample[training_sample_width: (training_sample_width + out_of_sample_width), :]

        # transposing
        training_sample = training_sample.T
        out_of_sample = out_of_sample.T

        """
        rows: returns
        columns : dates
        """

        # flipping, so that the oldest values are on the left
        training_sample = np.flip(training_sample, axis=1)
        out_of_sample = np.flip(out_of_sample, axis=1)

        # time de - meaning
        training_sample = training_sample - np.mean(training_sample,axis=1).reshape(-1,1)

        # training sample
        # ***************

        # computing sample covariance matrix
        # ---------------------------------------

        covariance_matrix = training_sample @ training_sample.T


        # estimating portfolio weights
        # ---------------------------------------

        N = covariance_matrix.shape[0]

        ones_vector = np.ones([N, 1])

        portfolio_weights = ones_vector * (1/N)

        tolerance = 5
        sum_check = np.round(np.sum(portfolio_weights, 0)[0],decimals =  tolerance )

        assert sum_check == 1.0, f"The sum of weights is not one, it is {sum_check}!"



        # out of sample
        # ***************

        for i in range(out_of_sample_width):
            portfolio_return_day_i = out_of_sample[:,i].T @ portfolio_weights

            daily_returns_vec[(t * out_of_sample_width) + i] = portfolio_return_day_i




    t1_optimizer = time.time()

    print(f"1/N portfolio total epalsed time: {t1_optimizer - t0_optimizer} seconds")

    return daily_returns_vec







#
# def portfolio_optimizer_poet_er(daily_returns,training_sample_width,out_of_sample_width):
#
#     t0_optimizer = time.time()
#
#     """
#     daily_returns
#     ------------
#     rows: dates
#     columns : returns
#     """
#
#     iter_num = int(np.floor((len(daily_returns) - training_sample_width) / out_of_sample_width))
#
#     storage_vector = np.zeros(iter_num)
#
#     storage_vector_k = np.zeros(iter_num)
#
#     iter_counter = 0
#
#     for t in range(iter_num):
#
#         iter_counter += 1
#
#         if iter_counter == 10:
#             print(f"iteration number: {t + 1}")
#             iter_counter = 0
#
#         whole_sample = daily_returns[(t * out_of_sample_width): (training_sample_width + out_of_sample_width) + (
#                     t * out_of_sample_width), :]
#
#         training_sample = whole_sample[0: training_sample_width, :]
#
#         out_of_sample = whole_sample[training_sample_width: (training_sample_width + out_of_sample_width), :]
#
#         # transposing
#         training_sample = training_sample.T
#         out_of_sample = out_of_sample.T
#
#         """
#         rows: returns
#         columns : dates
#         """
#
#         # flipping, so that the oldest values are on the left
#         training_sample = np.flip(training_sample, axis=1)
#         out_of_sample = np.flip(out_of_sample, axis=1)
#
#         # time de - meaning
#         training_sample = training_sample - np.mean(training_sample,axis=1).reshape(-1,1)
#
#         # training sample
#         # ***************
#
#         # computing eigenvalues of the training sample
#         # --------------------------------
#
#         eigen_vals = eigenvals_fun(training_sample)
#         index = np.argsort(eigen_vals)[::-1]
#         eigen_val_sort = eigen_vals[index]
#
#         # should the data be de-meaned prior to this or not?
#
#         # estimating factor number
#         # ------------------------
#
#         N, T = training_sample.shape
#
#         K = ER_fun((1 / (N * T)) * eigen_val_sort, k_max=8)[1]
#         # K = GR_fun(training_sample, (1 / (N * T)) * eigen_val_sort, k_max=8)[1]
#
#         # computing sample covariance matrix
#         # ---------------------------------------
#
#         covariance_matrix = training_sample @ training_sample.T
#
#         # estimating population covariance matrix
#         # ---------------------------------------
#
#         C = 0.5
#
#         POET_K = poet_fun(covariance_matrix, N, T, K, C)
#
#         # estimating portfolio weights
#         # ---------------------------------------
#
#         N = covariance_matrix.shape[0]
#
#         ones_vector = np.ones([N, 1])
#
#         # divisor = ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector
#
#         portfolio_weights = (np.linalg.inv(POET_K) @ ones_vector) / (
#                     ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector)
#
#         sum_check = np.sum(portfolio_weights, 0)
#
#         # out of sample
#         # ***************
#
#         sum_of_individual_returns = np.sum(out_of_sample, axis=1)
#
#         portfolio_return = sum_of_individual_returns.T @ portfolio_weights
#
#         storage_vector[t] = portfolio_return
#
#         storage_vector_k[t] = K
#
#     t1_optimizer = time.time()
#
#     print(f"poet_gr total epalsed time: {t1_optimizer - t0_optimizer} seconds")
#
#     return storage_vector, storage_vector_k
#
#
#
# def portfolio_optimizer_poet_gr(daily_returns,training_sample_width,out_of_sample_width):
#
#
#     """
#     daily_returns
#     ------------
#     rows: dates
#     columns : returns
#     """
#
#     t0_optimizer = time.time()
#
#     iter_num = int(np.floor((len(daily_returns) - training_sample_width) / out_of_sample_width))
#
#     storage_vector = np.zeros(iter_num)
#
#     storage_vector_k = np.zeros(iter_num)
#
#     iter_counter = 0
#
#     for t in range(iter_num):
#
#         iter_counter += 1
#
#         if iter_counter == 10:
#             print(f"iteration number: {t + 1}")
#             iter_counter = 0
#
#         whole_sample = daily_returns[(t * out_of_sample_width): (training_sample_width + out_of_sample_width) + (
#                     t * out_of_sample_width), :]
#
#         training_sample = whole_sample[0: training_sample_width, :]
#
#         out_of_sample = whole_sample[training_sample_width: (training_sample_width + out_of_sample_width), :]
#
#         # transposing
#         training_sample = training_sample.T
#         out_of_sample = out_of_sample.T
#
#         """
#         rows: returns
#         columns : dates
#         """
#
#         # flipping, so that the oldest values are on the left
#         training_sample = np.flip(training_sample, axis=1)
#         out_of_sample = np.flip(out_of_sample, axis=1)
#
#         # time de - meaning
#         training_sample = training_sample - np.mean(training_sample,axis=1).reshape(-1,1)
#
#         # training sample
#         # ***************
#
#         # computing eigenvalues of the training sample
#         # --------------------------------
#
#         eigen_vals = eigenvals_fun(training_sample)
#         index = np.argsort(eigen_vals)[::-1]
#         eigen_val_sort = eigen_vals[index]
#
#         # should the data be de-meaned prior to this or not?
#
#         # estimating factor number
#         # ------------------------
#
#         N, T = training_sample.shape
#
#         K = GR_fun(training_sample, (1 / (N * T)) * eigen_val_sort, k_max=8)[1]
#
#
#         # computing sample covariance matrix
#         # ---------------------------------------
#
#         covariance_matrix = training_sample @ training_sample.T
#
#         # estimating population covariance matrix
#         # ---------------------------------------
#
#         C = 0.5
#
#         POET_K = poet_fun(covariance_matrix, N, T, K, C)
#
#         # estimating portfolio weights
#         # ---------------------------------------
#
#         N = covariance_matrix.shape[0]
#
#         ones_vector = np.ones([N, 1])
#
#         # divisor = ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector
#
#         portfolio_weights = (np.linalg.inv(POET_K) @ ones_vector) / (
#                     ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector)
#
#         sum_check = np.sum(portfolio_weights, 0)
#
#         # out of sample
#         # ***************
#
#         sum_of_individual_returns = np.sum(out_of_sample, axis=1)
#
#         portfolio_return = sum_of_individual_returns.T @ portfolio_weights
#
#         storage_vector[t] = portfolio_return
#
#         storage_vector_k[t] = K
#
#     t1_optimizer = time.time()
#
#     print(f"poet_gr total epalsed time: {t1_optimizer - t0_optimizer} seconds")
#
#     return storage_vector, storage_vector_k
#
#
#
# def portfolio_optimizer_poet_ed(daily_returns,training_sample_width,out_of_sample_width):
#
#
#     """
#     daily_returns
#     ------------
#     rows: dates
#     columns : returns
#     """
#
#     t0_optimizer = time.time()
#
#     iter_num = int(np.floor((len(daily_returns) - training_sample_width) / out_of_sample_width))
#
#     storage_vector = np.zeros(iter_num)
#
#     storage_vector_k = np.zeros(iter_num)
#
#     iter_counter = 0
#
#     for t in range(iter_num):
#
#         iter_counter += 1
#
#         if iter_counter == 10:
#             print(f"iteration number: {t + 1}")
#             iter_counter = 0
#
#         whole_sample = daily_returns[(t * out_of_sample_width): (training_sample_width + out_of_sample_width) + (
#                     t * out_of_sample_width), :]
#
#         training_sample = whole_sample[0: training_sample_width, :]
#
#         out_of_sample = whole_sample[training_sample_width: (training_sample_width + out_of_sample_width), :]
#
#         # transposing
#         training_sample = training_sample.T
#         out_of_sample = out_of_sample.T
#
#         """
#         rows: returns
#         columns : dates
#         """
#
#         # flipping, so that the oldest values are on the left
#         training_sample = np.flip(training_sample, axis=1)
#         out_of_sample = np.flip(out_of_sample, axis=1)
#
#         # time de - meaning
#         training_sample = training_sample - np.mean(training_sample,axis=1).reshape(-1,1)
#
#         # training sample
#         # ***************
#
#         # computing eigenvalues of the training sample
#         # --------------------------------
#
#         eigen_vals = eigenvals_fun(training_sample)
#         index = np.argsort(eigen_vals)[::-1]
#         eigen_val_sort = eigen_vals[index]
#
#         # should the data be de-meaned prior to this or not?
#
#         # estimating factor number
#         # ------------------------
#
#         N, T = training_sample.shape
#
#         #K = GR_fun(training_sample, (1 / (N * T)) * eigen_val_sort, k_max=8)[1]
#         K = ED_fun((1 / T) * eigen_val_sort, r_max=8)[0]
#
#
#         # computing sample covariance matrix
#         # ---------------------------------------
#
#         covariance_matrix = training_sample @ training_sample.T
#
#         # estimating population covariance matrix
#         # ---------------------------------------
#
#         C = 0.5
#
#         POET_K = poet_fun(covariance_matrix, N, T, K, C)
#
#         # estimating portfolio weights
#         # ---------------------------------------
#
#         N = covariance_matrix.shape[0]
#
#         ones_vector = np.ones([N, 1])
#
#         # divisor = ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector
#
#         portfolio_weights = (np.linalg.inv(POET_K) @ ones_vector) / (
#                     ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector)
#
#         sum_check = np.sum(portfolio_weights, 0)
#
#         # out of sample
#         # ***************
#
#         sum_of_individual_returns = np.sum(out_of_sample, axis=1)
#
#         portfolio_return = sum_of_individual_returns.T @ portfolio_weights
#
#         storage_vector[t] = portfolio_return
#
#         storage_vector_k[t] = K
#
#     t1_optimizer = time.time()
#
#     print(f"poet_gr total epalsed time: {t1_optimizer - t0_optimizer} seconds")
#
#     return storage_vector, storage_vector_k
#
#
#
# def portfolio_optimizer_poet_bic3(daily_returns,training_sample_width,out_of_sample_width):
#
#
#     """
#     daily_returns
#     ------------
#     rows: dates
#     columns : returns
#     """
#
#     t0_optimizer = time.time()
#
#     iter_num = int(np.floor((len(daily_returns) - training_sample_width) / out_of_sample_width))
#
#     storage_vector = np.zeros(iter_num)
#
#     storage_vector_k = np.zeros(iter_num)
#
#     iter_counter = 0
#
#     for t in range(iter_num):
#
#         iter_counter += 1
#
#         if iter_counter == 10:
#             print(f"iteration number: {t + 1}")
#             iter_counter = 0
#
#         whole_sample = daily_returns[(t * out_of_sample_width): (training_sample_width + out_of_sample_width) + (
#                     t * out_of_sample_width), :]
#
#         training_sample = whole_sample[0: training_sample_width, :]
#
#         out_of_sample = whole_sample[training_sample_width: (training_sample_width + out_of_sample_width), :]
#
#         # transposing
#         training_sample = training_sample.T
#         out_of_sample = out_of_sample.T
#
#         """
#         rows: returns
#         columns : dates
#         """
#
#         # flipping, so that the oldest values are on the left
#         training_sample = np.flip(training_sample, axis=1)
#         out_of_sample = np.flip(out_of_sample, axis=1)
#
#         # time de - meaning
#         training_sample = training_sample - np.mean(training_sample,axis=1).reshape(-1,1)
#
#         # training sample
#         # ***************
#
#         # computing eigenvalues of the training sample
#         # --------------------------------
#
#         eigen_vals = eigenvals_fun(training_sample)
#         index = np.argsort(eigen_vals)[::-1]
#         eigen_val_sort = eigen_vals[index]
#
#         # should the data be de-meaned prior to this or not?
#
#         # estimating factor number
#         # ------------------------
#
#         N, T = training_sample.shape
#
#         #K = GR_fun(training_sample, (1 / (N * T)) * eigen_val_sort, k_max=8)[1]
#         K = BIC3_fun(training_sample, eigen_val_sort, k_max=8)[1]
#
#
#         # computing sample covariance matrix
#         # ---------------------------------------
#
#         covariance_matrix = training_sample @ training_sample.T
#
#         # estimating population covariance matrix
#         # ---------------------------------------
#
#         C = 0.5
#
#         POET_K = poet_fun(covariance_matrix, N, T, K, C)
#
#         # estimating portfolio weights
#         # ---------------------------------------
#
#         N = covariance_matrix.shape[0]
#
#         ones_vector = np.ones([N, 1])
#
#         # divisor = ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector
#
#         portfolio_weights = (np.linalg.inv(POET_K) @ ones_vector) / (
#                     ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector)
#
#         sum_check = np.sum(portfolio_weights, 0)
#
#         # out of sample
#         # ***************
#
#         sum_of_individual_returns = np.sum(out_of_sample, axis=1)
#
#         portfolio_return = sum_of_individual_returns.T @ portfolio_weights
#
#         storage_vector[t] = portfolio_return
#
#         storage_vector_k[t] = K
#
#     t1_optimizer = time.time()
#
#     print(f"poet_gr total epalsed time: {t1_optimizer - t0_optimizer} seconds")
#
#     return storage_vector, storage_vector_k
#


