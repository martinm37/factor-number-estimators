
import os
import time
import numpy as np

from src.computation_functions.eigenvalues_computation import eigenvals_fun
from src.computation_functions.old_3.c_min_calculation import c_min_calculation
from src.computation_functions.old_3.m_calculation_poet import m_calculation
from src.computation_functions.old_3.multifold_cross_validation_poet import c_mcv_poet_fun
from src.estimators.covariance_matrix.old.poet_pca_numba import poet_fun
from src.estimators.factor_number.ahn_horenstein import ER_fun, GR_fun
from src.estimators.factor_number.bai_ng import IC1_fun, BIC3_fun
from src.estimators.factor_number.onatski import ED_fun
from src.estimators.factor_number.wei_chen_numba import TKCV_fun
from src.models.portfolio_optim_configs import PortfolioOptimConfig, POETConfig
from src.models.portfolio_optim_solutions import POETSolution, BasicSolverSolution
from src.utils.computation_utils import double_demeaning, time_demeaning, time_demeaning_unit_variance
from src.utils.paths import get_results_path


class POETPortfolioOptimizer:

    def __init__(self, portfolio_optim_config: PortfolioOptimConfig, poet_config: POETConfig):

        self.p_o_config = portfolio_optim_config
        self.poet_config = poet_config


    def eigenvalue_based_k_estimator(self, daily_returns, estimator, k_max) -> POETSolution:

        t0_optimizer = time.time()

        print(f"k_hat estimator: {estimator}")

        with open(os.path.join(get_results_path(), "portfolio_optim_log.txt"), "a") as text_file:
            print(f"k_hat estimator: {estimator}",file=text_file)

        """
        daily_returns
        ------------
        rows: dates
        columns : returns
        """

        iter_num = int(np.floor((len(daily_returns) -
                                 self.p_o_config.training_sample_width) / self.p_o_config.out_of_sample_width))

        # here we always do analysis on the same size of out_of_sample_width = 21
        # therefore we need to floor it to the closest integer

        # preallocating vectors for storage
        daily_returns_vec = np.zeros(len(daily_returns) - self.p_o_config.training_sample_width)

        k_estimate_vec = np.zeros(iter_num)
        C_min_estimate_vec = np.zeros(iter_num)
        M_estimate_vec = np.zeros(iter_num)
        C_star_vec = np.zeros(iter_num)

        iter_counter = 0

        for t in range(iter_num):

            iter_counter += 1

            # if iter_counter == 10:
            #     print(f"iteration number: {t + 1}")
            #     iter_counter = 0

            t_start = time.time()

            whole_sample = daily_returns[(t * self.p_o_config.out_of_sample_width):
                                         (self.p_o_config.training_sample_width +
                                          self.p_o_config.out_of_sample_width) +
                                         (t * self.p_o_config.out_of_sample_width), :]

            training_sample = whole_sample[0: self.p_o_config.training_sample_width, :]

            out_of_sample = whole_sample[self.p_o_config.training_sample_width:
                                         (self.p_o_config.training_sample_width +
                                          self.p_o_config.out_of_sample_width), :]

            # transposing
            training_sample = training_sample.T
            out_of_sample = out_of_sample.T

            """
            rows: returns
            columns : dates
            """

            # flipping, so that the oldest values are on the left
            # ---------------------------------------------------
            training_sample = np.flip(training_sample, axis=1)
            out_of_sample = np.flip(out_of_sample, axis=1)


            # applying de-meaning on the sample data for the estimator
            # -------------------------------------------------------

            training_sample_k_hat = training_sample.copy()


            de_meaning_functions = {
                "IC1": time_demeaning_unit_variance,
                "BIC3": time_demeaning_unit_variance,
                "ED": double_demeaning,
                "ER": double_demeaning,
                "GR": double_demeaning
            }

            training_sample_k_hat = de_meaning_functions[estimator](training_sample_k_hat)



            # computing eigenvalues of the demeaned training sample
            # -----------------------------------------------------

            eigen_vals = eigenvals_fun(training_sample_k_hat)
            index = np.argsort(eigen_vals)[::-1]
            eigen_vals_sort = eigen_vals[index]


            # estimating factor number - eigenvalue based
            # -------------------------------------------

            eigenvalue_based_k_estimator_dictionary = {
                "ER": ER_fun,
                "GR": GR_fun,
                "ED": ED_fun,
                "IC1": IC1_fun,
                "BIC3": BIC3_fun
            }

            k_hat = eigenvalue_based_k_estimator_dictionary[estimator](training_sample_k_hat, eigen_vals_sort, k_max = k_max)[0]

            #k_hat = max(k_hat,1) # estimated factor number is always at least 1

            k_estimate_vec[t] = k_hat

            # time demeaning the training sample
            # ( not the out of sample, there the time means are important !!! )
            # ------------------------------------------------
            training_sample = time_demeaning(training_sample)

            # computing sample covariance matrix - for the POET estimator
            # ----------------------------------------------------------

            covariance_matrix = training_sample @ training_sample.T

            # POET: estimating population covariance matrix
            # ---------------------------------------

            N, T = training_sample.shape

            # estimating M_hat - upper bound of C_star

            M_hat = m_calculation(covariance_matrix, N, T, k_hat, C_min = 0.0,
                                  C_grid_precision = self.poet_config.C_grid_precision)

            M_estimate_vec[t] = M_hat

            # estimating C_min - lower bound of C_star

            C_min = c_min_calculation(covariance_matrix, N, T, k_hat,
                      C_min_start = 0.0, C_min_end = M_hat, C_grid_precision = self.poet_config.C_grid_precision)

            C_min_estimate_vec[t] = C_min


            # determining C_star by multifold cross validation, along the determined set [C_min_plus_eps + eps, M_hat]

            epsilon = 0.1

            C_star = c_mcv_poet_fun(training_sample,
                                    cross_val_fold_number=self.poet_config.cross_validation_fold_number,
                                    k_hat=k_hat, C_min_plus_eps = C_min + epsilon, M = M_hat,
                                    cross_val_precision= self.poet_config.cross_validation_precision)

            C_star_vec[t] = C_star

            # C_star = 0.5

            POET_K = poet_fun(covariance_matrix, N, T, k_hat, C_star)

            # estimating portfolio weights
            # ---------------------------------------

            ones_vector = np.ones([N, 1])

            # divisor = ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector

            portfolio_weights = ( (np.linalg.inv(POET_K) @ ones_vector) /
                                 (ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector) )

            tolerance = 5
            sum_check = np.round(np.sum(portfolio_weights, 0)[0], decimals=tolerance)

            assert sum_check == 1.0, f"The sum of weights is not one, it is {sum_check}!"

            # out of sample
            # ***************

            """
            redo this as a matrix vector multiplication
            """

            for i in range(self.p_o_config.out_of_sample_width):
                portfolio_return_day_i = out_of_sample[:, i].T @ portfolio_weights

                daily_returns_vec[(t * self.p_o_config.out_of_sample_width) + i] = portfolio_return_day_i

            # sum_of_individual_returns = np.sum(out_of_sample, axis=1)
            # portfolio_return = sum_of_individual_returns.T @ portfolio_weights
            # storage_vector_return[t] = portfolio_return

            # C_min_plus_eps = 0
            # M_hat = 0

            t_end = time.time()

            print(f"iter_num: {t + 1}, C_min: {C_min}, "
                  f"M_hat: {np.round(M_hat, 2)}, C_star: {C_star}, "
                  f"elapsed time: {np.round(t_end - t_start, 2)} seconds")

            with open(os.path.join(get_results_path(), "portfolio_optim_log.txt"), "a") as text_file:
                print(f"iter_num: {t + 1}, C_min: {C_min} "
                      f"M_hat: {np.round(M_hat, 2)}, C_star: {C_star}, "
                      f"elapsed time: {np.round(t_end - t_start, 2)} seconds", file = text_file)

        t1_optimizer = time.time()

        print(f"POET with k_hat estimator {estimator}: total epalsed time: "
              f" {round(t1_optimizer - t0_optimizer, 2)} sec,"
              f" {round((t1_optimizer - t0_optimizer) / 60, 2)} min,"
              f" {round((t1_optimizer - t0_optimizer) / (60 * 60), 2)} hour")

        with open(os.path.join(get_results_path(), "portfolio_optim_log.txt"), "a") as text_file:
            print(f"POET with k_hat estimator {estimator}: total epalsed time: "
                  f" {round(t1_optimizer - t0_optimizer, 2)} sec,"
                  f" {round((t1_optimizer - t0_optimizer) / 60, 2)} min,"
                  f" {round((t1_optimizer - t0_optimizer) / (60 * 60), 2)} hour",file=text_file)


        solution = POETSolution(daily_returns_vec = daily_returns_vec,
                                k_estimate_vec = k_estimate_vec,
                                C_min_estimate_vec=C_min_estimate_vec,
                                M_estimate_vec = M_estimate_vec,
                                C_star_vec= C_star_vec)

        return solution


    def cross_validation_based_k_estimator(self, daily_returns, estimator, fold_number, k_max) -> POETSolution:

        t0_optimizer = time.time()

        print(f"k_hat estimator: {estimator}")

        with open(os.path.join(get_results_path(), "portfolio_optim_log.txt"), "a") as text_file:
            print(f"k_hat estimator: {estimator}",file=text_file)

        """
        daily_returns
        ------------
        rows: dates
        columns : returns
        """

        iter_num = int(np.floor((len(daily_returns) -
                                 self.p_o_config.training_sample_width) / self.p_o_config.out_of_sample_width))

        # here we always do analysis on the same size of out_of_sample_width = 21
        # therefore we need to floor it to the closest integer

        # preallocating vectors for storage
        daily_returns_vec = np.zeros(len(daily_returns) - self.p_o_config.training_sample_width)

        k_estimate_vec = np.zeros(iter_num)
        C_min_estimate_vec = np.zeros(iter_num)
        M_estimate_vec = np.zeros(iter_num)
        C_star_vec = np.zeros(iter_num)

        iter_counter = 0

        for t in range(iter_num):

            iter_counter += 1

            # if iter_counter == 10:
            #     print(f"iteration number: {t + 1}")
            #     iter_counter = 0

            t_start = time.time()

            whole_sample = daily_returns[(t * self.p_o_config.out_of_sample_width):
                                         (self.p_o_config.training_sample_width +
                                          self.p_o_config.out_of_sample_width) +
                                         (t * self.p_o_config.out_of_sample_width), :]

            training_sample = whole_sample[0: self.p_o_config.training_sample_width, :]

            out_of_sample = whole_sample[self.p_o_config.training_sample_width:
                                         (self.p_o_config.training_sample_width +
                                          self.p_o_config.out_of_sample_width), :]

            # transposing
            training_sample = training_sample.T
            out_of_sample = out_of_sample.T

            """
            rows: returns
            columns : dates
            """

            # flipping, so that the oldest values are on the left
            # ---------------------------------------------------
            training_sample = np.flip(training_sample, axis=1)
            out_of_sample = np.flip(out_of_sample, axis=1)


            # applying de-meaning on the sample data for the estimator
            # -------------------------------------------------------

            training_sample_k_hat = training_sample.copy()

            de_meaning_functions = {
                "TKCV": time_demeaning_unit_variance
            }

            training_sample_k_hat = de_meaning_functions[estimator](training_sample_k_hat)

            # training sample
            # ***************


            # estimating factor number - eigenvalue based
            # -------------------------------------------

            cross_validation_based_k_estimator_dictionary = {
                "TKCV": TKCV_fun
            }

            k_hat = cross_validation_based_k_estimator_dictionary[estimator](training_sample_k_hat, fold_number, k_max = k_max)[0]

            k_estimate_vec[t] = k_hat


            # time demeaning the training sample ( not the out of sample, there the time means are important !!! )
            # ------------------------------------------------
            training_sample = time_demeaning(training_sample)

            # computing sample covariance matrix
            # ---------------------------------------

            covariance_matrix = training_sample @ training_sample.T

            # estimating population covariance matrix
            # ---------------------------------------

            N, T = training_sample.shape

            # estimating M_hat

            M_hat = m_calculation(covariance_matrix, N, T, k_hat, C_min = 0.0,
                                  C_grid_precision= self.poet_config.C_grid_precision)

            M_estimate_vec[t] = M_hat

            # estimating C_min

            C_min = c_min_calculation(covariance_matrix, N, T, k_hat,
                      C_min_start = 0.0, C_min_end = M_hat, C_grid_precision = self.poet_config.C_grid_precision)

            C_min_estimate_vec[t] = C_min


            # determining C_star by multifold cross validation, along the determined set [C_min + eps, M_hat]

            epsilon = 0.1

            C_star = c_mcv_poet_fun(training_sample,
                                    cross_val_fold_number=self.poet_config.cross_validation_fold_number,
                                    k_hat=k_hat, C_min_plus_eps = C_min + epsilon, M = M_hat,
                                    cross_val_precision=self.poet_config.cross_validation_precision)

            C_star_vec[t] = C_star

            # C_star = 0.5


            POET_K = poet_fun(covariance_matrix, N, T, k_hat, C_star)

            # estimating portfolio weights
            # ---------------------------------------

            ones_vector = np.ones([N, 1])

            # divisor = ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector

            portfolio_weights = ( (np.linalg.inv(POET_K) @ ones_vector) /
                                 (ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector) )

            tolerance = 5
            sum_check = np.round(np.sum(portfolio_weights, 0)[0], decimals=tolerance)

            assert sum_check == 1.0, f"The sum of weights is not one, it is {sum_check}!"

            # out of sample
            # ***************

            """
            redo this as a matrix vector multiplication
            """

            for i in range(self.p_o_config.out_of_sample_width):
                portfolio_return_day_i = out_of_sample[:, i].T @ portfolio_weights

                daily_returns_vec[(t * self.p_o_config.out_of_sample_width) + i] = portfolio_return_day_i

            # sum_of_individual_returns = np.sum(out_of_sample, axis=1)
            # portfolio_return = sum_of_individual_returns.T @ portfolio_weights
            # storage_vector_return[t] = portfolio_return

            t_end = time.time()

            # C_min_plus_eps = 0
            # M_hat = 0


            print(f"iter_num: {t + 1}, C_min: {C_min}, "
                  f"M_hat: {np.round(M_hat, 2)}, C_star: {C_star}, "
                  f"elapsed time: {np.round(t_end - t_start, 2)} seconds")

            with open(os.path.join(get_results_path(), "portfolio_optim_log.txt"), "a") as text_file:
                print(f"iter_num: {t + 1}, C_min: {C_min} "
                      f"M_hat: {np.round(M_hat, 2)}, C_star: {C_star}, "
                      f"elapsed time: {np.round(t_end - t_start, 2)} seconds", file = text_file)

        t1_optimizer = time.time()

        print(f"POET with k_hat estimator {estimator}: total epalsed time: "
              f" {round(t1_optimizer - t0_optimizer, 2)} sec,"
              f" {round((t1_optimizer - t0_optimizer) / 60, 2)} min,"
              f" {round((t1_optimizer - t0_optimizer) / (60 * 60), 2)} hour")

        with open(os.path.join(get_results_path(), "portfolio_optim_log.txt"), "a") as text_file:
            print(f"POET with k_hat estimator {estimator}: total epalsed time: "
                  f" {round(t1_optimizer - t0_optimizer, 2)} sec,"
                  f" {round((t1_optimizer - t0_optimizer) / 60, 2)} min,"
                  f" {round((t1_optimizer - t0_optimizer) / (60 * 60), 2)} hour",file=text_file)


        solution = POETSolution(daily_returns_vec = daily_returns_vec,
                                k_estimate_vec = k_estimate_vec,
                                C_min_estimate_vec= C_min_estimate_vec,
                                M_estimate_vec = M_estimate_vec,
                                C_star_vec= C_star_vec)

        return solution



    def dummy_k_estimator(self, daily_returns, k_hat) -> POETSolution:

        t0_optimizer = time.time()

        print(f"constant k_hat value: {k_hat}")

        with open(os.path.join(get_results_path(), "portfolio_optim_log.txt"), "a") as text_file:
            print(f"constant k_hat value: {k_hat}",file=text_file)

        """
        daily_returns
        ------------
        rows: dates
        columns : returns
        """

        iter_num = int(np.floor((len(daily_returns) -
                                 self.p_o_config.training_sample_width) / self.p_o_config.out_of_sample_width))

        # here we always do analysis on the same size of out_of_sample_width = 21
        # therefore we need to floor it to the closest integer

        # preallocating vectors for storage
        daily_returns_vec = np.zeros(len(daily_returns) - self.p_o_config.training_sample_width)

        k_estimate_vec = np.zeros(iter_num)
        C_min_estimate_vec = np.zeros(iter_num)
        M_estimate_vec = np.zeros(iter_num)
        C_star_vec = np.zeros(iter_num)

        iter_counter = 0

        for t in range(iter_num):

            iter_counter += 1

            # if iter_counter == 10:
            #     print(f"iteration number: {t + 1}")
            #     iter_counter = 0

            t_start = time.time()

            whole_sample = daily_returns[(t * self.p_o_config.out_of_sample_width):
                                         (self.p_o_config.training_sample_width +
                                          self.p_o_config.out_of_sample_width) +
                                         (t * self.p_o_config.out_of_sample_width), :]

            training_sample = whole_sample[0: self.p_o_config.training_sample_width, :]

            out_of_sample = whole_sample[self.p_o_config.training_sample_width:
                                         (self.p_o_config.training_sample_width +
                                          self.p_o_config.out_of_sample_width), :]

            # transposing
            training_sample = training_sample.T
            out_of_sample = out_of_sample.T

            """
            rows: returns
            columns : dates
            """

            # flipping, so that the oldest values are on the left
            # ---------------------------------------------------
            training_sample = np.flip(training_sample, axis=1)
            out_of_sample = np.flip(out_of_sample, axis=1)


            # constant value of k_hat
            # -----------------------
            k_hat = k_hat

            k_estimate_vec[t] = k_hat

            # time demeaning the training sample
            # ( not the out of sample, there the time means are important !!! )
            # ------------------------------------------------
            training_sample = time_demeaning(training_sample)

            # computing sample covariance matrix - for the POET estimator
            # ----------------------------------------------------------

            covariance_matrix = training_sample @ training_sample.T

            # POET: estimating population covariance matrix
            # ---------------------------------------

            N, T = training_sample.shape

            # estimating M_hat - upper bound of C_star

            M_hat = m_calculation(covariance_matrix, N, T, k_hat, C_min = 0.0,
                                  C_grid_precision = self.poet_config.C_grid_precision)

            M_estimate_vec[t] = M_hat

            # estimating C_min - lower bound of C_star

            C_min = c_min_calculation(covariance_matrix, N, T, k_hat,
                      C_min_start = 0.0, C_min_end = M_hat, C_grid_precision = self.poet_config.C_grid_precision)

            C_min_estimate_vec[t] = C_min


            # determining C_star by multifold cross validation, along the determined set [C_min_plus_eps + eps, M_hat]

            epsilon = 0.1

            C_star = c_mcv_poet_fun(training_sample,
                                    cross_val_fold_number=self.poet_config.cross_validation_fold_number,
                                    k_hat=k_hat, C_min_plus_eps = C_min + epsilon, M = M_hat,
                                    cross_val_precision= self.poet_config.cross_validation_precision)

            C_star_vec[t] = C_star

            # C_star = 0.5

            POET_K = poet_fun(covariance_matrix, N, T, k_hat, C_star)

            # estimating portfolio weights
            # ---------------------------------------

            ones_vector = np.ones([N, 1])

            # divisor = ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector

            portfolio_weights = ( (np.linalg.inv(POET_K) @ ones_vector) /
                                 (ones_vector.T @ np.linalg.inv(POET_K) @ ones_vector) )

            tolerance = 5
            sum_check = np.round(np.sum(portfolio_weights, 0)[0], decimals=tolerance)

            assert sum_check == 1.0, f"The sum of weights is not one, it is {sum_check}!"

            # out of sample
            # ***************

            """
            redo this as a matrix vector multiplication
            """

            for i in range(self.p_o_config.out_of_sample_width):
                portfolio_return_day_i = out_of_sample[:, i].T @ portfolio_weights

                daily_returns_vec[(t * self.p_o_config.out_of_sample_width) + i] = portfolio_return_day_i

            # sum_of_individual_returns = np.sum(out_of_sample, axis=1)
            # portfolio_return = sum_of_individual_returns.T @ portfolio_weights
            # storage_vector_return[t] = portfolio_return

            # C_min_plus_eps = 0
            # M_hat = 0

            t_end = time.time()

            print(f"iter_num: {t + 1}, C_min: {C_min}, "
                  f"M_hat: {np.round(M_hat, 2)}, C_star: {C_star}, "
                  f"elapsed time: {np.round(t_end - t_start, 2)} seconds")

            with open(os.path.join(get_results_path(), "portfolio_optim_log.txt"), "a") as text_file:
                print(f"iter_num: {t + 1}, C_min: {C_min} "
                      f"M_hat: {np.round(M_hat, 2)}, C_star: {C_star}, "
                      f"elapsed time: {np.round(t_end - t_start, 2)} seconds", file = text_file)

        t1_optimizer = time.time()

        print(f"POET with constant k_hat value - {k_hat}: total epalsed time: "
              f" {round(t1_optimizer - t0_optimizer, 2)} sec,"
              f" {round((t1_optimizer - t0_optimizer) / 60, 2)} min,"
              f" {round((t1_optimizer - t0_optimizer) / (60 * 60), 2)} hour")

        with open(os.path.join(get_results_path(), "portfolio_optim_log.txt"), "a") as text_file:
            print(f"POET with constant k_hat value - {k_hat}: total epalsed time: "
                  f" {round(t1_optimizer - t0_optimizer, 2)} sec,"
                  f" {round((t1_optimizer - t0_optimizer) / 60, 2)} min,"
                  f" {round((t1_optimizer - t0_optimizer) / (60 * 60), 2)} hour",file=text_file)


        solution = POETSolution(daily_returns_vec = daily_returns_vec,
                                k_estimate_vec = k_estimate_vec,
                                C_min_estimate_vec=C_min_estimate_vec,
                                M_estimate_vec = M_estimate_vec,
                                C_star_vec= C_star_vec)

        return solution




class BasicPortfolioOptimizer:

    def __init__(self, portfolio_optim_config: PortfolioOptimConfig):

        self.p_o_config = portfolio_optim_config

    def portfolio_optim_sample_covariance_matrix(self, daily_returns) -> BasicSolverSolution:

        t0_optimizer = time.time()

        """
        daily_returns
        ------------
        rows: dates
        columns : returns
        """

        iter_num = int(np.floor((len(daily_returns) - self.p_o_config.training_sample_width)
                                / self.p_o_config.out_of_sample_width))

        # storage_vector_return = np.zeros(iter_num)

        daily_returns_vec = np.zeros(len(daily_returns) - self.p_o_config.training_sample_width)

        iter_counter = 0

        for t in range(iter_num):

            iter_counter += 1

            if iter_counter == 10:
                print(f"iteration number: {t + 1}")
                iter_counter = 0

            whole_sample = daily_returns[(t * self.p_o_config.out_of_sample_width):
                                         (self.p_o_config.training_sample_width + self.p_o_config.out_of_sample_width)
                                         + (t * self.p_o_config.out_of_sample_width), :]

            training_sample = whole_sample[0: self.p_o_config.training_sample_width, :]

            out_of_sample = whole_sample[self.p_o_config.training_sample_width:
                                         (self.p_o_config.training_sample_width + self.p_o_config.out_of_sample_width), :]

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
            training_sample = training_sample - np.mean(training_sample, axis=1).reshape(-1, 1)

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

            for i in range(self.p_o_config.out_of_sample_width):
                portfolio_return_day_i = out_of_sample[:, i].T @ portfolio_weights

                daily_returns_vec[(t * self.p_o_config.out_of_sample_width) + i] = portfolio_return_day_i

        t1_optimizer = time.time()

        print(f"sample_covariance total epalsed time: {t1_optimizer - t0_optimizer} seconds")

        solution = BasicSolverSolution(daily_returns_vec = daily_returns_vec)

        return solution

    def portfolio_optim_one_over_n(self, daily_returns) -> BasicSolverSolution:

        t0_optimizer = time.time()

        """
        daily_returns
        ------------
        rows: dates
        columns : returns
        """

        iter_num = int(np.floor((len(daily_returns) - self.p_o_config.training_sample_width)
                                / self.p_o_config.out_of_sample_width))

        # storage_vector_return = np.zeros(iter_num)

        daily_returns_vec = np.zeros(len(daily_returns) - self.p_o_config.training_sample_width)

        iter_counter = 0

        for t in range(iter_num):

            iter_counter += 1

            if iter_counter == 10:
                print(f"iteration number: {t + 1}")
                iter_counter = 0

            whole_sample = daily_returns[(t * self.p_o_config.out_of_sample_width):
                                         (self.p_o_config.training_sample_width + self.p_o_config.out_of_sample_width) +
                                         (t * self.p_o_config.out_of_sample_width), :]

            training_sample = whole_sample[0: self.p_o_config.training_sample_width, :]

            out_of_sample = whole_sample[self.p_o_config.training_sample_width:
                                         (self.p_o_config.training_sample_width + self.p_o_config.out_of_sample_width), :]

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
            training_sample = training_sample - np.mean(training_sample, axis=1).reshape(-1, 1)

            # training sample
            # ***************

            # estimating portfolio weights
            # ---------------------------------------

            N = training_sample.shape[0]

            ones_vector = np.ones([N, 1])

            portfolio_weights = ones_vector * (1 / N)

            tolerance = 5
            sum_check = np.round(np.sum(portfolio_weights, 0)[0], decimals=tolerance)

            assert sum_check == 1.0, f"The sum of weights is not one, it is {sum_check}!"

            # out of sample
            # ***************

            for i in range(self.p_o_config.out_of_sample_width):
                portfolio_return_day_i = out_of_sample[:, i].T @ portfolio_weights

                daily_returns_vec[(t * self.p_o_config.out_of_sample_width) + i] = portfolio_return_day_i


        t1_optimizer = time.time()

        print(f"1/N portfolio total epalsed time: {t1_optimizer - t0_optimizer} seconds")

        solution = BasicSolverSolution(daily_returns_vec = daily_returns_vec)

        return solution









