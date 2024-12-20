
import numpy as np
import pandas as pd



def index_finder(vector,choice):
    if choice == "max":
        if np.all(vector == 0):
            return ([100000])
        else:
            return np.where(vector == vector.max())
    elif choice == "min":
        return np.where(vector == vector.min())



def rmse_stat_fun(output_cube, true_r, precision):

    dims_length, estimator_num, iter_num = output_cube.shape

    squared_differences = np.zeros((dims_length, estimator_num, iter_num), dtype=np.float64)

    for iteration in range(iter_num):
        squared_differences[:, :, iteration] = np.square(output_cube[:, :, iteration] - true_r)


    SSD = np.sum(squared_differences, 2)

    RMSE = np.sqrt(SSD / iter_num)

    RMSE = np.round(RMSE, decimals = precision)

    #result = pd.DataFrame(np.zeros((dims_length, estimator_num), dtype=object))
    result = np.zeros((dims_length, estimator_num), dtype=object)

    for i in range(dims_length):

        row = RMSE[i,:]
        row_index = index_finder(row, "min")

        bcklshB = "\B"

        for j in range(estimator_num):
            if np.any(row_index[0] == j):
                # result.iloc[i, j] = f"{bcklshB}{{({RMSE[i,j]})}}"
                result[i, j] = f"{bcklshB}{{({RMSE[i, j]})}}"
            else:
                # result.iloc[i, j] = f"({RMSE[i,j]})"
                result[i, j] = f"({RMSE[i, j]})"

    return result


def under_exact_over_estimation_stat_fun(output_cube, true_r,precision):


    dims_length, estimator_num, iter_num = output_cube.shape

    # result = pd.DataFrame(np.zeros((dims_length, estimator_num), dtype=object))
    result = np.zeros((dims_length, estimator_num), dtype=object)


    for i in range(dims_length):

        underestimated_row = np.zeros(estimator_num, dtype=np.int64)
        exact_row = np.zeros(estimator_num, dtype=np.int64)
        overestimated_row = np.zeros(estimator_num, dtype=np.int64)

        for j in range(estimator_num):

            output_cube_subset = output_cube[i,j,:]

            underestimated = int(np.round(len(output_cube_subset[output_cube_subset <  true_r ]) / iter_num * 100 , decimals = precision))
            exact = int(np.round(len(output_cube_subset[output_cube_subset == true_r]) / iter_num * 100, decimals = precision))
            overestimated = int(np.round(len(output_cube_subset[output_cube_subset >  true_r  ]) / iter_num * 100 , decimals = precision))

            underestimated_row[j] = underestimated
            exact_row[j] = exact
            overestimated_row[j] = overestimated

        exact_row_index = index_finder(exact_row,"max")

        bcklshB = "\B"

        for j in range(estimator_num):
            if np.any(exact_row_index[0] == j):
                #result.iloc[i, j] = f"{underestimated_row[j]}/{bcklshB}{{{exact_row[j]}}}/{overestimated_row[j]}"
                result[i, j] = f"{underestimated_row[j]}/{bcklshB}{{{exact_row[j]}}}/{overestimated_row[j]}"
            else:
                #result.iloc[i, j] = f"{underestimated_row[j]}/{exact_row[j]}/{overestimated_row[j]}"
                result[i, j] = f"{underestimated_row[j]}/{exact_row[j]}/{overestimated_row[j]}"

    return result


def joined_mc_stats_latex_fun(output_cube, true_r, precision_ueo_estimation, precision_rmse):

    dims_length, estimator_num, iter_num = output_cube.shape

    ueo_estimation_stat = under_exact_over_estimation_stat_fun(output_cube, true_r, precision = precision_ueo_estimation)
    rmse_stat = rmse_stat_fun(output_cube, true_r, precision = precision_rmse)

    # joined_stat = pd.DataFrame(np.zeros((dims_length, estimator_num), dtype=object))
    joined_stat = np.zeros((dims_length, estimator_num), dtype=object)

    for i in range(dims_length):
        for j in range(estimator_num):
            # joined_stat.iloc[i, j] = f"{ueo_estimation_stat.iloc[i, j]} {rmse_stat.iloc[i, j]}"
            joined_stat[i, j] = f"{ueo_estimation_stat[i, j]} {rmse_stat[i, j]}"

    #print("xdd")

    return joined_stat


# sum_check = underestimated + exact + overestimated
# assert sum_check == 100, f"something's wrong, sum check is {sum_check}, place {(i,j)}"

"""
apparently this is more complicated than I thought, e.g. i have % = {33.33, 33.33, 33.33}, 
which of them do I round up to 34% ??????
"""