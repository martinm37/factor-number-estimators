
import numpy as np
import pandas as pd



def mc_dimensions_setting(N_list,T_list):
    n_length = len(N_list)
    t_length = len(T_list)
    setting = np.zeros((n_length*t_length,2), dtype=np.intc)

    for j in range(t_length):
        for i in range(n_length):
            #setting[(j*n_length)+i,(j*n_length)+i] = N_list[i],T_list[j]
            setting[(j * n_length) + i, 0] = N_list[i]
            setting[(j * n_length) + i, 1] = T_list[j]

    return setting

"""
it is in this form i nested in j, because for every 
j of time we have several variations of i across n

"""

# ***********************************************************************

def df_maker(input_tbl,result_tbl,col_names):
    intermediate_tbl = np.concatenate((input_tbl,result_tbl),axis=1,dtype=object)
    intermediate_df = pd.DataFrame(data = intermediate_tbl,
                                   columns = col_names )
    return intermediate_df

def df_joiner(input_tbl,result_tbl,col_names):
    # if input_tbl.shape[1] == 1:
    #     input_df = pd.Series(input_tbl)
    # else:
    input_df = pd.DataFrame(input_tbl)
    intermediate_df = pd.concat((input_df,result_tbl),axis = 1)
    intermediate_df.columns = col_names

    return intermediate_df



# ***********************************************************************

def rmse_fun(output_cube, true_r):

    dims_length, estimator_num, iter_num = output_cube.shape

    squared_differences = np.zeros((dims_length, estimator_num, iter_num), dtype=np.float64)

    for iteration in range(iter_num):
        squared_differences[:, :, iteration] = np.square(output_cube[:, :, iteration] - true_r)


    SSD = np.sum(squared_differences, 2)

    RMSE = np.sqrt(SSD / iter_num)

    return np.round(RMSE, decimals=2)



def under_over_estimation_stat_fun(output_cube, true_r):


    dims_length, estimator_num, iter_num = output_cube.shape

    result = np.zeros((dims_length, estimator_num), dtype=object)

    for i in range(dims_length):
        for j in range(estimator_num):
            output_cube_subset = output_cube[i,j,:]
            underestimated = np.round(len(output_cube_subset[output_cube_subset <  true_r ]) / iter_num * 100 , decimals = 2)
            overestimated = np.round(len(output_cube_subset[output_cube_subset >  true_r  ]) / iter_num * 100 , decimals = 2)
            result[i,j] = f"{underestimated}/{overestimated}"

            #print("hello there")


    return result



def under_exact_over_estimation_stat_fun(output_cube, true_r):


    dims_length, estimator_num, iter_num = output_cube.shape

    result = np.zeros((dims_length, estimator_num), dtype=object)

    for i in range(dims_length):
        for j in range(estimator_num):
            output_cube_subset = output_cube[i,j,:]
            underestimated = int(np.round(len(output_cube_subset[output_cube_subset <  true_r ]) / iter_num * 100 , decimals = 0))
            exact = int(np.round(len(output_cube_subset[output_cube_subset == true_r]) / iter_num * 100, decimals=0))
            overestimated = int(np.round(len(output_cube_subset[output_cube_subset >  true_r  ]) / iter_num * 100 , decimals = 0))
            assert underestimated + exact + overestimated == 100, "something's wrong"
            result[i,j] = f"{underestimated}/{exact}/{overestimated}"

            #print("hello there")


    return result









def poet_k_hat_stats_fun(results_data,k_hat_estimator):

    poet_k_hat_stats = {
        "MEAN": np.mean(results_data[k_hat_estimator], axis=0) * 250,
        "STD": np.std(results_data[k_hat_estimator]) * np.sqrt(250) ,
        "SHARPE_RATIO": np.mean(results_data[k_hat_estimator], axis=0) / np.std(results_data[k_hat_estimator]) # shouldnt this also include the risk free rate?

    }

    return pd.DataFrame(poet_k_hat_stats, index=[0])



def poet_stats_full_raw(results_data,desired_estimators):

    stats_list = []

    df_index = ["MEAN","STD","SHARPE_RATIO"]


    for i in range(len(desired_estimators)):
        stats_list.append({
            "MEAN": np.mean(results_data[desired_estimators[i]], axis=0) * 250,
            "STD": np.std(results_data[desired_estimators[i]]) * np.sqrt(250) ,
            "SHARPE_RATIO": np.mean(results_data[desired_estimators[i]], axis=0) /
                            np.std(results_data[desired_estimators[i]]) # shouldnt this also include the risk free rate?

        })

    stats_df = pd.DataFrame(stats_list,index=desired_estimators, columns=df_index)

    return stats_df.transpose()


def poet_stats_full_rounded(results_data,desired_estimators):

    stats_list = []

    df_index = ["MEAN","STD","SHARPE_RATIO"]

    for i in range(len(desired_estimators)):

        mean = np.mean(results_data[desired_estimators[i]], axis=0) * 250
        std = np.std(results_data[desired_estimators[i]]) * np.sqrt(250)

        if (desired_estimators[i] == "sample_covar_mat" and mean == 0 and std == 0) :
            sharpe_ratio = 0 # here we do not use the covar mat
        else:
            sharpe_ratio = mean / std       # shouldnt this also include the risk free rate?

        stats_list.append({
            "MEAN": np.round(mean, decimals = 2),
            "STD": np.round(std, decimals = 2) ,
            "SHARPE_RATIO": np.round(sharpe_ratio, decimals = 2)
        })

    stats_df = pd.DataFrame(stats_list,index=desired_estimators, columns=df_index)

    return stats_df.transpose()

