


import numpy as np

from src.estimators.covariance_matrix.old.poet_numba import r_k_threshold_fun, r_k_raw_fun


def c_mcv_poet_fun(data, cross_val_fold_number, k_hat, C_min_plus_eps, M, cross_val_precision):

    # validation goes across the time dimension T

    N, T = data.shape

    """
    cross_val_fold_number = 5
    validation = 1 / cross_val_fold_number = 1 / 5
    training = 1 - validation = 4 / 5
    
    """

    validation_share = 1 / cross_val_fold_number
    training_share = 1 - validation_share

    """
    add an assert statement that the division should not
    leave anything behind, yielding a whole number
    """

    validation_split = int( np.round( T * validation_share, decimals = 5 ) )

    C_vector = np.arange(C_min_plus_eps, M + cross_val_precision, cross_val_precision) # closed interval []
    C_matrix = np.concatenate([C_vector.reshape(-1,1), np.zeros(len(C_vector)).reshape(-1,1)],axis = 1)


    for c in range(len(C_vector)):
        #print(C_matrix[c,0])

        inter_num = int( np.round( T / validation_split, decimals = 5) )

        sfn_vector = np.zeros(inter_num)

        testing_vector = np.zeros(inter_num,dtype=np.int64)

        for i in range(inter_num):

            training_set_1 = data[:, 0 :(i * validation_split )]
            training_set_2 = data[:, validation_split + (i * validation_split ): ]
            training_set = np.concatenate([training_set_1,training_set_2],axis = 1)

            training_N, training_T = training_set.shape


            validation_set = data[:, (i * validation_split ): validation_split + (i * validation_split )]

            validation_N, validation_T = validation_set.shape


            training_covariance_matrix = training_set @ training_set.T
            validation_covariance_matrix = validation_set @ validation_set.T

            R_K_T_J1 = r_k_threshold_fun(covariance_matrix=training_covariance_matrix, N=training_N, T=training_T, K=k_hat, C=C_matrix[c,0])

            R_K_J2 = r_k_raw_fun(covariance_matrix = validation_covariance_matrix, N=validation_N, K=k_hat)

            # squared frobenius norm
            # -----------------------
            difference_mat = R_K_T_J1 - R_K_J2
            sfn_vector[i] = np.trace(difference_mat.T @ difference_mat)

        # computing the average across the folds
        # --------------------------------------
        C_matrix[c,1] = np.mean(sfn_vector)


    argmin = np.argmin(C_matrix[:,1])
    C_star = np.round(C_matrix[argmin,0],decimals = 5)

    return C_star













# Sigma_POET = poet_fun(covariance_matrix=training_covariance_matrix,N=training_N, T=training_T,K=k_hat,C=C_matrix[c,0])
#
# # do i need to check this here ??????
# # checking the diagonality of the matrix
# if np.all(Sigma_POET == np.diag(np.diag(Sigma_POET))):
#     testing_vector[i] = 1
#
# Sigma_validation = validation_covariance_matrix

# # squared frobenius norm
# difference_mat = Sigma_POET - Sigma_validation
# sfn_vector[i] = np.trace(difference_mat.T @ difference_mat)
# print(f"c iter {C_matrix[c,0]}, i iter: {i}, squared frobenius norm of difference: {np.trace(difference_mat.T @ difference_mat)}")


"""
here it needs to be checked if the matrix is not diagonal,
if it is the for loop stops
there needs to be a counter as well, and the matrix will be
cut to that counter
then the argmin of the matrix is computed
-> this is now delegated to the function computing M separately for the whole POET matrix
"""





