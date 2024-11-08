


import numpy as np

from src.estimators.covariance_matrix.poet_cls_thresholder import poet_cls_thresholder

def poet_cls_c_multif_cross_val(u_matrix, cross_val_fold_number, C_min_plus_eps, M, cross_val_precision):

    # validation goes across the time dimension T

    N, T = u_matrix.shape

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

        squared_frobenius_nrm_vec = np.zeros(inter_num)


        for i in range(inter_num):

            # training set - on this we compute Sigma_u_T
            # -------------------------------------------

            u_matrix_training_set_1 = u_matrix[:, 0:(i * validation_split)]
            u_matrix_training_set_2 = u_matrix[:, validation_split + (i * validation_split):]
            u_matrix_training_set = np.concatenate([u_matrix_training_set_1, u_matrix_training_set_2], axis = 1)

            #mean_vec_1 = np.mean(u_matrix_training_set,axis = 1)


            Sigma_u_T_J1 = poet_cls_thresholder(u_matrix=u_matrix_training_set, C=C_matrix[c, 0])


            # validation set
            # --------------
            u_matrix_validation_set = u_matrix[:, (i * validation_split): validation_split + (i * validation_split)]

            validation_N, validation_T = u_matrix_validation_set.shape

            #mean_vec_2 = np.mean(u_matrix_validation_set,axis = 1)

            Sigma_u_J2 = (1/validation_T) * u_matrix_validation_set @ u_matrix_validation_set.T


            # squared frobenius norm
            # -----------------------
            difference_mat = Sigma_u_T_J1 - Sigma_u_J2
            squared_frobenius_nrm_vec[i] = np.trace(difference_mat.T @ difference_mat)

        # computing the average across the folds
        # --------------------------------------
        C_matrix[c,1] = np.mean(squared_frobenius_nrm_vec)


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





