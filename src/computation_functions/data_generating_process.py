
"""
code for the data generating process
"""


# importing packages
# ------------------
import numpy as np

"""
Lambda is not redrawn every period,
just once per simulation!!!!
"""

# creating the class of rng
#rng = np.random.default_rng()


# main DGP function:

def dgp_nt_correlations(Lambda, N, T, burning_period, r, SNR, rho, beta, J, rng):
    """
    rng is an instance of random number generator class created in the calling file
    """

    # creating the class of rng
    # rng = np.random.default_rng(seed=simulation_seed)

    F = rng.normal(0, 1, size=(r, T))  # matrix of factors

    v_mat = rng.normal(0, 1, size=(N, T + burning_period))
    e_mat_ph = np.zeros((N, T + burning_period))
    e_mat = np.zeros((N, T + burning_period))

    # filling cross sectional correlations:
    for i in range(0, N):
        e_mat_ph[i, :] = v_mat[i, :] + \
                         beta * (np.sum(v_mat[np.maximum(i - J, 0):i, :], axis=0) + \
                                 np.sum(v_mat[(i + 1):np.minimum(N, i + J + 1), :], axis=0))

    # filling up the first period t = 1
    e_mat[:, 0] = e_mat_ph[:, 0]

    # filling temporal correlations
    for t in range(1, T + burning_period):  # start in t = 2
        e_mat[:, t] = rho * e_mat[:, t - 1] + e_mat_ph[:, t]

    e_mat = np.flip(e_mat, axis=1)

    # removing the burning period
    e_mat = e_mat[:, 0: -burning_period]

    # setting standardization
    theta = ((1 - rho ** 2) / (1 + 2 * J * beta ** 2)) * (1 / SNR)
    u_mat = np.sqrt(theta) * e_mat

    X = Lambda @ F + u_mat  # panel of data

    return X



















def dgp_basic(Lambda, N, T, r, SNR, rng):

    """
    rng is an instance of random number generator class created in the calling file
    """

    # creating the class of rng
    # rng = np.random.default_rng(seed = simulation_seed)

    #Lambda = norm.rvs(0,1,(N,r)) # matrix of factor loadings
    F = rng.normal(0,1,size = (r,T)) # matrix of factors
    e_mat = rng.normal(0,1,size = (N,T)) # error matrix

    # setting standardization
    rho = 0
    beta = 0
    J = 0
    theta = ((1 - rho ** 2) / (1 + 2 * J * beta ** 2)) * (1 / SNR)
    u_mat = np.sqrt(theta) * e_mat

    X = Lambda @ F + u_mat  # panel of data

    return X




def dgp_t_correlations(Lambda, N, T,burning_period, r, SNR, rho, rng):

    # creating the class of rng
    # rng = np.random.default_rng(seed=simulation_seed)

    F = rng.normal(0, 1, size=(r, T))  # matrix of factors


    v_mat = rng.normal(0, 1, size=(N, T+burning_period))
    e_mat = np.zeros((N,T+burning_period))


    # filling up the first period t = 1
    e_mat[:,0] = v_mat[:,0]

    # recursive filling    
    for t in range(1,T+burning_period): # start in t = 2
        e_mat[:,t] = rho * e_mat[:,t-1] + v_mat[:,t]

    e_mat = np.flip(e_mat,axis=1)

    # removing the burning period
    e_mat = e_mat[:, 0: -burning_period]

    # setting standardization
    beta = 0
    J = 0
    theta = ((1 - rho ** 2) / (1 + 2 * J * beta ** 2)) * (1 / SNR)
    u_mat = np.sqrt(theta) * e_mat

    X = Lambda @ F + u_mat  # panel of data

    return X



def dgp_n_correlations(Lambda, N, T, r, SNR, beta, J, rng):

    # creating the class of rng
    # rng = np.random.default_rng(seed=simulation_seed)

    #Lambda = norm.rvs(0,1,(N,r)) # matrix of factor loadings
    F = rng.normal(0, 1, size=(r, T))  # matrix of factors

    v_mat = rng.normal(0, 1, size=(N, T))
    e_mat = np.zeros((N,T))


    # just cross sectional correlations:
    for i in range(0,N):
        e_mat[i,:] = v_mat[i,:] + \
                     beta * (np.sum(v_mat[np.maximum(i-J,0):i,:],axis=0) + \
                             np.sum(v_mat[(i+1):np.minimum(N,i+J+1),:],axis=0))

    e_mat = np.flip(e_mat,axis=1)

    # setting standardization
    rho = 0
    theta = ((1 - rho ** 2) / (1 + 2 * J * beta ** 2)) * (1 / SNR)
    u_mat = np.sqrt(theta) * e_mat

    X = Lambda @ F + u_mat  # panel of data

    return X
















