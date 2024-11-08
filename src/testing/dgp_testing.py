

import numpy as np

from src.computation_functions.data_generating_process import dgp_nt_correlations,dgp_t_correlations



N = 100
T = 250
burning_period = 100


SNR = 1.2
rho = 0.25
beta = 0.25
J = 8

k_max = 8

r = 3

simulation_seed = 1372

rng = np.random.default_rng(seed=simulation_seed)

Lambda = rng.normal(0,1,size = (N, r)) # matrix of factor loadings - generated once per setting



X = dgp_nt_correlations(Lambda, N, T, burning_period, r, SNR, rho, beta, J, rng)










print("hello xdd")




















# creating the class of rng
rng = np.random.default_rng()

T = 5

test_mat = np.zeros([5,T])

test_mat[:,[0]] = np.zeros([5,1])

for i in range(1,T):
    test_mat[:,[i]] = np.ones([5,1]) * i

"""
T is 5 - we have 5 time periods, but loop runs from 1 as the 
1st period is done separately
"""

test_mat = np.flip(test_mat,axis=1)

# flipping


# adding the burning period


T = 5
burning_period = 100

test_mat = np.zeros([5,T+burning_period])

# creating the initial observation
test_mat[:,[0]] = np.zeros([5,1])

for i in range(1,T+burning_period):
    test_mat[:,[i]] = np.ones([5,1]) * i

test_mat = np.flip(test_mat,axis=1)

# removing the burning period
test_mat = test_mat[:,0:-burning_period]



print("hello xdd")



def dgp_t_correlations(Lambda, N, T,burning_period, r, theta, rho):

    #Lambda = norm.rvs(0,1,(N,r)) # matrix of factor loadings
    F = rng.normal(0, 1, size=(r, T))  # matrix of factors

    v_mat = rng.normal(0, 1, size=(N, T))
    e_mat = np.zeros((N,T))
    u_mat = np.zeros((N,T))


    # filling up the first period t = 1
    e_mat[:,0] = v_mat[:,0]

    # recursive filling
    for t in range(1,T): # start in t = 2
        e_mat[:,t] = rho * e_mat[:,t-1] + v_mat[:,t]

    e_mat = np.flip(e_mat,axis=1)

    #u_mat = np.sqrt((1 - rho**2)/(1 + 2*J*(beta**2))) * e_mat # centering

    X = Lambda@F + np.sqrt(theta)*e_mat # panel of data

    return X