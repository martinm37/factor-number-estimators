

import numpy as np
import numba

from src.computation_functions.data_generating_process import dgp_nt_correlations
from src.estimators.covariance_matrix.old.poet import poet_fun
from src.estimators.covariance_matrix.old.poet_numba import poet_fun_numba

simulation_seed = 1372

N = 150
T = 250
K = 3
C = 0.7

r = 3

rng = np.random.default_rng(seed=simulation_seed)

Lambda = rng.normal(0, 1, size=(N, r))  # matrix of factor loadings - generated once per setting

# standardizing rows of lambda
Lambda_row_sum = np.sqrt(
    np.sum(Lambda * Lambda, axis=1).reshape(-1, 1))  # * is hadamard product , -> l2 norm for each row
Lambda = Lambda / Lambda_row_sum  # / is hadamard division? , -> standardizing

X = dgp_nt_correlations(Lambda=Lambda, N=N, T=T,
                                        burning_period=100,
                                        r=3, SNR=0.5, rho=0.5, beta=0.5, J=8, rng=rng)

covariance_matrix = X @ X.T

POET_python = poet_fun(covariance_matrix, N, T, K, C)

POET_python_numba = poet_fun_numba(covariance_matrix, N, T, K, C)

print("POET_python")
print(POET_python)

print("POET_python_numba")
print(POET_python_numba)

print("--------------")



print("hmmmmmm")



def first(ee):
    U, S, Vh = np.linalg.svd(ee, full_matrices=True)
    return U, S, Vh


@numba.jit(nopython = True)
def second(ee):
    U, S, Vh = np.linalg.svd(ee, full_matrices=True)
    return U, S, Vh


a = first(covariance_matrix)[0]

b = second(covariance_matrix)[0]

U1, S1, Vh1 = first(covariance_matrix)

U2, S2, Vh2 = second(covariance_matrix)


print("svd from normal python")
print(a)

print("svd from python with numba")
print(b)

result = np.allclose(a,b,rtol=1e-05, atol=1e-08)

print(f"result: {result}")

# okay amazing this is not the issue


def sum_python(S,U):

    Sum_K = np.zeros([N, N], dtype=np.float64)
    for i in range(0, K):
        # this selects the first K decompositions
        Sum_K += S[i] * U[:, [i]] @ U[:, [i]].T

    return Sum_K

@numba.jit(nopython = True)
def sum_numba(S,U):

    Sum_K = np.zeros((N, N), dtype=numba.float64)
    for i in range(0, K):
        # this selects the first K decompositions
        Sum_K += S[i] * U[:, (i)] @ U[:, (i)].T

    return Sum_K


python_sum = sum_python(S1,U1)

numba_sum = sum_numba(S2,U2)

print("sum from normal python")
print(python_sum[1,:])

print("sum from python with numba")
print(numba_sum[1,:])

result2 = np.allclose(python_sum,numba_sum,rtol=1e-05, atol=1e-08)

print(f"result: {result2}")



def dumb1(a):
    return a[1,:]

@numba.jit(nopython = True)
def dumb2(a):
    return a[1, :]

aa = dumb1(U1)

bb = dumb2(U1)

print(f"slicing:{np.allclose(aa,bb,rtol=1e-05, atol=1e-08)}")
print(aa.shape)
print(bb.shape)


def operation_python(S,U):
    i = 1
    return S[i] * U[:, [i]] @ U[:, [i]].T

# @numba.jit(nopython = True)
# def operation_numba(S,U):
#     i = 1
#     return S[i] * U[:, i] @ U[:, i].T


@numba.njit(fastmath=True)
def numba_matmul(A,B):
    return A @ B


@numba.jit(nopython = True)
def operation_numba(S,U):
    i = 1
    first = U[:, i].copy()
    first = np.reshape(first,(-1,1))
    second = U[:, i].copy()
    second = np.reshape(second,(-1,1))
    second = second.T
    return  S[i] * numba_matmul(first, second)


hmm1 = operation_python(S1,U1)
hmm2 = operation_numba(S1,U1)


print("--------------------")
print(hmm1)
print(hmm2)

print(f"hmm:{np.allclose(hmm1,hmm2,rtol=1e-05, atol=1e-08)}")

