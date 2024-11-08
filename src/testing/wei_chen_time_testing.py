import timeit

#
#
# N = 500
# T = N
# r = 3
# theta = r
#
# Lambda = norm.rvs(0,1,(N,r))
# X = dgp.DGP_V3_b(Lambda=Lambda,N=N,T=T,r=r,theta=theta)
#
# N, T = X.shape
# j = int(N / 5)
# k = int(T / 5)
#
# cProfile.run("WC.TKCV_fun(X=X,j=j,k=j,k_max=8)")
#
#
#
#
# """
# slice = 10 for the following:
#
# 500 x 500
# 25.384 seconds - old OLS method
# 26.068  - new
#
# 750 x 750
# 91.741 seconds - new method
# 114.211 seconds - old method
# """



##########################


mysetup = """

import numpy as np
import timeit
import cProfile

from scipy.stats import norm # normal distribution

import src.utils.data_generating_process as dgp
import src.estimators.wei_chen as WC

N = 500
T = N
r = 3
theta = r

Lambda = norm.rvs(0,1,(N,r))
X = dgp.DGP_V3_b(Lambda=Lambda,N=N,T=T,r=r,theta=theta)

N, T = X.shape
j = int(N / 5)
k = int(T / 5)


"""

# code snippet whose execution time is to be measured
mycode = "WC.TKCV_fun(X=X,j=j,k=k,k_max=8)"

iter_num = 100


time = timeit.timeit(stmt=mycode,
                     setup=mysetup,
                     number=iter_num)


print(time / iter_num)

"""
5 sec for 500x500 slice 50
13.42 for 700x700 slice 70

2.48 sec with the new slices
4.42 with 100 iterations
"""