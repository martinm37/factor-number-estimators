




import numpy as np
import numba


@numba.jit(nopython=True)
def poet_cls_preprocessor(X,K):

    """
    :param X: (N x T) panel data
    :param K: estimated factor number
    :return[0]: LLT_matrix -> matrix of factor loading times its transpose
    :return[1]: u_matrix -> matrix of errors
    """

    N,T = X.shape

    gram_matrix = X.T @ X

    eigen_vals, eigen_vecs = np.linalg.eigh(gram_matrix)

    index = np.argsort(eigen_vals)[::-1]
    eigen_vecs_sort = eigen_vecs[:, index]

    F_matrix = np.sqrt(T) * eigen_vecs_sort[:,:K]

    L_Matrix = (1/T) * X @ F_matrix

    u_matrix = X - L_Matrix @ F_matrix.T

    return L_Matrix @ L_Matrix.T, u_matrix


