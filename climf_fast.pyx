# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Corey Lynch <coreylynch9@gmail.com>
#

import numpy as np
import scipy 
from libc.math cimport exp, log, pow
cimport numpy as np
cimport cython

np.import_array()

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INTEGER

cdef extern from 'cblas.h':
    double ddot "cblas_ddot"(int N, double* X, int incX, double* Y,
                             int incY) nogil

cdef DOUBLE g(DOUBLE x) nogil:
    """sigmoid function"""
    x = max(min(x, 25), -25)
    return 1/(1+exp(-x))

cdef DOUBLE dg(DOUBLE x) nogil:
    """derivative of sigmoid function"""
    return exp(x)/(1+exp(x))**2

cdef void precompute_f(DOUBLE[::1] f, 
                       DOUBLE[:, ::1] U, DOUBLE[:, ::1] V, 
                       INTEGER *x_ind_ptr,
                       int xnnz,
                       int num_factors,
                       int i) nogil:
    """precompute f[j] = <U[i],V[j]>
        params:
          data: scipy csr sparse matrix containing user->(item,count)
          U   : user factors
          V   : item factors
          i   : user of interest
        returns:
          dot products <U[i],V[j]> for all j in data[i]
    """
    cdef unsigned int j = 0
    cdef unsigned int factor = 0

    # create f as a ndarray of len(nnz)
    for j in range(xnnz):
        f[j] = ddot(num_factors, &U[i, 0], 1, &V[x_ind_ptr[j], 0], 1)
        #dot_prod = 0.0
        #for factor in range(num_factors):
        #    dot_prod += U[i,factor] * V[x_ind_ptr[j],factor]
        #f[j] = dot_prod


def compute_mrr_fast(np.ndarray[int, ndim=1, mode='c'] test_user_ids, np.ndarray test_user_data,
                np.ndarray[DOUBLE, ndim=2, mode='c'] U, np.ndarray[DOUBLE, ndim=2, mode='c'] V):
    
    cdef np.ndarray[DOUBLE, ndim=1, mode='c'] mrr = np.zeros(test_user_ids.shape[0], dtype=np.float64, order="c")
    cdef unsigned int ix
    cdef unsigned int i
    cdef unsigned int item
    cdef unsigned int item_idx
    cdef unsigned int user_idx
    cdef set items
    cdef np.ndarray[DOUBLE, ndim=1, mode='c'] predictions = np.zeros(V.shape[0], dtype=np.float64, order="c")
    cdef DOUBLE pred
    cdef int rank
    cdef int num_factors = U.shape[1]
    cdef np.ndarray[INTEGER, ndim=1, mode='c'] test_user

    for i in range(test_user_ids.shape[0]):
        test_user = test_user_data[i]
        items = {item for item in test_user}
        for item_idx in range(V.shape[0]):
            pred = 0.0
            for factor in range(num_factors):
                user_idx = test_user_ids[i]
                pred += U[user_idx, factor] * V[item_idx, factor]
            predictions[item_idx] = pred
    
        ranked_preds = np.argsort(predictions)
        ranked_preds= ranked_preds[::-1]
        for rank,item in enumerate(ranked_preds):
            if item in items:
                mrr[i] = 1.0/(rank+1)
                break
    assert(len(mrr) == len(test_user_ids))
    return np.mean(mrr)


cpdef climf_fast(int[::1] X_indices, int[::1] X_indptr, 
                 DOUBLE[:, ::1] U, DOUBLE[:, ::1] V, 
                 int[::1] user_indices,
                 double lbda,
                 double gamma,
                 int n_factors,
                 int shuffle,
                 int seed):

    # get the data information into easy vars
    cdef Py_ssize_t n_users = U.shape[0]
    cdef Py_ssize_t n_items = V.shape[0]

    cdef unsigned int i, lo, hi
    cdef int[::1] x_ind_ptr

    cdef DOUBLE[::1] f = np.zeros(n_items, dtype=np.float64, order='c')

    cdef DOUBLE[::1] dU = np.zeros(n_factors, dtype=np.float64, order="c")
    cdef DOUBLE[::1] dV = np.zeros(n_factors, dtype=np.float64, order="c")

    cdef DOUBLE[::1] V_j_minus_V_k = np.zeros(n_factors, dtype=np.float64, order="c")

    if shuffle > 0:
        dataset.shuffle(seed)

    with nogil:
        for i in range(n_users):
            lo, hi = X_indptr[user_indices[i]], X_indptr[user_indices[i] + 1]
            x_ind_ptr = X_indices[lo:hi]
            climf_fast_u(x_ind_ptr, U, V, dU, dV, f, V_j_minus_V_k, i, lbda, gamma, n_factors)
        

cdef void climf_fast_u(int[::1] x_ind_ptr, DOUBLE[:, ::1] U, DOUBLE[:, ::1] V, 
                       DOUBLE[::1] dU, DOUBLE[::1] dV, DOUBLE[::1] f,
                       DOUBLE[::1] V_j_minus_V_k, unsigned int i, double lbda,
                       double gamma, int n_factors) nogil:
    # helper variable
    cdef int xnnz = x_ind_ptr.shape[0]
    cdef DOUBLE y = 0.0
    cdef unsigned int j, k, idx, idx_j, idx_k

    cdef DOUBLE dVUpdate, dUUpdate

    # dU = -lbda * U[i]
    for idx in range(n_factors):
        dU[idx] = -lbda * U[i, idx]

    precompute_f(f, U, V, x_ind_ptr, xnnz, n_factors, i)

    for j in range(xnnz):
        idx_j = x_ind_ptr[j]
        # dV = g(-f[j])-lbda*V[j]
        for idx in range(n_factors):
                dV[idx] = g(-f[j]) - lbda * V[idx_j, idx]

        for k in range(xnnz):
            dVUpdate = dg(f[j]-f[k])*(1/(1-g(f[k]-f[j]))-1/(1-g(f[j]-f[k])))
            # dV += dg(f[j]-f[k])*(1/(1-g(f[k]-f[j]))-1/(1-g(f[j]-f[k])))*U[i]
            for idx in range(n_factors):
                dV[idx] += dVUpdate * U[i, idx]
            
        # V[j] += gamma*dV
        for idx in range(n_factors):
            V[idx_j, idx] += gamma * dV[idx]
        # dU += g(-f[j])*V[idx_j]
        dUUpdate = g(-f[j])
        for idx in range(n_factors):
            dU[idx] += dUUpdate * V[idx_j, idx]
        for k in range(xnnz):
            idx_k = x_ind_ptr[k]
            # dU += (V[j]-V[k])*dg(f[k]-f[j])/(1-g(f[k]-f[j]))
            for idx in range(n_factors):
                V_j_minus_V_k[idx] = V[idx_j, idx] - V[idx_k, idx]
            for idx in range(n_factors):
                dU[idx] += V_j_minus_V_k[idx] * dg(f[k]-f[j])/(1-g(f[k]-f[j]))
    # U[i] += gamma*dU
    for idx in range(n_factors):
        U[i, idx] += gamma * dU[idx]
