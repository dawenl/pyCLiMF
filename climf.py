"""
CLiMF Collaborative Less-is-More Filtering, a variant of latent factor CF
which optimises a lower bound of the smoothed reciprocal rank of "relevant"
items in ranked recommendation lists.  The intention is to promote diversity
as well as accuracy in the recommendations.  The method assumes binary
relevance data, as for example in friendship or follow relationships.

CLiMF: Learning to Maximize Reciprocal Rank with Collaborative Less-is-More Filtering
Yue Shi, Martha Larson, Alexandros Karatzoglou, Nuria Oliver, Linas Baltrunas, Alan Hanjalic
ACM RecSys 2012
"""

from math import exp, log
import sys
import time
import numpy as np
import random
from climf_fast import climf_fast, CSRDataset, compute_mrr_fast

def _make_dataset(X):
    """Create ``Dataset`` abstraction for sparse and dense inputs."""
    y_i = np.ones(X.shape[0], dtype=np.float64, order='C')
    sample_weight = np.ones(X.shape[0], dtype=np.float64, order='C') # ignore sample weight for the moment
    dataset = CSRDataset(X.data, X.indptr, X.indices, y_i, sample_weight)
    return dataset

class CLiMF:
    def __init__(self, dim=10, lbda=0.001, gamma=0.0001, max_iters=5, verbose=True,
                 shuffle=True, seed=28):
        self.dim = dim
        self.lbda = lbda
        self.gamma = gamma
        self.max_iters = max_iters
        self.verbose = verbose
        self.shuffle = 1 if shuffle else 0
        self.seed = seed

    def fit(self, X):
        data = _make_dataset(X)
        self.U = 0.01*np.random.random_sample(size=(X.shape[0], self.dim))
        self.V = 0.01*np.random.random_sample(size=(X.shape[1], self.dim))

        num_train_sample_users = min(X.shape[0],100)
        train_sample_users = np.array(random.sample(xrange(X.shape[0]),num_train_sample_users), dtype=np.int32)
        sample_user_data = np.array([np.array(X.getrow(i).indices, dtype=np.int32) for i in train_sample_users])
        
        for it in xrange(self.max_iters):
            start_t = time.time()
            climf_fast(data, self.U, self.V, self.lbda, self.gamma, self.dim, 
                       self.shuffle, self.seed)
            t = time.time() - start_t
            print('iteration {0}:'.format(it+1))
            print('train mrr = {:.8f} (time = {:.2f})'.format(compute_mrr_fast(train_sample_users, sample_user_data, self.U, self.V), t))
            sys.stdout.flush()

    def compute_mrr(self, testdata):
        return compute_mrr_fast(np.array(range(testdata.shape[0]), dtype=np.int32), 
                                np.array([np.array(testdata.getrow(i).indices, dtype=np.int32) for i in range(testdata.shape[0])]), 
                                self.U, self.V)
