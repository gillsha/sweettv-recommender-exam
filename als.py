"""Weighted implicit ALS (Hu, Koren & Volinsky), exact normal-equation solver.

p_ui=1 if observed, else 0. c_ui=1+alpha*log1p(n_ui) if observed,
else 1. Minimize sum c_ui*(p_ui-x_u@y_i)^2 + reg*(||X||²+||Y||²).
No unobserved pair is materialized. The missing-data baseline is Y.T@Y.
"""
import numpy as np
from numba import njit,prange,set_num_threads
from threadpoolctl import threadpool_limits

@njit(parallel=True,cache=False)
def solve_side(indptr,indices,confidence,fixed,regularization):
    factors=fixed.shape[1]
    gram=fixed.T@fixed
    result=np.zeros((len(indptr)-1,factors),np.float64)
    for row in prange(len(indptr)-1):
        lo,hi=indptr[row],indptr[row+1]
        if lo==hi:
            continue
        a=gram.copy()
        b=np.zeros(factors,np.float64)
        for j in range(factors): a[j,j]+=regularization
        for ptr in range(lo,hi):
            idx=indices[ptr];c=confidence[ptr]
            for j in range(factors):
                b[j]+=c*fixed[idx,j]
                for k in range(factors):
                    a[j,k]+=(c-1)*fixed[idx,j]*fixed[idx,k]
        result[row]=np.linalg.solve(a,b)
    return result

class ImplicitALS:
    def __init__(self,factors=32,regularization=.1,iterations=20,random_state=2026,num_threads=2):
        self.factors=factors;self.regularization=regularization
        self.iterations=iterations;self.random_state=random_state;self.num_threads=num_threads

    def fit(self,confidence):
        set_num_threads(self.num_threads)
        csr=confidence.tocsr().astype(np.float64);csc=csr.T.tocsr()
        rng=np.random.default_rng(self.random_state)
        self.item_factors=rng.normal(0,.01,(csr.shape[1],self.factors))
        with threadpool_limits(limits=1):
            for iteration in range(self.iterations):
                self.user_factors=solve_side(csr.indptr,csr.indices,csr.data,self.item_factors,self.regularization)
                self.item_factors=solve_side(csc.indptr,csc.indices,csc.data,self.user_factors,self.regularization)
        return self

def verify_solver():
    """Independent dense objective verifies one update and cold-user behavior."""
    from scipy.sparse import csr_matrix
    fixed=np.array([[.1,.2],[.3,-.2],[.5,.4]],dtype=np.float64)
    conf=csr_matrix([[3.,0.,5.],[0.,0.,0.]])
    actual=solve_side(conf.indptr,conf.indices,conf.data,fixed,.1)
    c=np.array([3.,1.,5.]);p=np.array([1.,0.,1.])
    expected=np.linalg.solve(fixed.T@np.diag(c)@fixed+.1*np.eye(2),fixed.T@(c*p))
    assert np.allclose(actual[0],expected,atol=1e-10)
    assert np.allclose(actual[1],0)
    return {'dense_normal_equation_max_error':float(np.max(np.abs(actual[0]-expected))),'cold_user_zero_factors':True}
