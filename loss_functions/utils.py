import numpy as np
import scipy


def safe_sparse_add(a, b):
    """
    Implemenet a+b compatible with different types of input.
    Supports scalars, numpy arrays and scipy.sparse objects.
    """
    both_sparse = scipy.sparse.issparse(a) and scipy.sparse.issparse(b)
    one_is_scalar = isinstance(a, (int, float)) or isinstance(b, (int, float))
    if both_sparse or one_is_scalar:
        # both are sparse, keep the result sparse
        return a + b
    else:
        # one of them is non-sparse, convert
        # everything to dense.
        if scipy.sparse.issparse(a):
            a = a.toarray()
            if a.ndim == 2 and b.ndim == 1:
                b.ravel()
        elif scipy.sparse.issparse(b):
            b = b.toarray()
            if b.ndim == 2 and a.ndim == 1:
                b = b.ravel()
        return a + b
    

def safe_sparse_inner_prod(a, b):
    if scipy.sparse.issparse(a) and scipy.sparse.issparse(b):
        if a.ndim == 2 and a.shape[1] == b.shape[0]:
            return (a @ b)[0, 0]
        if a.shape[0] == b.shape[0]:
            return (a.T @ b)[0, 0]
        return (a @ b.T)[0, 0]
    if scipy.sparse.issparse(a):
        a = a.toarray()
    elif scipy.sparse.issparse(b):
        b = b.toarray()
    return a @ b


def safe_sparse_multiply(a, b):
    if scipy.sparse.issparse(a) and scipy.sparse.issparse(b):
        return a.multiply(b)
    if scipy.sparse.issparse(a):
        a = a.toarray()
    elif scipy.sparse.issparse(b):
        b = b.toarray()
    return np.multiply(a, b)


def safe_sparse_norm(a, ord=None):
    if scipy.sparse.issparse(a):
        return scipy.sparse.linalg.norm(a, ord=ord)
    return np.linalg.norm(a, ord=ord)
