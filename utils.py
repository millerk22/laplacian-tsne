import numpy as np
from sklearn.metrics.pairwise import  pairwise_distances

def opt_bisection(f, xmin=0.0, xmax=10.0, tol=1e-4, itermax=1000, debug=False):
    r"""
    Function to find root of $f(x)$ via bisection method. Default settings of left and right endpoints
    are with the entropy-perplexity smoothing in mind.
    """

    other_iters = 0
    assert f(xmin) > 0.0
    while f(xmax) > 0.0:
        if f(xmin)*f(xmax) < 0:
            raise ValueError(f"f(xmin) = {f(xmin)} (should be > 0) and f(xmax) ={f(xmax)} (want to be < 0)")
        xmax *= 10
        other_iters += 1
        if other_iters > 10:
            break
    
    if other_iters != 0 and debug:
        print(other_iters)
        
    it = 0
    if abs(f(xmin)) <= tol:
        if debug: print(it)
        return xmin 
    if abs(f(xmax)) <= tol:
        if debug: print(it)
        return xmax
    
    
    while it <= itermax:
        c = 0.5*(xmin + xmax)
        if abs(f(c)) <= tol:
            if debug: print(it)
            return c 
        if f(xmin)*f(c) > 0.0:
            xmin = c 
        else:
            xmax = c 
        it += 1
    print(f"------- WARNING: Bisection did not converge in {itermax} iterations to tol = {tol}, current err = {abs(f(c))} -------")
    return None


def _rpcholesky(X_, k, tol=1e-5, returnG=False):
    n = X_.shape[0]
    rng = np.random.default_rng()
    
    diags = np.ones(n).astype(float)
    orig_trace = float(n) # for the kernel k(x_i, x_j) = 1/(1 + ||x_i - x_j||^2)

    landmarks = []
    G = np.zeros((k,n))

    for i in range(k):
        idx = rng.choice(range(n), p=diags/diags.sum())
        landmarks.append(idx)  # add this pivot to the landmark set
        idx_row = 1. / (1. + pairwise_distances(X_, X_[idx,:].reshape(1, -1), metric="euclidean", squared=True)).flatten()
        G[i,:] = (idx_row - G[:i, idx].T @ G[:i,:]) / np.sqrt(diags[idx])
        diags -= G[i,:]**2.
        diags = diags.clip(min=0)

        if tol > 0.0 and diags.sum() <= tol*orig_trace:
            G = G[:i,:]
            break

    if returnG:
        return landmarks, G
    
    return landmarks


